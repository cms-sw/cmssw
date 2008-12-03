#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/DetId/interface/DetId.h"
// hope not... #include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "TGeoManager.h"
#include "TFile.h"
#include "TTree.h"
#include "TEveGeoNode.h"
#include "TEveTrans.h"
#include "TColor.h"
#include "TROOT.h"
#include <iostream>
#include <sstream>
#include "TPRegexp.h"
DetIdToMatrix::~DetIdToMatrix()
{
   // ATTN: not sure I own the manager
   // if ( manager_ ) delete manager_;
}

void DetIdToMatrix::loadGeometry(const char* fileName)
{
   // ATTN: not sure if I can close the file and keep the manager in the memory
   //       it's not essential for id to matrix functionality, but the geo manager
   //       should be available for access to the geometry if it's needed
   TFile* f = new TFile(fileName);
   manager_ = 0;
   // load geometry
   manager_ = (TGeoManager*)f->Get("cmsGeo");
   if (!manager_) {
      std::cout << "ERROR: cannot find geometry in the file. Initialization failed." << std::endl;
      return;
   }
}

void DetIdToMatrix::loadMap(const char* fileName)
{
   if (!manager_) {
      std::cout << "ERROR: CMS detector geometry is not available. DetId to Matrix map Initialization failed." << std::endl;
      return;
   }

   TFile f(fileName);
   // ATTN: not sure who owns the object
   TTree* tree = (TTree*)f.Get("idToGeo");
   if (!tree) {
      std::cout << "ERROR: cannot find detector id map in the file. Initialization failed." << std::endl;
      return;
   }
   unsigned int id;
   char path[1000];
   tree->SetBranchAddress("id",&id);
   tree->SetBranchAddress("path",&path);
   for ( unsigned int i = 0; i < tree->GetEntries(); ++i) {
      tree->GetEntry(i);
      idToPath_[id] = path;
   }
   f.Close();
}

const TGeoHMatrix* DetIdToMatrix::getMatrix( unsigned int id ) const
{
   std::map<unsigned int, TGeoHMatrix>::const_iterator itr = idToMatrix_.find(id);
   if ( itr != idToMatrix_.end() ) return &(itr->second);

   const char* path = getPath( id );
   if ( ! path ) return 0;
   if ( ! manager_->cd(path) ) {
      std::cout << "ERROR: incorrect path " << path << "\nfor DetId: " << id << std::endl;
      return 0;
   }

   // CSC chamber frame has local coordinates rotated with respect
   // to the reference framed used in the offline reconstruction
   // -z is endcap is also reflected
   static const TGeoRotation inverseCscRotation("iCscRot",0,90,0);

   DetId detId(id);
   if ( detId.subdetId() == MuonSubdetId::CSC ) {
      TGeoHMatrix m = (*(manager_->GetCurrentMatrix()))*inverseCscRotation;
      if ( m.GetTranslation()[2]<0 ) m.ReflectX(kFALSE);
      idToMatrix_[id] = m;
      return &idToMatrix_[id];
   } else if ( detId.subdetId() == MuonSubdetId::RPC ) {
     RPCDetId rpcid(detId);
      // std::cout << "id: " << detId.rawId() << std::endl;
     if ( rpcid.region() == -1 || rpcid.region() == 1 ) {
       // std::cout << "before: " << std::endl;
       // (*(manager_->GetCurrentMatrix())).Print();
       TGeoHMatrix m = (*(manager_->GetCurrentMatrix()))*inverseCscRotation;
       if ( rpcid.region() == 1 ) m.ReflectY(kFALSE);
       idToMatrix_[id] = m;
       // std::cout << "after: " << std::endl;
       // m.Print();
       return &idToMatrix_[id];
     }
      /* else {
	std::cout << "BARREL station: " << rpcid.station() << std::endl;
	(*(manager_->GetCurrentMatrix())).Print();
       }
      */
   }
   TGeoHMatrix m = *(manager_->GetCurrentMatrix());

   // some ECAL crystall are reflected
   if ( detId.det() == DetId::Ecal && m.IsReflection() ) m.ReflectX(kFALSE);

   idToMatrix_[id] = m;
   return &idToMatrix_[id];
}

const char* DetIdToMatrix::getPath( unsigned int id ) const
{
   std::map<unsigned int, std::string>::const_iterator itr = idToPath_.find(id);
   if ( itr != idToPath_.end() )
     return itr->second.c_str();
   else
     return 0;
}

const TGeoVolume* DetIdToMatrix::getVolume( unsigned int id ) const
{
   std::map<unsigned int, std::string>::const_iterator itr = idToPath_.find(id);
   if ( itr != idToPath_.end() ) {
      manager_->cd(itr->second.c_str());
      return manager_->GetCurrentVolume();
   }
   else
     return 0;
}

std::vector<unsigned int> DetIdToMatrix::getAllIds() const
{
   std::vector<unsigned int> ids;
   for ( std::map<unsigned int, std::string>::const_iterator itr = idToPath_.begin(); itr != idToPath_.end(); ++itr )
     ids.push_back( itr->first );
   return ids;
}

std::vector<unsigned int> DetIdToMatrix::getMatchedIds( const char* regular_expression ) const
{
   std::vector<unsigned int> ids;
   TPRegexp regexp( regular_expression );
   for ( std::map<unsigned int, std::string>::const_iterator itr = idToPath_.begin(); itr != idToPath_.end(); ++itr )
     if ( regexp.MatchB(itr->second) ) ids.push_back( itr->first );
   return ids;
}


TEveGeoShape* DetIdToMatrix::getShape(const char* path, const char* name, const TGeoMatrix* matrix /* = 0 */) const
{
   if ( ! manager_ || ! path || ! name ) return 0;
   manager_->cd(path);
   // it's possible to get a corrected matrix from outside
   // if it's not provided, we take whatever the geo manager has
   if ( ! matrix ) matrix = manager_->GetCurrentMatrix();

   TEveGeoShape* shape = new TEveGeoShape(name,path);
   shape->SetTransMatrix(*matrix);
   TGeoShape* gs = manager_->GetCurrentVolume()->GetShape();
   UInt_t id = TMath::Max(gs->GetUniqueID(), UInt_t(1));
   gs->SetUniqueID(id);
   shape->SetShape(gs);
   TGeoVolume* volume = manager_->GetCurrentVolume();
   shape->SetMainColor(volume->GetLineColor());
   shape->SetRnrSelf(kTRUE);
   shape->SetRnrChildren(kTRUE);
   return shape;
}

TEveGeoShape* DetIdToMatrix::getShape( unsigned int id,
				       bool corrected /* = false */ ) const
{
   std::ostringstream s;
   s << id;
   if ( corrected )
     return getShape( getPath(id), s.str().c_str(), getMatrix(id) );
   else
     return getShape( getPath(id), s.str().c_str() );
}

TEveElementList* DetIdToMatrix::getAllShapes(const char* elementListName /*= "CMS"*/) const
{
   TEveElementList* container = new TEveElementList(elementListName );
   for ( std::map<unsigned int, std::string>::const_iterator itr = idToPath_.begin(); itr != idToPath_.end(); ++itr )
     container->AddElement( getShape(itr->first) );
   return container;
}

