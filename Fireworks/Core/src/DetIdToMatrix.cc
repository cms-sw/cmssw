#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/DetId/interface/DetId.h"
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
#include "TSystem.h"
#include "TGeoArb8.h"
#include "TEveVSDStructs.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
DetIdToMatrix::~DetIdToMatrix()
{
   // ATTN: not sure I own the manager
   // if ( manager_ ) delete manager_;
}

TFile* DetIdToMatrix::findFile(const char* fileName)
{
   TString file;
   if ( fileName[0] == '/')
   {
      file= fileName;
   }
   else
   {
      if ( const char* cmspath = gSystem->Getenv("CMSSW_BASE") ) {
         file += cmspath;
         file += "/";
      }
      file += fileName;
   }
   if ( !gSystem->AccessPathName(file.Data()) ) {
      return TFile::Open(file);
   }

   const char* searchpath = gSystem->Getenv("CMSSW_SEARCH_PATH");
   if ( searchpath == 0 ) return 0;
   TString paths(searchpath);
   TObjArray* tokens = paths.Tokenize(":");
   for ( int i=0; i<tokens->GetEntries(); ++i ) {
      TObjString* path = (TObjString*)tokens->At(i);
      TString fullFileName(path->GetString());
      fullFileName += "/Fireworks/Geometry/data/";
      fullFileName += fileName;
      if ( !gSystem->AccessPathName(fullFileName.Data()) )
         return TFile::Open(fullFileName.Data());
   }
   return 0;
}

void DetIdToMatrix::loadGeometry(const char* fileName)
{
   manager_ = 0;
   if ( TFile* f = findFile(fileName) ) {
      // load geometry
      manager_ = (TGeoManager*)f->Get("cmsGeo");
      f->Close();
   } else {
      std::cout << "ERROR: failed to find geometry file. Initialization failed." << std::endl;
      return;
   }
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

   TFile* f = findFile(fileName);
   if ( !f )  {
      std::cout << "ERROR: failed to find geometry file. Initialization failed." << std::endl;
      return;
   }
   TTree* tree = (TTree*)f->Get("idToGeo");
   if (!tree) {
      std::cout << "ERROR: cannot find detector id map in the file. Initialization failed." << std::endl;
      return;
   }
   unsigned int id;
   char path[1000];
   Float_t points[24];
   bool loadPoints = tree->GetBranch("points")!=0;
   tree->SetBranchAddress("id",&id);
   tree->SetBranchAddress("path",&path);
   if (loadPoints) tree->SetBranchAddress("points",&points);
   for ( unsigned int i = 0; i < tree->GetEntries(); ++i) {
      tree->GetEntry(i);
      idToPath_[id] = path;
      if (loadPoints) {
         std::vector<TEveVector> p(8);
         for(unsigned int j=0; j<8; ++j) p[j].Set(points[3*j],points[3*j+1],points[3*j+2]);
         idToPoints_[id] = p;
      }
   }
   f->Close();
}

const TGeoHMatrix* DetIdToMatrix::getMatrix( unsigned int id ) const
{
   std::map<unsigned int, TGeoHMatrix>::const_iterator itr = idToMatrix_.find(id);
   if ( itr != idToMatrix_.end() ) return &(itr->second);

   const char* path = getPath( id );
   if ( !path ) return 0;
   if ( !manager_->cd(path) ) {
      std::cout << "ERROR: incorrect path " << path << "\nfor DetId: " << id << std::endl;
      return 0;
   }

   // CSC chamber frame has local coordinates rotated with respect
   // to the reference framed used in the offline reconstruction
   // -z is endcap is also reflected
   static const TGeoRotation inverseCscRotation("iCscRot",0,90,0);

   DetId detId(id);
   if (detId.det() == DetId::Muon) {
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
   if ( !manager_ || !path || !name ) return 0;
   manager_->cd(path);
   // it's possible to get a corrected matrix from outside
   // if it's not provided, we take whatever the geo manager has
   if ( !matrix ) matrix = manager_->GetCurrentMatrix();

   TEveGeoShape* shape = new TEveGeoShape(name,path);
   shape->SetTransMatrix(*matrix);
   TGeoShape* gs = manager_->GetCurrentVolume()->GetShape();
   //------------------------------------------------------------------------------//
   // FIXME !!!!!!!!!!!!!!
   // hack zone to make CSC complex shape visible
   // loop over bool shapes till we get something non-composite on the left side.
   if ( TGeoCompositeShape* composite = dynamic_cast<TGeoCompositeShape*>(gs) ){
     int depth = 0;
     TGeoShape* nextShape(gs);
     do {
       if ( depth > 10 ) break;
       nextShape = composite->GetBoolNode()->GetLeftShape();
       composite = dynamic_cast<TGeoCompositeShape*>(nextShape);
       ++depth;
     } while ( depth<10 && composite!=0 );
     if ( composite == 0 ) gs = nextShape;
   }
   //------------------------------------------------------------------------------//
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

std::vector<TEveVector> DetIdToMatrix::getPoints(unsigned int id) const
{
   // reco geometry points
   std::map<unsigned int, std::vector<TEveVector> >::const_iterator points = idToPoints_.find(id);
   if ( points == idToPoints_.end() ) {
      printf("Warning: no reco geometry is found for id: %d\n", id);
      return std::vector<TEveVector>();
   } else {
      return points->second;
   }
}

