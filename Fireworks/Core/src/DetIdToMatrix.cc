#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "TGeoManager.h"
#include "TFile.h"
#include "TTree.h"
#include <iostream>

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
      if ( manager_->cd(path) ) {
	 idToPath_[id] = path;
	 idToMatrix_[id] = *(manager_->GetCurrentMatrix());
      }
      else
	std::cout << "WARNING: incorrect path " << path << "\nSkipped DetId: " << id << std::endl;
   }
   f.Close();
}

const TGeoHMatrix* DetIdToMatrix::getMatrix( unsigned int id )
{
   std::map<unsigned int, TGeoHMatrix>::const_iterator itr = idToMatrix_.find(id);
   if ( itr != idToMatrix_.end() )
     return &(itr->second);
   else
     return 0;
}

const char* DetIdToMatrix::getPath( unsigned int id )
{
   std::map<unsigned int, std::string>::const_iterator itr = idToPath_.find(id);
   if ( itr != idToPath_.end() )
     return itr->second.c_str();
   else
     return 0;
}
   
const TGeoVolume* DetIdToMatrix::getVolume( unsigned int id )
{
   std::map<unsigned int, std::string>::const_iterator itr = idToPath_.find(id);
   if ( itr != idToPath_.end() ) {
      manager_->cd(itr->second.c_str());
      return manager_->GetCurrentVolume();
   }
   else
     return 0;
}
   
std::vector<unsigned int> DetIdToMatrix::getAllIds()
{
   std::vector<unsigned int> ids;
   for ( std::map<unsigned int, std::string>::const_iterator itr = idToPath_.begin(); itr != idToPath_.end(); ++itr )
     ids.push_back( itr->first );
   return ids;
}
   

  
