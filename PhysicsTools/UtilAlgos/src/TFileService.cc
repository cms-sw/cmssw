#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TFile.h"
using namespace edm;
using namespace std;

TFileService::TFileService( const ParameterSet & cfg, ActivityRegistry & r ) :
  TFileDirectory( "", "", new TFile( cfg.getParameter<string>( "fileName" ).c_str() , "RECREATE" ), "" ),
  file_( TFileDirectory::file_ ) {
  r.watchPreModuleConstruction( this, & TFileService::setDirectoryName ); 
  r.watchPreModule( this, & TFileService::setDirectoryName ); 
  r.watchPreModuleBeginJob( this, & TFileService::setDirectoryName ); 
}

TFileService::~TFileService() {
  file_->Write();
  file_->Close();
  delete file_;
}

void TFileService::setDirectoryName( const ModuleDescription & desc ) {
  dir_ = desc.moduleLabel_;
  descr_ = ( dir_ + " (" + desc.moduleName_ + ") folder" ).c_str();
}
