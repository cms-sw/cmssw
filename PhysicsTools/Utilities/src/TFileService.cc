#include "PhysicsTools/Utilities/interface/TFileService.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TFolder.h"
#include "TROOT.h"
#include "TFile.h"
#include "FWCore/Utilities/interface/Exception.h"
using namespace std;
using namespace edm;

TFileService::TFileService( const ParameterSet & cfg, ActivityRegistry & r ) :
  file_( new TFile( cfg.getParameter<string>( "fileName" ).c_str() , "RECREATE" ) ) {
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
  currentModuleLabel_ = desc.moduleLabel_;
  currentModulenName_ = desc.moduleName_;
}

void TFileService::cd() const {
  TDirectory * dir = file_->GetDirectory( currentModuleLabel_.c_str() );
  if ( dir == 0 )
    dir = file_->mkdir( currentModuleLabel_.c_str(), 
			(currentModuleLabel_ + " (" + currentModulenName_ + ") folter" ).c_str() );
  if ( dir == 0 )   
    throw 
      cms::Exception( "InvalidDirectory" ) 
	<< "Can't create directory " << currentModuleLabel_;
  bool ok = file_->cd( currentModuleLabel_.c_str() );
  if ( ! ok )
    throw 
      cms::Exception( "InvalidDirectory" ) 
	<< "Can't change directory to newly created: " << currentModuleLabel_;
}
