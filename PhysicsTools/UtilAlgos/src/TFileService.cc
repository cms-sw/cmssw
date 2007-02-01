#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
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
  file_( new TFile( cfg.getParameter<string>( "fileName" ).c_str() , "RECREATE" ) ),
  setcd_( true ) {
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
  setcd_ = true;
}

void TFileService::cd() const {
  if ( setcd_ ) {
    TDirectory * dir = file_->GetDirectory( currentModuleLabel_.c_str() );
    if ( dir == 0 )
      dir = file_->mkdir( currentModuleLabel_.c_str(), 
			  (currentModuleLabel_ + " (" + currentModulenName_ + ") folder" ).c_str() );
    if ( dir == 0 )   
      throw 
	cms::Exception( "InvalidDirectory" ) 
	  << "Can't create directory " << currentModuleLabel_;
    bool ok = file_->cd( currentModuleLabel_.c_str() );
    if ( ! ok )
      throw 
	cms::Exception( "InvalidDirectory" ) 
	  << "Can't change directory to newly created: " << currentModuleLabel_;
    setcd_ = false;
  } 
}

void TFileService::cd( const std::string & dirName ) const {
  setcd_ = true;
  cd();
  TDirectory * dir = file_->GetDirectory( currentModuleLabel_.c_str() );
  if ( dir == 0 )   
    throw 
      cms::Exception( "InvalidDirectory" ) 
	<< "Can't get current directory ";
  const char * name = dirName.c_str();
  dir = dir->mkdir( name );
  if ( dir == 0 ) {
    throw 
      cms::Exception( "InvalidDirectory" ) 
	<< "Can't create sub-directory " << name;
  }
  bool ok = file_->cd( ( currentModuleLabel_ + "/" + name ).c_str() );
  if ( ! ok )
    throw 
      cms::Exception( "InvalidDirectory" ) 
	<< "Can't change directory to newly created: " << name;
  setcd_ = false;
}
