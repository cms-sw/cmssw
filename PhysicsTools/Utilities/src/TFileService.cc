#include "PhysicsTools/Utilities/interface/TFileService.h"
using namespace std;

TFileService::TFileService( const edm::ParameterSet & cfg, edm::ActivityRegistry & ) :
  file_( new TFile( cfg.getParameter<string>( "fileName" ).c_str() , "RECREATE" ) ) {
}

TFileService::~TFileService() {
  file_->Write();
  file_->Close();
  delete file_;
}
