#include <TFile.h>
#include <TTree.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/EventBasicData.h"

EventBasicData::EventBasicData( const edm::ParameterSet &roCONFIG):
  // Read configuration
  oOFileName_( 
    roCONFIG.getUntrackedParameter<std::string>( 
      "oOFileName", 
      "EventBasicData_out.root")) {
}

// +---------------------------------------------------------------------------
// |  PROTECTED
// +---------------------------------------------------------------------------
void EventBasicData::beginJob( const edm::EventSetup &roEVENT_SETUP) {
  // Reopen output file
  // [Note: object will be destroyed automatically due to std::auto_ptr<...>]
  poOFile_ = std::auto_ptr<TFile>( new TFile( oOFileName_.c_str(), "RECREATE"));

  // Create General Tree and reserve leafs in it
  // [Note: object will be destroyed automatically once Output file is closed]
  poGenTree_ = new TTree( "GenTree", "General Tree");
  poGenTree_->Branch( "nRun",      &oGenVal_.nRun,      "nRun/I");
  poGenTree_->Branch( "nLclEvent", &oGenVal_.nLclEvent, "nLclEvent/I");
  poGenTree_->Branch( "nLTime",    &oGenVal_.nLTime,    "nLTime/I");
}

void EventBasicData::endJob() {
  // Write buffers and Close opened file
  poOFile_->Write();
  poOFile_->Close();
}

void EventBasicData::analyze( const edm::Event      &roEVENT,
                                    const edm::EventSetup &roEVENT_SETUP) {

  // Extract basic parameters for trees
  oGenVal_.nRun      = roEVENT.id().run();
  oGenVal_.nLclEvent = roEVENT.id().event();
  oGenVal_.nLTime    = roEVENT.time().value();

  poGenTree_->Fill();
}
