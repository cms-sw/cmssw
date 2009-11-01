#include "DQM/HLTEvF/interface/HLTMonMuonClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

HLTMonMuonClient::HLTMonMuonClient(const edm::ParameterSet& ps){
 
  //indir_   = ps.getUntrackedParameter<string>("input_dir","HLT/HLTMonMuon/Summary");
  //outdir_  = ps.getUntrackedParameter<string>("output_dir","HLT/HLTMonMuon/Client");    

  dbe_ = NULL;
  //if (ps.getUntrackedParameter < bool > ("DQMStore", false)) {
  dbe_ = Service < DQMStore > ().operator->();
  dbe_->setVerbose(0);
    //}

  //if (dbe_ != NULL) {
  //  dbe_->setCurrentFolder(outdir_);
  //}

}

HLTMonMuonClient::~HLTMonMuonClient(){}

//--------------------------------------------------------
void HLTMonMuonClient::beginJob(const EventSetup& context){
  
}

//--------------------------------------------------------
void HLTMonMuonClient::beginRun(const Run& r, const EventSetup& context) {
 

}

//--------------------------------------------------------
void HLTMonMuonClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void HLTMonMuonClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){

}

//--------------------------------------------------------
void HLTMonMuonClient::analyze(const Event& e, const EventSetup& context){

}

//--------------------------------------------------------
void HLTMonMuonClient::endRun(const Run& r, const EventSetup& context){}

//--------------------------------------------------------
void HLTMonMuonClient::endJob(void){}

