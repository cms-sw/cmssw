/*
 *  $Date: 2012/05/18 09:23:50 $
 *  $Revision: 1.1 $
 *  \author N. Srimanobhas
 */

#include "DQMOffline/Trigger/interface/HLTInclusiveVBFClient.h"

using namespace std;
using namespace edm;


HLTInclusiveVBFClient::HLTInclusiveVBFClient( const edm::ParameterSet& iConfig ):conf_(iConfig) {
  
  //
  dbe_ = Service<DQMStore>().operator->();

  //
  if (!dbe_){
    edm::LogError("HLTInclusiveVBFClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  
  //
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if(dbe_) dbe_->setVerbose(0);
  }
 
  //
  debug_ = false;
  verbose_ = false;

  //
  processname_ = iConfig.getParameter<std::string>("processname");

  //
  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  if (debug_) std::cout << hltTag_ << std::endl;
  
  //
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
  if(dbe_) dbe_->setCurrentFolder(dirName_);
}


HLTInclusiveVBFClient::~HLTInclusiveVBFClient() {

}


void HLTInclusiveVBFClient::beginJob() {

}


void HLTInclusiveVBFClient::beginRun(const edm::Run& r, const edm::EventSetup& context) {

}


void HLTInclusiveVBFClient::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {

}


void HLTInclusiveVBFClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

}


void HLTInclusiveVBFClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
  runClient_();
}


void HLTInclusiveVBFClient::endRun(const edm::Run& r, const edm::EventSetup& context) {
}


void HLTInclusiveVBFClient::endJob() {

}

void HLTInclusiveVBFClient::runClient_() {
  
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);

  LogDebug("HLTInclusiveVBFClient") << "runClient" << std::endl;
  if (debug_) std::cout << "runClient" << std::endl; 

}
