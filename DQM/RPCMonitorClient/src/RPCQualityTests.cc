// -*- C++ -*-
//
// Package:    RPCQualityTests
// Class:      RPCQualityTests
// 
/**\class RPCQualityTests RPCQualityTests.cc DQM/RPCQualityTests/src/RPCQualityTests.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Anna Cimmino
//         Created:  Wed Mar  5 20:43:10 CET 2008
// $Id: RPCQualityTests.cc,v 1.1 2008/04/25 14:32:40 cimmino Exp $
//
//

#include "DQM/RPCMonitorClient/interface/RPCQualityTests.h"
#include  "DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h"
#include "DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h"

//DQMServices
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;
using namespace std;

/////////////////REMEMBER TO PUT THE LOG MESSAGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

RPCQualityTests::RPCQualityTests(const ParameterSet& iConfig)

{
  parameters_ = iConfig;
  
  //check enabling
  enableQTests_ =parameters_.getUntrackedParameter<bool> ("EnableQualityTests",true); 
  enableMonitorDaemon_ =parameters_.getUntrackedParameter<bool> ("EnableMonitorDeamon",false); 


  //get prescale factor
  prescaleFactor_ = parameters_.getUntrackedParameter<int>("diagnosticPrescale", 1);

  // DQM Client name
  clientName_ = parameters_.getUntrackedParameter<string>("ClientName", "RPCMonitorClient");
  if ( enableMonitorDaemon_ ) {
    // DQM Collector hostname
    hostName_ = parameters_.getUntrackedParameter<string>("HostName", "localhost");
    // DQM Collector port
    hostPort_ = parameters_.getUntrackedParameter<int>("HostPort", 9090);
  }

  //get dqm input file
  enableInputFile_ =parameters_.getUntrackedParameter<bool> ("EnableInputFile",false);
  if (enableInputFile_)  hostName_ = parameters_.getUntrackedParameter<string>("DQMInputFile", "DQM.root");

  //get event info folder (remenber: this module must run after EventInfo!!!!
  eventInfoPath_ = parameters_.getUntrackedParameter<string>("EventInfoPath", "RPC/EventInfo");

  //get qtest list  
  qtestList_.push_back("DeadChannelTest");
  qtestList_.push_back("MultiplicityTest");
  qtestList_= parameters_.getUntrackedParameter<std::vector<std::string> >("QualityTestList",qtestList_);
  

  map<string , RPCClient*> qtestMap= makeQTestMap();

  //loop on qTest list and get the appropiate qTest modules
  for(map<string, RPCClient *>::iterator it = qtestMap.begin();it!= qtestMap.end(); it++){
    if(find(qtestList_.begin(),qtestList_.end(),(*it).first)!=qtestList_.end()) qtests_.push_back((*it).second);
  }//end loop on qTest list

}


RPCQualityTests::~RPCQualityTests(){
  
 LogVerbatim ("QualityTests") << "[RPCQualityTests]: Called Destructor";
 /*  for ( unsigned int i=0; i<qtests_.size(); i++ ) {
    delete qtests_[i];
  }

  if ( enableMonitorDaemon_ ) delete mui_;*/
}


// ------------ method called once each job just before starting event loop  ------------
void RPCQualityTests::beginJob(const EventSetup& iSetup){

  if (!enableQTests_) return;                 ;

  edm::LogVerbatim ("QualityTests") << "[RPCQualityTests]: Begin job"<<qtests_.size();

  nevents_=0;
  
  if ( enableMonitorDaemon_ ) {
    
    // start DQM user interface instance
    // will attempt to reconnect upon connection problems (w/ a 5-sec delay)  
    mui_ = new DQMOldReceiver(hostName_, hostPort_, clientName_, 5);
    dbe_ = mui_->getBEInterface();
    
  } else {
    
    // get hold of back-end interface  
    mui_ = 0;
    dbe_ = Service<DQMStore>().operator->();
   
    //get histos from file
    if ( inputFile_.size() != 0 && dbe_ )   dbe_->open(inputFile_); 
  }
 
  dbe_->setVerbose(1);

   
  //Do whatever the begin jobs of all qtest modules do
  for ( unsigned int i=0; i<qtests_.size(); i++ ) {
    qtests_[i]->beginJob(dbe_);
  }
}



// begin run
void  RPCQualityTests::beginRun(const Run& r, const EventSetup& c){
  
  if (!enableQTests_) return;
  
  for ( unsigned int i=0; i<qtests_.size(); i++ ) {
    qtests_[i]->beginRun(r,c);
  }
  
}

/// Begin Lumi block method
void RPCQualityTests::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  if (!enableQTests_) return;

  for ( unsigned int i=0; i<qtests_.size(); i++ ) {
   qtests_[i]->beginLuminosityBlock(lumiSeg,context);
  }
}



// ------------ method called to for each event  ------------
void RPCQualityTests::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  //   using namespace edm;
 if (!enableQTests_) return;
  nevents_++;


  for ( unsigned int i=0; i<qtests_.size(); i++ ) {
   qtests_[i]->analyze( iEvent,iSetup);
  }
}

/// End Lumi Block method
void RPCQualityTests::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c){
  
  if (!enableQTests_ ||lumiSeg.id().luminosityBlock() % prescaleFactor_ != 0 ) return;

  for ( unsigned int i=0; i<qtests_.size(); i++ ) {
   qtests_[i]->endLuminosityBlock( lumiSeg, c);
  }

}
//


//end run 
void  RPCQualityTests::endRun(const Run& r, const EventSetup& c){

 if (!enableQTests_) return;
  for ( unsigned int i=0; i<qtests_.size(); i++ ) {
   qtests_[i]->endRun(r,c);
  }
}


//end job
void RPCQualityTests::endJob() {
 if (!enableQTests_) return;
}






//private methods
map<std::string , RPCClient*> RPCQualityTests::makeQTestMap() {


  map<std::string , RPCClient*> qtmap;

  qtmap["DeadChannelTest"] = new RPCDeadChannelTest(parameters_);
  qtmap["MultiplicityTest"] = new RPCMultiplicityTest(parameters_);
  return qtmap;

}
