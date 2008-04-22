#include "DQM/L1TMonitorClient/interface/L1TDTTPGClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TRandom.h"

#include <TF1.h>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <TProfile.h>
#include <TProfile2D.h>

using namespace edm;
using namespace std;

L1TDTTPGClient::L1TDTTPGClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

L1TDTTPGClient::~L1TDTTPGClient(){
 LogInfo("TriggerDQM")<<"[TriggerDQM]: ending... ";
}

//--------------------------------------------------------
void L1TDTTPGClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  monitorName_ =
  parameters_.getUntrackedParameter<string>("monitorName","");
  cout << "Monitor name = " << monitorName_ << endl;
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  LogInfo( "TriggerDQM");

      
}

//--------------------------------------------------------
void L1TDTTPGClient::beginJob(const EventSetup& context){

  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  // do your thing
  dbe_->setCurrentFolder(monitorName_);


// booking
//  test_ =
//    dbe_->book1D("test","test",100,0,100);
}

//--------------------------------------------------------
void L1TDTTPGClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void L1TDTTPGClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
   // clientHisto->Reset();
}
//--------------------------------------------------------

void L1TDTTPGClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){
			  
}			  
//--------------------------------------------------------
void L1TDTTPGClient::analyze(const Event& e, const EventSetup& context){
   cout << "L1TDTTPGClient::analyze" << endl;
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;


}

//--------------------------------------------------------
void L1TDTTPGClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void L1TDTTPGClient::endJob(){
}


TH1F * L1TDTTPGClient::get1DHisto(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogInfo("TriggerDQM") << "ME NOT FOUND.";
    return NULL;
  }

  return me_->getTH1F();
}

TH2F * L1TDTTPGClient::get2DHisto(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogInfo("TriggerDQM") << "ME NOT FOUND.";
    return NULL;
  }

  return me_->getTH2F();
}



TProfile2D * L1TDTTPGClient::get2DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogInfo("TriggerDQM") << "ME NOT FOUND.";
    return NULL;
  }

  return me_->getTProfile2D();
}


TProfile * L1TDTTPGClient::get1DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogInfo("TriggerDQM") << "ME NOT FOUND.";
    return NULL;
  }

  return me_->getTProfile();
}






