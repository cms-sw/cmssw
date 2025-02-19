#include "DQM/L1TMonitorClient/interface/L1TCSCTFClient.h"

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

L1TCSCTFClient::L1TCSCTFClient(const edm::ParameterSet& ps){
  parameters=ps;
  initialize();
}

L1TCSCTFClient::~L1TCSCTFClient(){}

//--------------------------------------------------------
void L1TCSCTFClient::initialize(){
  counterLS  = 0;
  counterEvt = 0;

  // get back-end interface
  dbe = Service<DQMStore>().operator->();

  input_dir   = parameters.getUntrackedParameter<string>("input_dir","");
  output_dir  = parameters.getUntrackedParameter<string>("output_dir","");
  prescaleLS  = parameters.getUntrackedParameter<int>("prescaleLS",-1);
  prescaleEvt = parameters.getUntrackedParameter<int>("prescaleEvt",-1);

  m_runInEventLoop = parameters.getUntrackedParameter<bool>("runInEventLoop", false);
  m_runInEndLumi = parameters.getUntrackedParameter<bool>("runInEndLumi", false);
  m_runInEndRun = parameters.getUntrackedParameter<bool>("runInEndRun", false);
  m_runInEndJob = parameters.getUntrackedParameter<bool>("runInEndJob", false);

}

//--------------------------------------------------------
void L1TCSCTFClient::beginJob(void){
  // get backendinterface
  dbe = Service<DQMStore>().operator->();

  // do your thing
  dbe->setCurrentFolder(output_dir);
  csctferrors_ = dbe->book1D("csctferrors_","CSCTF Errors",6,0,6);
  dbe->setCurrentFolder(input_dir);
}

//--------------------------------------------------------
void L1TCSCTFClient::beginRun(const Run& r, const EventSetup& context) {}

//--------------------------------------------------------
void L1TCSCTFClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void L1TCSCTFClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){

    if (m_runInEndLumi) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TCSCTFClient::analyze(const Event& e, const EventSetup& context){

   counterEvt++;
   if (prescaleEvt<1) return;
   if (prescaleEvt>0 && counterEvt%prescaleEvt!=0) return;
   
   // there is no loop on events in the offline harvesting step
   // code here will not be executed offline

   if (m_runInEventLoop) {

       processHistograms();
   }

}

//--------------------------------------------------------
void L1TCSCTFClient::endRun(const Run& r, const EventSetup& context) {

    if (m_runInEndRun) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TCSCTFClient::endJob(void){

    if (m_runInEndJob) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TCSCTFClient::processHistograms() {

    dbe->setCurrentFolder(input_dir);

    vector<string> meVec = dbe->getMEs();
    for(vector<string>::const_iterator it=meVec.begin(); it!=meVec.end(); it++){
      string full_path = input_dir + "/" + (*it);
      MonitorElement *me =dbe->get(full_path);
      if( !me ){
         LogInfo("TriggerDQM")<<full_path<<" NOT FOUND.";
         continue;
      }

      //  But for now we only do a simple workaround
      if( (*it) != "CSCTF_errors" ) continue;
      TH1F *errors = me->getTH1F();
      csctferrors_->getTH1F()->Reset();
      if(!errors) continue;
      for(int bin=1; bin<=errors->GetXaxis()->GetNbins(); bin++)
         csctferrors_->Fill(bin-0.5,errors->GetBinContent(bin));
    }

}

