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
  parameters=ps;
  initialize();
}

HLTMonMuonClient::~HLTMonMuonClient(){}

//--------------------------------------------------------
void HLTMonMuonClient::initialize(){
  //counterLS  = 0;
  //counterEvt = 0;

  // get back-end interface
  //dbe = Service<DQMStore>().operator->();

  //input_dir   = parameters.getUntrackedParameter<string>("input_dir","");
  //output_dir  = parameters.getUntrackedParameter<string>("output_dir","");
  //prescaleLS  = parameters.getUntrackedParameter<int>("prescaleLS",-1);
  //prescaleEvt = parameters.getUntrackedParameter<int>("prescaleEvt",-1);
}

//--------------------------------------------------------
void HLTMonMuonClient::beginJob(const EventSetup& context){
  // get backendinterface
  dbe = Service<DQMStore>().operator->();

  // do your thing
  dbe->setCurrentFolder(output_dir);
  //hltmonmuonerrors_ = dbe->book1D("hltmonmuonerrors_","HLTMonMuon Errors",6,0,6);
  dbe->setCurrentFolder(input_dir);
}

//--------------------------------------------------------
void HLTMonMuonClient::beginRun(const Run& r, const EventSetup& context) {}

//--------------------------------------------------------
void HLTMonMuonClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void HLTMonMuonClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){
/*
   vector<string> meVec = dbe->getMEs();
   for(vector<string>::const_iterator it=meVec.begin(); it!=meVec.end(); it++){
     string full_path = input_dir + "/" + (*it);
     MonitorElement *me =dbe->get(full_path);
     if( !me ){
        LogInfo("TriggerDQM")<<full_path<<" NOT FOUND.";
        continue;
     }

     // The commented code below requires to be developed to support QT framework---------------------------

     std::vector<QReport *> Qtest_map = me->getQReports();
     for(std::vector<QReport *>::const_iterator it=Qtest_map.begin(); it!=Qtest_map.end(); it++){
        string qt_name   = (*it)->getQRName();
        int    qt_status = (*it)->getStatus();

        switch(qt_status){
           case dqm::qstatus::WARNING:    break;
           case dqm::qstatus::ERROR:      break;
           case dqm::qstatus::DISABLED:   break;
           case dqm::qstatus::INVALID:    break;
           case dqm::qstatus::INSUF_STAT: break;
           default: break;
        }

//   get bad channel list
        std::vector<dqm::me_util::Channel> badChannels=(*it)->getBadChannels();
        for(vector<dqm::me_util::Channel>::iterator badchsit=badChannels.begin(); badchsit!=badChannels.end(); badchsit++){
           int ix = badchsit->getBinX();
           int iy = badchsit->getBinY();
           (*badchsit).getContents();
        }

     }
  //------------------------------------------------------------
     //  But for now we only do a simple workaround
     if( (*it) != "CSCTF_errors" ) continue;
     TH1F *errors = me->getTH1F();
     csctferrors_->getTH1F()->Reset();
     if(!errors) continue;
     for(int bin=1; bin<=errors->GetXaxis()->GetNbins(); bin++)
        csctferrors_->Fill(bin-0.5,errors->GetBinContent(bin));
   }
*/
}

//--------------------------------------------------------
void HLTMonMuonClient::analyze(const Event& e, const EventSetup& context){
/* 
   counterEvt++;
   if (prescaleEvt<1) return;
   if (prescaleEvt>0 && counterEvt%prescaleEvt!=0) return;

   // The code below duplicates one from endLuminosityBlock function
   vector<string> meVec = dbe->getMEs();
   for(vector<string>::const_iterator it=meVec.begin(); it!=meVec.end(); it++){
     string full_path = input_dir + "/" + (*it);
     MonitorElement *me =dbe->get(full_path);
     if( !me ){
        LogError("TriggerDQM")<<full_path<<" NOT FOUND.";
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
*/
}

//--------------------------------------------------------
void HLTMonMuonClient::endRun(const Run& r, const EventSetup& context){}

//--------------------------------------------------------
void HLTMonMuonClient::endJob(void){}

