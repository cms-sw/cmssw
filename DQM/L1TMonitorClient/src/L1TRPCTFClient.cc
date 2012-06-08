#include "DQM/L1TMonitorClient/interface/L1TRPCTFClient.h"

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

L1TRPCTFClient::L1TRPCTFClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

L1TRPCTFClient::~L1TRPCTFClient(){
 LogInfo("TriggerDQM")<<"[TriggerDQM]: ending... ";
}

//--------------------------------------------------------
void L1TRPCTFClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","");
//  cout << "Monitor name = " << monitorName_ << endl;
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
//  cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
//  cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  output_dir_ = parameters_.getUntrackedParameter<string>("output_dir","");
//  cout << "DQM output dir = " << output_dir_ << endl;
  input_dir_ = parameters_.getUntrackedParameter<string>("input_dir","");
//  cout << "DQM input dir = " << input_dir_ << endl;
  
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose", false);
  
  m_runInEventLoop = parameters_.getUntrackedParameter<bool>("runInEventLoop", false);
  m_runInEndLumi = parameters_.getUntrackedParameter<bool>("runInEndLumi", false);
  m_runInEndRun = parameters_.getUntrackedParameter<bool>("runInEndRun", false);
  m_runInEndJob = parameters_.getUntrackedParameter<bool>("runInEndJob", false);


  LogInfo( "TriggerDQM");

      
}

//--------------------------------------------------------
void L1TRPCTFClient::beginJob(void){

  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";

  // get backendinterface
  dbe_ = Service<DQMStore>().operator->();  

  dbe_->setCurrentFolder(output_dir_);

  m_deadChannels = dbe_->book2D("RPCTF_deadchannels",
                                "RPCTF deadchannels",
                                33, -16.5, 16.5,
                                144,  -0.5, 143.5);
  m_noisyChannels =  dbe_->book2D("RPCTF_noisychannels",
                                "RPCTF noisy channels",
                                33, -16.5, 16.5,
                                144,  -0.5, 143.5);
}
//--------------------------------------------------------
void L1TRPCTFClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void L1TRPCTFClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
   // clientHisto->Reset();
}
//--------------------------------------------------------

void L1TRPCTFClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c)
{
   if (verbose_) std::cout <<  "L1TRPCTFClient::endLuminosityBlock" << std::endl;

   if (m_runInEndLumi) {

       processHistograms();
   }

}			  

//--------------------------------------------------------
void L1TRPCTFClient::analyze(const Event& e, const EventSetup& context) {
    //   cout << "L1TRPCTFClient::analyze" << endl;
    counterEvt_++;
    if (prescaleEvt_ < 1)
        return;
    if (prescaleEvt_ > 0 && counterEvt_ % prescaleEvt_ != 0)
        return;

    // there is no loop on events in the offline harvesting step
    // code here will not be executed offline

    if (m_runInEventLoop) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TRPCTFClient::endRun(const Run& r, const EventSetup& context){

    if (m_runInEndRun) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TRPCTFClient::endJob() {

    if (m_runInEndJob) {

        processHistograms();
    }

}


//--------------------------------------------------------
void L1TRPCTFClient::processHistograms() {

    dbe_->setCurrentFolder(input_dir_);

   {

     MonitorElement *me
         = dbe_->get( (input_dir_+"/RPCTF_muons_eta_phi_bx0").c_str() );

     if (me){
       const QReport *qreport;

       qreport = me->getQReport("DeadChannels_RPCTF_2D");
       if (qreport) {
         vector<dqm::me_util::Channel> badChannels = qreport->getBadChannels();
         for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin();
              channel != badChannels.end();
              ++channel)
         {
           m_deadChannels->setBinContent((*channel).getBinX(),
                                         (*channel).getBinY(),
                                         100);
         } // for(badchannels)
       } //if (qreport)

       qreport = me->getQReport("HotChannels_RPCTF_2D");
       if (qreport) {
         vector<dqm::me_util::Channel> badChannels = qreport->getBadChannels();
         for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin();
              channel != badChannels.end();
              ++channel)
         {
           // (*channel).getBinY() == 0 for NoisyChannels QTEST
           m_noisyChannels->setBinContent((*channel).getBinX(), 100);
         } // for(badchannels)
       } //if (qreport)
 //      else std::cout << "dupa" << std::endl;
     } // if (me)


   }


   if (verbose_)
   {
     std::vector<string> meVec = dbe_->getMEs();
     for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {

         std::string full_path = input_dir_ + "/" + (*it);
         MonitorElement * me =dbe_->get(full_path);

         // for this MEs, get list of associated QTs
         std::vector<QReport *> Qtest_map = me->getQReports();

         if (Qtest_map.size() > 0) {
           std::cout << "Test: " << full_path << std::endl;
           for (std::vector<QReport *>::const_iterator it = Qtest_map.begin();
                 it != Qtest_map.end();
                 ++it)
           {
               std::cout
                   << " Name "<< (*it)->getQRName()
                   << " Status " <<  (*it)->getStatus()
                   <<std::endl;

               std::vector<dqm::me_util::Channel> badChannels=(*it)->getBadChannels();

               vector<dqm::me_util::Channel>::iterator badchsit = badChannels.begin();
               while(badchsit != badChannels.end())
               {
                 int ix = (*badchsit).getBinX();
                 int iy = (*badchsit).getBinY();
                 std::cout << "(" << ix <<","<< iy << ") ";
                 ++badchsit;
               }
               std::cout << std::endl;

           }
         }

     } //
   }


}


