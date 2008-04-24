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
  
  LogInfo( "TriggerDQM");

      
}

//--------------------------------------------------------
void L1TRPCTFClient::beginJob(const EventSetup& context){

  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";

  // get backendinterface
  dbe_ = Service<DQMStore>().operator->();  

  dbe_->setCurrentFolder(output_dir_);
  
  m_phipackedbad = dbe_->book1D("RPCTF_phi_valuepacked_bad",
                                "RPCTF bad channels in phipacked", 144, -0.5, 143.5 ) ;


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
   
//    dbe_->setCurrentFolder(output_dir_);
   dbe_->setCurrentFolder(input_dir_);
   std::vector<string> meVec = dbe_->getMEs();
   
   //std::string urrDir = dbe_->pwd();     
   
   /*dbe_->setCurrentFolder(output_dir_);
   std::vector<string> addVec = dbe_->getMEs();
   meVec.insert(meVec.end(), addVec.begin(), addVec.end());
   */
   
   
   for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      

      std::string full_path = input_dir_ + "/" + (*it);
      MonitorElement * me =dbe_->get(full_path);
      
      // for this MEs, get list of associated QTs
      std::vector<QReport *> Qtest_map = me->getQReports();
   
      if (Qtest_map.size() > 0) {
         if (verbose_) std::cout << "Test: " << full_path << std::endl;
         for (std::vector<QReport *>::const_iterator it = Qtest_map.begin(); 
              it != Qtest_map.end(); 
              ++it)
         {
            if (verbose_) std::cout 
                     << "   "<< (*it)->getQRName() 
                     << " " <<  (*it)->getStatus() 
                     <<std::endl;
            
            std::vector<dqm::me_util::Channel> badChannels=(*it)->getBadChannels();
            
            vector<dqm::me_util::Channel>::iterator badchsit = badChannels.begin();
            while(badchsit != badChannels.end())                           
            {                             
               int ix = (*badchsit).getBinX();     
               m_phipackedbad->setBinContent(ix,1);
               //int iy = (*badchsit).getBinY();  
               if (verbose_) std::cout << " " << ix;
               ++badchsit;
            }
            if (verbose_) std::cout << std::endl;
            
            
            
            
            
            
            
            
         }
      }
      
   } // 
   
}			  
//--------------------------------------------------------
void L1TRPCTFClient::analyze(const Event& e, const EventSetup& context){
//   cout << "L1TRPCTFClient::analyze" << endl;
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

   


}

//--------------------------------------------------------
void L1TRPCTFClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void L1TRPCTFClient::endJob(){
}




