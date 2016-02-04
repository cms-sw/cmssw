/*
 * \file DQMClientExample.cc
 * \author M. Zanetti - CERN
 *
 * Last Update:
 * $Date: 2009/12/14 22:22:23 $
 * $Revision: 1.16 $
 * $Author: wmtan $
 *
 */


/*  Description: Simple example showing how to access the histograms 
 *  already defined and filled by the DQM producer(source) 
 *  and how to access the quality test results.
 */

/* Jan 17, 2009: the code has been modified significantly
 * to steer the client operations  
 * Author: D.Volyanskyy
 */


#include "DQMServices/Examples/interface/DQMClientExample.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TF1.h>
#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

//==================================================================//
//================= Constructor and Destructor =====================//
//==================================================================//
DQMClientExample::DQMClientExample(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

DQMClientExample::~DQMClientExample(){
}

//==================================================================//
//======================= Initialise ===============================//
//==================================================================//
void DQMClientExample::initialize(){ 

  ////---- initialise Event and LS counters
  counterEvt_=0;   counterLS_  = 0; 
  counterClientOperation = 0;
  
  ////---- get DQM store interface
  dbe_ = Service<DQMStore>().operator->();
  
  ////---- define base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  cout << "DQMClientExample: Monitor name = " << monitorName_ << endl;
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;

  ////--- get steerable parameters
  prescaleLS_  = parameters_.getUntrackedParameter<int>("prescaleLS",  -1);
  cout << "DQMClientExample: DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "DQMClientExample: DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;

  //-- QTestName that is going to bu run on clienHisto, as defined in XML file
  QTestName_ = parameters_.getUntrackedParameter<string>("QTestName","exampleQTest");  
  cout << "DQMClientExample: QTest name to be ran on clientHisto = " << QTestName_ << endl;
  //-- define where to run the Client
  clientOnEachEvent  = parameters_.getUntrackedParameter<bool>("clientOnEachEvent",false);
  if(clientOnEachEvent) cout << "DQMClientExample: run Client on each event" << endl;
  clientOnEndLumi  = parameters_.getUntrackedParameter<bool>("clientOnEndLumi",true);
  if(clientOnEndLumi) cout << "DQMClientExample: run Client at the end of each lumi section" << endl;
  clientOnEndRun   = parameters_.getUntrackedParameter<bool>("clientOnEndRun",false);
  if(clientOnEndRun) cout << "DQMClientExample: run Client at the end of each run" << endl;
  clientOnEndJob   = parameters_.getUntrackedParameter<bool>("clientOnEndJob",false);
  if(clientOnEndJob) cout << "DQMClientExample: run Client at the end of the job" << endl;

}
//==================================================================//
//========================= beginJob ===============================//
//==================================================================//
void DQMClientExample::beginJob(){

   ////---- get DQM store interface 
  dbe_ = Service<DQMStore>().operator->();

  ////---- define the directory where the clientHisto will be located;
  dbe_->setCurrentFolder(monitorName_+"DQMclient");
  clientHisto = dbe_->book1D("clientHisto", "Guassian fit results.", 2, 0, 1);
}

//==================================================================//
//========================= beginRun ===============================//
//==================================================================//
void DQMClientExample::beginRun(const Run& r, const EventSetup& context) {
}

//==================================================================//
//==================== beginLuminosityBlock ========================//
//==================================================================//
void DQMClientExample::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
   // clientHisto->Reset();
}

//==================================================================//
//==================== analyse (takes each event) ==================//
//==================================================================//
void DQMClientExample::analyze(const Event& e, const EventSetup& context){
  counterEvt_++;
  if(clientOnEachEvent){
   if (prescaleEvt_>0 && counterEvt_ % prescaleEvt_ == 0) performClient();
  }
}

//==================================================================//
//========================= endLuminosityBlock =====================//
//==================================================================//
void DQMClientExample::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
 counterLS_ ++;
  if(!clientOnEachEvent && clientOnEndLumi){
  if( prescaleLS_ > 0  && counterLS_ % prescaleLS_ == 0) performClient();
 } 
}
//==================================================================//
//============================= endRun =============================//
//==================================================================//
void DQMClientExample::endRun(const Run& r, const EventSetup& context){
  if(clientOnEndRun) performClient();
}

//==================================================================//
//============================= endJob =============================//
//==================================================================//
void DQMClientExample::endJob(){
   std::cout << "DQMSourceClient::endJob()" << std::endl;
   if(clientOnEndJob) performClient();
}


//==================================================================//
//======================= performClient ============================//
//==================================================================//
void DQMClientExample::performClient(){

   std::cout << "***** run  Client operations as defined in: *****" << std::endl;
   std::cout << "***** DQMClientExample::performClient    ********" << std::endl;

  //----------------------------------------------------------------------------------------------
  // example how to retrieve a ME created by the DQM producer (in Examples/src/DQMSourceExample.cc) 
  //---------------------------------------------------------------------------------------------
   float mean =0;  float rms = 0;
  
 ////---- take histo2 histogram created by the DQM Source: DQMSourceExample.cc
  string histoName = monitorName_ + "DQMsource/C1/histo2";
  MonitorElement * meHisto = dbe_->get(histoName);
  ////---- check whether it is found 
  if (meHisto) { 
    ////--- extract histo from the ME
    if (TH1F *rootHisto = meHisto->getTH1F()) {
      if(counterClientOperation<1)
      std::cout<<"=== DQMClientExample::performClient => the histo is found: entries="<< rootHisto->GetEntries() << "\n" 
              <<" mean=" << rootHisto->GetMean() << " rms=" << rootHisto->GetRMS() << std::endl;

      ////---- make the Gaussian fit to this histogram
      TF1 *f1 = new TF1("f1","gaus",1,3);
      rootHisto->Fit("f1");
      mean = f1->GetParameter(1);
      rms = f1->GetParameter(2);
    }
  clientHisto->setBinContent(1,mean);
  clientHisto->setBinContent(2,rms);
  }

  else { //when it is not found !
  clientHisto->setBinContent(1,-1);
  clientHisto->setBinContent(2,-1);
 if(counterClientOperation<1)  edm::LogError ("DQMClientExample") <<"--- The following ME :"<< histoName <<" cannot be found\n" 
            		       <<"--- on the DQM source side !\n";
  }
 
  //----------------------------------------------------------------------------------------------
  // example  how to access the Quality test result for clientHisto
  //---------------------------------------------------------------------------------------------

 // qtest to be ran on clientHisto is defined in Examples/test/QualityTests.xml
  const QReport * theQReport = clientHisto->getQReport(QTestName_);
  if(theQReport) {
    if(counterClientOperation<1)
    edm::LogWarning ("DQMClientExample") <<"*** Summary of Quality Test for clientHisto: \n"
                                       //<<"---  value  ="<< theQReport->getQTresult()<<"\n"
                                      <<"--- status ="<< theQReport->getStatus() << "\n"
                                      <<"--- message ="<< theQReport->getMessage() << "\n";

   vector<dqm::me_util::Channel> badChannels = theQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
        if(counterClientOperation<1) 
        edm::LogError ("DQMClientExample") <<" Bad channels: "<<(*channel).getBin()<<" "<<(*channel).getContents();
    }
  } 
  else { 
  if(counterClientOperation<1) edm::LogError ("DQMClientExample") <<"--- No QReport is found for clientHisto!\n"
  << "--- Please check your XML and config files -> QTestName ( " << QTestName_ << " )must be the same!\n";
  }

 counterClientOperation++;
}

