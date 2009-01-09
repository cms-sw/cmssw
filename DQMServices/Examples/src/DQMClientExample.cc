
/*
 * \file DQMClientExample.cc
 * \author M. Zanetti - CERN
 *
 * Last Update:
 * $Date: 2009/01/09 15:43:33 $
 * $Revision: 1.11 $
 * $Author: dvolyans $
 *
 */

/*  Description: Simple example showing how to access the histograms 
 *  already defined and filled by the DQM producer(source) 
 *  and how to access the quality test results.
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

DQMClientExample::DQMClientExample(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

DQMClientExample::~DQMClientExample(){
}

//--------------------------------------------------------
void DQMClientExample::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  cout << "DQMClientExample: Monitor name = " << monitorName_ << endl;
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;

  QTestName_ = parameters_.getUntrackedParameter<string>("QTestName","exampleQTest"); 
  cout << "DQMClientExample: QTest name to be ran on clientHisto = " << QTestName_ << endl;

  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  cout << "DQMClientExample: DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "DQMClientExample: DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
    


}

//--------------------------------------------------------
void DQMClientExample::beginJob(const EventSetup& context){

  // get back-end interface  
  dbe_ = Service<DQMStore>().operator->();

  // define the directory where the clientHosto will be located;
  dbe_->setCurrentFolder(monitorName_+"DQMclient");
  clientHisto = dbe_->book1D("clientHisto", "Guassian fit results.", 2, 0, 1);
}

//--------------------------------------------------------
void DQMClientExample::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void DQMClientExample::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
   // clientHisto->Reset();
}

//--------------------------------------------------------
void DQMClientExample::analyze(const Event& e, const EventSetup& context){
    
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

//    // endLuminosityBlock(); // FIXME call client here
// 
// }
// 
// 
// //--------------------------------------------------------
// void DQMClientExample::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
//                                           const EventSetup& context) {
// 					  
//   counterLS_++;
//   if ( prescaleLS_>0 && counterLS_%prescaleLS_ != 0 ) return;
//   // do your thing here
//   
  
 //----------------------------------------------------------------------------------------------
 // example how to retrieve a ME created by the DQM producer (in Examples/src/DQMSourceExample.cc) 
 //---------------------------------------------------------------------------------------------
   float mean =0;  float rms = 0;

  string histoName = monitorName_ + "DQMsource/QTests/MeanTrue";
  MonitorElement * meHisto = dbe_->get(histoName);
  if (meHisto)  
  { //when it is found, do some stuff 
    if (TH1F *rootHisto = meHisto->getTH1F())
    {
      TF1 *f1 = new TF1("f1","gaus",1,3);
      rootHisto->Fit("f1");
      mean = f1->GetParameter(1);
      rms = f1->GetParameter(2);
    }
  clientHisto->setBinContent(1,mean);
  clientHisto->setBinContent(2,rms);
  }

  else
  { //when it is not found !
  clientHisto->setBinContent(1,-1);
  clientHisto->setBinContent(2,-1);
  if(counterEvt_==1) edm::LogError ("DQMClientExample") <<"--- The following ME :"<< histoName <<" cannot be found\n" 
            		       <<"--- on the DQM source side !\n";
  }
 
  // qtest to be ran on clientHisto is defined in Examples/test/QualityTests.xml

}

//--------------------------------------------------------
void DQMClientExample::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void DQMClientExample::endJob(){

  //-- example  how to access the Quality test result
  const QReport * theQReport = clientHisto->getQReport(QTestName_);
  if(theQReport) {
    edm::LogWarning ("DQMClientExample") <<"*** Summary of Quality Test for clientHisto: \n"
                                       //<<"---  value  ="<< theQReport->getQTresult()<<"\n"
                                      <<"--- status ="<< theQReport->getStatus() << "\n"
                                      <<"--- message ="<< theQReport->getMessage() << "\n";
   vector<dqm::me_util::Channel> badChannels = theQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
      edm::LogError ("DQMClientExample") <<" Bad channels: "<<(*channel).getBin()<<" "<<(*channel).getContents();
    }
  } 

  else {
  edm::LogError ("DQMClientExample") <<"--- No QReport is found for clientHisto!\n"
  << "--- Please check your XML and config files -> QTestName ( " << QTestName_ << " )must be the same!\n";
  }

}




