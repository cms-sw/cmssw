
/*
 * \file DQMClientExample.cc
 * \author M. Zanetti - CERN
 *
 * Last Update:
 * $Date: 2007/11/21 20:43:55 $
 * $Revision: 1.7 $
 * $Author: ameyer $
 *
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
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  // base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  cout << "Monitor name = " << monitorName_ << endl;
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;

  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;

      
}

//--------------------------------------------------------
void DQMClientExample::beginJob(const EventSetup& context){

  // get backendinterface  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  // do your thing
  dbe_->setCurrentFolder(monitorName_+"C1/Tests");
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
  
  string histoName = monitorName_ + "C1/C2/histo4";

  float mean =0;
  float rms = 0;

  MonitorElement * meHisto = dbe_->get(histoName);

  if (meHisto) {
    
    MonitorElementT<TNamed>* tNamedHisto = dynamic_cast<MonitorElementT<TNamed>*>(meHisto);
    if(tNamedHisto) {

      TH1F * rootHisto = dynamic_cast<TH1F *> (tNamedHisto->operator->());
      if(rootHisto) {

	TF1 *f1 = new TF1("f1","gaus",1,3);
	rootHisto->Fit("f1");
	mean = f1->GetParameter(1);
	rms = f1->GetParameter(2);
      }
    }
  }

  clientHisto->setBinContent(1,mean);
  clientHisto->setBinContent(2,rms);


  string criterionName = parameters_.getUntrackedParameter<string>("QTestName","exampleQTest"); 
  const QReport * theQReport = clientHisto->getQReport(criterionName);
  if(theQReport) {
    vector<dqm::me_util::Channel> badChannels = theQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
      edm::LogError ("DQMClientExample") <<" Bad channels: "<<(*channel).getBin()<<" "<<(*channel).getContents();
    }
    edm::LogWarning ("DQMClientExample") <<"-------- "<<theQReport->getMessage()<<" ------- "<<theQReport->getStatus();
  } 

}

//--------------------------------------------------------
void DQMClientExample::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void DQMClientExample::endJob(){
}




