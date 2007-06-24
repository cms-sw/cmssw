/*
 * \file MonitorClient.cc
 * 
 * $Date: 2007/06/21 15:05:55 $
 * $Revision: 1.1.2.1 $
 * \author M. Zanetti - CERN
 *
 */


#include "DQMServices/Examples/interface/MonitorClient.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DQMServices/Core/interface/MonitorElementBaseT.h>

// Geometry
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TF1.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

MonitorClient::MonitorClient(const edm::ParameterSet& ps){
  
  edm::LogVerbatim ("monitorClient") <<"[MonitorClient]: Constructor";

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);


}


MonitorClient::~MonitorClient(){

  edm::LogVerbatim ("monitorClient") <<"MonitorClient: analyzed " << nevents << " events";

}

void MonitorClient::endJob(){

  edm::LogVerbatim ("monitorClient") <<"[MonitorClient] endjob called!";

  dbe->rmdir("C1/Tests");
}

void MonitorClient::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("monitorClient") <<"[MonitorClient]: BeginJob";

  dbe->setCurrentFolder("C1/Tests");

  clientHisto = dbe->book1D("clientHisto", "Guassian fit results.", 2, 0, 1);
  
  nevents = 0;

}



  

void MonitorClient::analyze(const edm::Event& e, const edm::EventSetup& context){

  // counts number of updats (online mode) or number of events (standalone mode)
  nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
       nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;

  edm::LogVerbatim ("monitorClient") <<"[MonitorClient]: "<<nevents<<" updates";


  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string histoName = folderRoot + "C1/C2/histo4";

  float mean =0;
  float rms = 0;

  MonitorElement * meHisto = dbe->get(histoName);
  if (meHisto) {
    
    edm::LogVerbatim ("monitorelement") <<"[MonitorClient]: I've got the histo!!";	
    
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


  string criterionName = parameters.getUntrackedParameter<string>("QTestName","exampleQTest"); 
  const QReport * theQReport = clientHisto->getQReport(criterionName);
  if(theQReport) {
    vector<dqm::me_util::Channel> badChannels = theQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
      edm::LogError ("monitorClient") <<" Bad channels: "<<(*channel).getBin()<<" "<<(*channel).getContents();
    }
    edm::LogWarning ("monitorClient") <<"-------- "<<theQReport->getMessage()<<" ------- "<<theQReport->getStatus();
  } 


}




