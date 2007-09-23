/*
 * \file DQMClientExample.cc
 * 
 * $Date: 2007/08/31 09:39:07 $
 * $Revision: 1.4 $
 * \author M. Zanetti - CERN
 *
 */


#include "DQMServices/Examples/interface/DQMClientExample.h"

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

DQMClientExample::DQMClientExample(const edm::ParameterSet& ps)
  : DQMAnalyzer(ps) {
  edm::LogVerbatim ("DQMClientExample") <<"[DQMClientExample]: Constructor";
}


DQMClientExample::~DQMClientExample(){
  edm::LogVerbatim ("DQMClientExample") <<"DQMClientExample: analyzed " << getNumLumiSecs() << " events";
}


//--------------------------------------------------------
void DQMClientExample::beginJob(const EventSetup& context){
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginJob(context);

  // then do your thing
  edm::LogVerbatim ("DQMClientExample") <<"[DQMClientExample]: BeginJob";
  dbe->setCurrentFolder(PSrootFolder+"C1/Tests");
  clientHisto = dbe->book1D("clientHisto", "Guassian fit results.", 2, 0, 1);
}

//--------------------------------------------------------
void DQMClientExample::beginRun(const EventSetup& context) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginRun(context);

  // then do your thing
  edm::LogVerbatim ("DQMClientExample") <<"[DQMClientExample]: Begin of Run";
}

//--------------------------------------------------------
void DQMClientExample::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginLuminosityBlock(lumiSeg,context);

  // then do your thing
  edm::LogVerbatim ("DQMClientExample") <<"[DQMClientExample]: Begin of LS transition";
}

//--------------------------------------------------------
void DQMClientExample::analyze(const Event& e, const EventSetup& context){

  // call DQMAnalyzer some place
  DQMAnalyzer::analyze(e,context);
}


//--------------------------------------------------------
void DQMClientExample::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
  // do your thing here
  edm::LogVerbatim ("DQMClientExample") <<"[DQMClientExample]: End of LS transition, performing the DQM client operation";
  if ( getNumLumiSecs()%PSprescale != 0 ) return;
  edm::LogVerbatim ("DQMClientExample") <<"[DQMClientExample]: "<<getNumLumiSecs()<<" updates";

  string folderRoot = PSrootFolder ;
  string histoName = folderRoot + "C1/C2/histo4";

  float mean =0;
  float rms = 0;

  MonitorElement * meHisto = dbe->get(histoName);

  if (meHisto) {
    
    edm::LogVerbatim ("monitorelement") <<"[DQMClientExample]: I've got the histo!!";	
    
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
      edm::LogError ("DQMClientExample") <<" Bad channels: "<<(*channel).getBin()<<" "<<(*channel).getContents();
    }
    edm::LogWarning ("DQMClientExample") <<"-------- "<<theQReport->getMessage()<<" ------- "<<theQReport->getStatus();
  } 

  // call DQMAnalyzer at the end 
  DQMAnalyzer::endLuminosityBlock(lumiSeg,context);
}

//--------------------------------------------------------
void DQMClientExample::endRun(const Run& r, const EventSetup& context){
  // do your thing here
  
  // call DQMAnalyzer at the end
  DQMAnalyzer::endRun(r,context); 
}

//--------------------------------------------------------
void DQMClientExample::endJob(){
  // do your thing here
  edm::LogVerbatim ("DQMClientExample") <<"[DQMClientExample] endjob called!";
  dbe->rmdir(PSrootFolder+"C1/Tests");

  // call DQMAnalyzer in the end
  DQMAnalyzer::endJob();
}




