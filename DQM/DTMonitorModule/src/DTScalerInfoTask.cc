/*
 * \file DTScalerInfoTask.cc
 * 
 * $Date: 2011/10/19 10:05:54 $
 * $Revision: 1.1 $
 * \author C. Battilana - CIEMAT
 *
*/

#include "DQM/DTMonitorModule/src/DTScalerInfoTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// DT DQM
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include <sstream>
#include <iostream>
#include <fstream>

using namespace edm;
using namespace std;

DTScalerInfoTask::DTScalerInfoTask(const edm::ParameterSet& ps) : 
  nEvents(0) {
  
  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask") 
    << "[DTScalerInfoTask]: Constructor"<<endl;

  theScalerTag = ps.getUntrackedParameter<InputTag>("inputTagScaler");
  theParams = ps; 
  theDQMStore = edm::Service<DQMStore>().operator->();

}


DTScalerInfoTask::~DTScalerInfoTask() {

  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask") 
    << "[DTScalerInfoTask]: analyzed " << nEvents << " events" << endl;

}


void DTScalerInfoTask::beginJob() {
 
  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask") 
    << "[DTScalerInfoTask]: BeginJob" << endl;

}


void DTScalerInfoTask::beginRun(const edm::Run& run, const edm::EventSetup& context) {

  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask") 
    << "[DTScalerInfoTask]: BeginRun" << endl;   

  bookHistos();

}


void DTScalerInfoTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  nEventsInLS=0;

  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask") 
    << "[DTScalerInfoTask]: Begin of LS transition" << endl;
  
  }

void DTScalerInfoTask::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask") 
    << "[DTScalerInfoTask]: End of LS transition" << endl;

  
  int block = lumiSeg.luminosityBlock();

  map<string,DTTimeEvolutionHisto* >::const_iterator histoIt  = trendHistos.begin();
  map<string,DTTimeEvolutionHisto* >::const_iterator histoEnd = trendHistos.end();
  for(;histoIt!=histoEnd;++histoIt) {
    histoIt->second->updateTimeSlot(block, nEventsInLS);
  }
  
}


void DTScalerInfoTask::endJob() {

  LogVerbatim("DTDQM|DTMonitorModule|DTScalerInfoTask") 
    << "[DTScalerInfoTask]: analyzed " << nEvents << " events" << endl;

}


void DTScalerInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  
  nEvents++;
  nEventsInLS++;
  nEventMonitor->Fill(nEvents);

  //retrieve the luminosity
  edm::Handle<LumiScalersCollection> lumiScalers;
  e.getByLabel(theScalerTag, lumiScalers);
  LumiScalersCollection::const_iterator lumiIt = lumiScalers->begin();
  trendHistos["AvgLumivsLumiSec"]->accumulateValueTimeSlot(lumiIt->instantLumi());

}


void DTScalerInfoTask::bookHistos() {

  theDQMStore->setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = theDQMStore->bookFloat("nProcessedEventsScalerInfo");

  theDQMStore->setCurrentFolder("DT/00-DataIntegrity/ScalerInfo");

  string histoName = "AvgLumivsLumiSec";
  string histoTitle = "Average Lumi vs LumiSec";
  trendHistos[histoName] = new DTTimeEvolutionHisto(theDQMStore,histoName,histoTitle,200,10,true,0);

}
