/*
 * \file DTScalerInfoTask.cc
 *
 * \author C. Battilana - CIEMAT
 *
*/

#include "DQM/DTMonitorModule/src/DTScalerInfoTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// DT DQM
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

#include <sstream>
#include <iostream>
#include <fstream>

using namespace edm;
using namespace std;

DTScalerInfoTask::DTScalerInfoTask(const edm::ParameterSet& ps) :
  nEvents(0) {

  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask")
    << "[DTScalerInfoTask]: Constructor"<<endl;

  scalerToken_ = consumes<LumiScalersCollection>(
      ps.getUntrackedParameter<InputTag>("inputTagScaler"));
  theParams = ps;
}


DTScalerInfoTask::~DTScalerInfoTask() {

  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask")
    << "[DTScalerInfoTask]: analyzed " << nEvents << " events" << endl;

}


void DTScalerInfoTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) {

  LogTrace("DTDQM|DTMonitorModule|DTScalerInfoTask")
    << "[DTScalerInfoTask]: BeginRun" << endl;
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

void DTScalerInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nEvents++;
  nEventsInLS++;
  nEventMonitor->Fill(nEvents);

  //retrieve the luminosity
  edm::Handle<LumiScalersCollection> lumiScalers;
  if (e.getByToken(scalerToken_, lumiScalers)) {
    if (lumiScalers->begin() != lumiScalers->end()) {
      LumiScalersCollection::const_iterator lumiIt = lumiScalers->begin();
      trendHistos["AvgLumivsLumiSec"]->accumulateValueTimeSlot(lumiIt->instantLumi());
    }
    else {
      LogVerbatim("DTDQM|DTMonitorModule|DTScalerInfoTask")
	<< "[DTScalerInfoTask]: LumiScalersCollection size == 0" << endl;
    }
  }
  else {
    LogVerbatim("DTDQM|DTMonitorModule|DTScalerInfoTask")
      << "[DTScalerInfoTask]: LumiScalersCollection getByToken call failed" << endl;
  }

}

void DTScalerInfoTask::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & context) {

  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = ibooker.bookFloat("nProcessedEventsScalerInfo");

  ibooker.setCurrentFolder("DT/00-DataIntegrity/ScalerInfo");

  string histoName = "AvgLumivsLumiSec";
  string histoTitle = "Average Lumi vs LumiSec";
  trendHistos[histoName] = new DTTimeEvolutionHisto(ibooker,histoName,histoTitle,200,10,true,0);

}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
