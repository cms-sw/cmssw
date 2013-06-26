/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/03/02 14:00:29 $
 *  $Revision: 1.1 $
 *  \author C. Battilana - CIEMAT
 *  \author P. Bellan - INFN PD
 *  \author A. Branca = INFN PD
 */


#include "DQM/DTMonitorClient/src/DTDCSByLumiSummary.h"
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include<string>


using namespace std;
using namespace edm;


DTDCSByLumiSummary::DTDCSByLumiSummary(const ParameterSet& pset) {}


DTDCSByLumiSummary::~DTDCSByLumiSummary() {}


void DTDCSByLumiSummary::beginJob(){

  theDQMStore = Service<DQMStore>().operator->();

  // book the ME
  theDQMStore->setCurrentFolder("DT/EventInfo/DCSContents");

  totalDCSFraction = theDQMStore->bookFloat("DTDCSSummary");  
  totalDCSFraction->setLumiFlag(); // set LumiFlag to DCS content value (save it by lumi)

  globalHVSummary = theDQMStore->book2D("HVGlbSummary","HV Status Summary",1,1,13,5,-2,3);
  globalHVSummary->setAxisTitle("Sectors",1);
  globalHVSummary->setAxisTitle("Wheel",2);

  for(int wh=-2;wh<=2;wh++){

    stringstream wheel_str; wheel_str << wh;

    MonitorElement* FractionWh = theDQMStore->bookFloat("DT_Wheel"+wheel_str.str());  
    FractionWh->setLumiFlag(); // set LumiFlag to DCS content value (save it by lumi)

    totalDCSFractionWh.push_back(FractionWh);
  }

  globalHVSummary->Reset();

}


void DTDCSByLumiSummary::beginLuminosityBlock(const LuminosityBlock& lumi, const  EventSetup& setup) {

  // CB LumiFlag marked products are reset on LS boundaries
  totalDCSFraction->Reset(); 

  for(int wh=-2;wh<=2;wh++){
    totalDCSFractionWh[wh+2]->Reset();
  }

}


void DTDCSByLumiSummary::endLuminosityBlock(const LuminosityBlock&  lumi, const  EventSetup& setup){

  // Get the by lumi product plot from the task
  int lumiNumber = lumi.id().luminosityBlock();

  bool null_pointer_histo(0);

  std::vector<float> wh_activeFrac;

  for(int wh=-2;wh<=2;wh++){

    stringstream wheel_str; wheel_str << wh;	

    string hActiveUnitsPath = "DT/EventInfo/DCSContents/hActiveUnits"+wheel_str.str();
    MonitorElement *hActiveUnits = theDQMStore->get(hActiveUnitsPath);

    if (hActiveUnits) {
      float activeFrac = static_cast<float>(hActiveUnits->getBinContent(2)) /  // CB 2nd bin is # of active channels
        hActiveUnits->getBinContent(1);    // first bin is overall number of channels

      if(activeFrac < 0.) activeFrac=-1;

      wh_activeFrac.push_back(activeFrac);

      // Fill by lumi Certification ME
      totalDCSFraction->Fill(activeFrac); 
      totalDCSFractionWh[wh+2]->Fill(activeFrac);

    } else {
      LogTrace("DTDQM|DTMonitorClient|DTDCSByLumiSummary")
        << "[DTDCSByLumiSummary]: got null pointer retrieving histo at :" 
        << hActiveUnitsPath << " for lumi # " << lumiNumber
        << "client operation not performed." << endl;

      null_pointer_histo=true;
    }    

  } // end loop on wheels

  if(!null_pointer_histo) dcsFracPerLumi[lumiNumber] = wh_activeFrac; // Fill map to be used to compute trend plots

}


void DTDCSByLumiSummary::endRun(const edm::Run& run, const edm::EventSetup& setup) {


  // Book trend plots ME & loop on map to fill it with by lumi info
  map<int,std::vector<float> >::const_iterator fracPerLumiIt  = dcsFracPerLumi.begin();
  map<int,std::vector<float> >::const_iterator fracPerLumiEnd = dcsFracPerLumi.end();

  int fLumi = dcsFracPerLumi.begin()->first;
  int lLumi = dcsFracPerLumi.rbegin()->first;
  theDQMStore->setCurrentFolder("DT/EventInfo/DCSContents");

  int nLumis = lLumi-fLumi + 1.;

  // trend plots
  for(int wh=-2; wh<=2; wh++) {

    stringstream wheel_str; wheel_str << wh;	

    DTTimeEvolutionHisto* trend;

    trend = new DTTimeEvolutionHisto(theDQMStore, "hDCSFracTrendWh" + wheel_str.str(), "Fraction of DT-HV ON Wh" + wheel_str.str(),
        nLumis, fLumi, 1, false, 2);

    hDCSFracTrend.push_back(trend);

  }

  float goodLSperWh[5] = {0,0,0,0,0}; 
  float badLSperWh[5] = {0,0,0,0,0};

  // fill trend plots and save infos for summaryPlot
  for(;fracPerLumiIt!=fracPerLumiEnd;++fracPerLumiIt) {

    for(int wh=-2; wh<=2; wh++) {

      std::vector<float> activeFracPerWh;
      activeFracPerWh =  fracPerLumiIt->second;

      hDCSFracTrend[wh+2]->setTimeSlotValue(activeFracPerWh[wh+2],fracPerLumiIt->first);

      if( activeFracPerWh[wh+2] > 0 ) { // we do not count the lumi were the DTs are off (no real problem), 
        // even if this can happen in the middle of a run (real problem: to be fixed)
        if( activeFracPerWh[wh+2] > 0.9 ) goodLSperWh[wh+2]++;
        else { 
          badLSperWh[wh+2]++;
        }
      } else {  // there is no HV value OR all channels OFF
        if( activeFracPerWh[wh+2] < 0 ) badLSperWh[wh+2]=-1;       // if there were no HV values, activeFrac returning -1
      }

    }

  }

  // fill summaryPlot
  for(int wh=-2; wh<=2; wh++) {

    if( goodLSperWh[wh+2] != 0 || badLSperWh[wh+2] == -1 ) {

      float r = badLSperWh[wh+2]/fabs(goodLSperWh[wh+2] + badLSperWh[wh+2]);
      if( r > 0.5 ) globalHVSummary->Fill(1,wh,0);
      else globalHVSummary->Fill(1,wh,1); 
      if( r == -1 ) globalHVSummary->Fill(1,wh,-1);    

    } else {

      globalHVSummary->Fill(1,wh,0);

    }

  }

}


void DTDCSByLumiSummary::endJob() {  

}


void DTDCSByLumiSummary::analyze(const Event& event, const EventSetup& setup){ 

}

