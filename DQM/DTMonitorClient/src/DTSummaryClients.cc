

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/07/07 20:43:11 $
 *  $Revision: 1.5 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTSummaryClients.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h" 
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <stdio.h>
#include <string>

using namespace edm;
using namespace std;


DTSummaryClients::DTSummaryClients(const edm::ParameterSet& ps) : nevents(0) {

  edm::LogVerbatim ("DTSummaryClient") << "[DTSummaryClients]: Constructor";
  
  
  dbe = edm::Service<DQMStore>().operator->();

}

DTSummaryClients::~DTSummaryClients(){

  edm::LogVerbatim ("DTSummaryClient") << "DTSummaryClients: analyzed " << nevents << " events";

}

void DTSummaryClients::beginRun(Run const& run, EventSetup const& eSetup) {

  edm::LogVerbatim ("DTSummaryClient") <<"[DTSummaryClients]: BeginRun"; 

  // book the summary histos
  dbe->setCurrentFolder("DT/EventInfo"); 
  summaryReport = dbe->bookFloat("reportSummary");
  // Initialize to 1 so that no alarms are thrown at the beginning of the run
  summaryReport->Fill(1.);

  summaryReportMap = dbe->book2D("reportSummaryMap","DT Report Summary Map",12,1,13,5,-2,3);
  summaryReportMap->setAxisTitle("sector",1);
  summaryReportMap->setAxisTitle("wheel",2);

  dbe->setCurrentFolder("DT/EventInfo/reportSummaryContents");

  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream streams;
    streams << "DT_Wheel" << wheel;
    string meName = streams.str();    
    theSummaryContents.push_back(dbe->bookFloat(meName));
    // Initialize to 1 so that no alarms are thrown at the beginning of the run
    theSummaryContents[wheel+2]->Fill(1.);
  }




}


void DTSummaryClients::endJob(void){
  
  edm::LogVerbatim ("DTSummaryClient") <<"[DTSummaryClients]: endJob"; 

}


void DTSummaryClients::endRun(Run const& run, EventSetup const& eSetup) {
  
  edm::LogVerbatim ("DTSummaryClient") <<"[DTSummaryClients]: endRun"; 

}


void DTSummaryClients::analyze(const edm::Event& e, const edm::EventSetup& context){

   nevents++;
   edm::LogVerbatim ("DTSummaryClient") << "[DTSummaryClients]: "<<nevents<<" events";
   
}


void DTSummaryClients::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  LogVerbatim ("DTSummaryClient")
    <<"[DTSummaryClients]: End of LS transition, performing the DQM client operation";

  bool noDTData = false;

  // Check if DT data in each ROS have been read out and set the SummaryContents and the ErrorSummary
  // accordignly
  MonitorElement * dataIntegritySummary = dbe->get("DT/00-DataIntegrity/DataIntegritySummary");
  if(dataIntegritySummary != 0) {
  int nDisabledFED = 0;
  for(int wheel = 1; wheel != 6; ++wheel) { // loop over the wheels
    int nDisablesROS = 0;
    for(int sect = 1; sect != 13; ++sect) { // loop over sectors
      if(dataIntegritySummary->getBinContent(sect,wheel) == 1) {
	nDisablesROS++;
      }
    }
    if(nDisablesROS == 12) {
      nDisabledFED++;
      theSummaryContents[wheel-1]->Fill(0);
    }
  }
  
  if(nDisabledFED == 5) {
    noDTData = true;
    summaryReport->Fill(-1);
  }
  
  } else {
    cout <<  "Data Integrity Summary not found with name: DT/00-DataIntegrity/DataIntegritySummary" <<endl;
  }

  double totalStatus = 0;
  // protection 
  bool occupancyFound = true;

  // Fill the map using, at the moment, only the information from DT occupancy
  // problems at a granularity smaller than the chamber are ignored
  for(int wheel=-2; wheel<=2; wheel++){ // loop over wheels
    // retrieve the occupancy summary
    stringstream str;
    str << "DT/01-Digi/OccupancySummary_W" << wheel;
    MonitorElement * wheelOccupancySummary =  dbe->get(str.str());
    if(wheelOccupancySummary != 0) {
      int nFailingChambers = 0;
      for(int sector=1; sector<=12; sector++){ // loop over sectors
	for(int station = 1; station != 5; ++station) { // loop over stations
	  if(wheelOccupancySummary->getBinContent(sector, wheel+3) != 4) {
	    summaryReportMap->Fill(sector, wheel, 0.25);
	  } else {
	    nFailingChambers++;
	  }
	}

      }
      theSummaryContents[wheel+2]->Fill((48.-nFailingChambers)/48.);
      totalStatus += (48.-nFailingChambers)/48.;
    } else {
      occupancyFound = false;
      cout << " Wheel Occupancy Summary not found with name: " << str.str() << endl;
    }
  }


  if(occupancyFound && !noDTData)
    summaryReport->Fill(totalStatus/5.);
}


