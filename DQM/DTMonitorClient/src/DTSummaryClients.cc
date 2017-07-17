/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */


#include <DQM/DTMonitorClient/src/DTSummaryClients.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

using namespace edm;
using namespace std;


DTSummaryClients::DTSummaryClients(const ParameterSet& ps) : nevents(0) {

  LogVerbatim("DTDQM|DTMonitorClient|DTSummaryClients") << "[DTSummaryClients]: Constructor";

  bookingdone = 0;

}

DTSummaryClients::~DTSummaryClients(){
  LogVerbatim ("DTDQM|DTMonitorClient|DTSummaryClients") << "DTSummaryClients: analyzed " << nevents << " events";
  
}

void DTSummaryClients::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
  
  LogVerbatim ("DTDQM|DTMonitorClient|DTSummaryClients") <<"[DTSummaryClients]: endJob"; 

}

  void DTSummaryClients::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                         edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {
  
  if (!bookingdone) {

  // book the summary histos

  ibooker.setCurrentFolder("DT/EventInfo"); 

  summaryReport = ibooker.bookFloat("reportSummary");
  // Initialize to 1 so that no alarms are thrown at the beginning of the run
  summaryReport->Fill(1.);

  summaryReportMap = ibooker.book2D("reportSummaryMap","DT Report Summary Map",12,1,13,5,-2,3);
  summaryReportMap->setAxisTitle("sector",1);
  summaryReportMap->setAxisTitle("wheel",2);

  ibooker.setCurrentFolder("DT/EventInfo/reportSummaryContents");

  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream streams;
    streams << "DT_Wheel" << wheel;
    string meName = streams.str();    

    theSummaryContents.push_back(ibooker.bookFloat(meName));
    // Initialize to 1 so that no alarms are thrown at the beginning of the run
    theSummaryContents[wheel+2]->Fill(1.);
  }

}
  bookingdone = 1; 


  LogVerbatim("DTDQM|DTMonitorClient|DTSummaryClients")
    << "[DTSummaryClients]: End of LS transition, performing the DQM client operation" << endl;

  // reset the monitor elements
  summaryReportMap->Reset();
  summaryReport->Reset();
  for(int ii = 0; ii != 5; ++ii) {
    theSummaryContents[ii]->Reset();
  }

  bool noDTData = false;

  // Check if DT data in each ROS have been read out and set the SummaryContents and the ErrorSummary
  // accordignly
  MonitorElement * dataIntegritySummary = igetter.get("DT/00-DataIntegrity/DataIntegritySummary");
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
    LogError("DTDQM|DTMonitorClient|DTSummaryClients")
      << "Data Integrity Summary not found with name: DT/00-DataIntegrity/DataIntegritySummary" <<endl;
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
    MonitorElement * wheelOccupancySummary =  igetter.get(str.str());
    if(wheelOccupancySummary != 0) {
      int nFailingChambers = 0;
      for(int sector=1; sector<=12; sector++){ // loop over sectors
	for(int station = 1; station != 5; ++station) { // loop over stations
	  double chamberStatus = wheelOccupancySummary->getBinContent(sector, station);
	  LogTrace("DTDQM|DTMonitorClient|DTSummaryClients")
	    << "Wheel: " << wheel << " Stat: " << station << " Sect: " << sector << " status: " << chamberStatus << endl;
	  if(chamberStatus != 4) {
	    summaryReportMap->Fill(sector, wheel, 0.25);
	  } else {
	    nFailingChambers++;
	  }
	  LogTrace("DTDQM|DTMonitorClient|DTSummaryClients") << " sector (" << sector << ") status on the map is: "
							     << summaryReportMap->getBinContent(sector, wheel+3) << endl;
	}

      }
      theSummaryContents[wheel+2]->Fill((48.-nFailingChambers)/48.);
      totalStatus += (48.-nFailingChambers)/48.;
    } else {
      occupancyFound = false;
      LogError("DTDQM|DTMonitorClient|DTSummaryClients")<< " Wheel Occupancy Summary not found with name: " << str.str() << endl;
    }
  }


  if(occupancyFound && !noDTData)
    summaryReport->Fill(totalStatus/5.);

}


