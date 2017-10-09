/*
 *  See header file for a description of this class.
 *
 *  \author M. Pelliccioni - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */


#include <DQM/DTMonitorClient/src/DTOfflineSummaryClients.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <cmath>

using namespace edm;
using namespace std;


DTOfflineSummaryClients::DTOfflineSummaryClients(const ParameterSet& ps) : nevents(0) {

  LogVerbatim("DTDQM|DTMonitorClient|DTOfflineSummaryClients") << "[DTOfflineSummaryClients]: Constructor";
 
  bookingdone = 0;

}

DTOfflineSummaryClients::~DTOfflineSummaryClients(){
  LogVerbatim ("DTDQM|DTMonitorClient|DTOfflineSummaryClients") << "DTOfflineSummaryClients: analyzed " << nevents << " events";
  
}


void DTOfflineSummaryClients::beginRun (const edm::Run& r, const edm::EventSetup& c)  {};

void DTOfflineSummaryClients::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                                         edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {

  LogVerbatim("DTDQM|DTMonitorClient|DTOfflineSummaryClients")
    << "[DTOfflineSummaryClients]: End of LS transition" << endl;

}

void DTOfflineSummaryClients::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  LogVerbatim ("DTDQM|DTMonitorClient|DTOfflineSummaryClients") <<"[DTOfflineSummaryClients]: end job. Performin client operation"; 

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

  // reset the monitor elements
  summaryReportMap->Reset();
  summaryReport->Fill(0.);
  for(int ii = 0; ii != 5; ++ii) {
    theSummaryContents[ii]->Fill(0.);
  }

  // Fill the map using, at the moment, only the information from DT chamber efficiency
  // problems at a granularity smaller than the chamber are ignored
  for(int wheel=-2; wheel<=2; wheel++) { // loop over wheels
    // retrieve the chamber efficiency summary
    stringstream str;
    str << "DT/05-ChamberEff/EfficiencyMap_All_W" << wheel;
    MonitorElement * segmentWheelSummary =  igetter.get(str.str());
    if(segmentWheelSummary != 0) {

      float nFailingChambers = 0.;

      for(int sector=1; sector<=12; sector++) { // loop over sectors

	double meaneff = 0.;
	double errorsum = 0.;

	for(int station = 1; station != 5; ++station) { // loop over stations

	  const double tmpefficiency = segmentWheelSummary->getBinContent(sector, station);
	  const double tmpvariance = pow(segmentWheelSummary->getBinError(sector, station),2);

	  if(tmpefficiency == 0 || tmpvariance == 0){
	    nFailingChambers++;
	    continue;
	  }

	  meaneff += tmpefficiency/tmpvariance;
	  errorsum += 1./tmpvariance;

	  if(tmpefficiency < 0.2) nFailingChambers++;

	  LogTrace("DTDQM|DTMonitorClient|DTOfflineSummaryClients")
	    << "Wheel: " << wheel << " Stat: " << station << " Sect: " << sector << " status: " << meaneff/errorsum << endl;
	}

	const double eff_result = meaneff/errorsum;

	if(eff_result > 0.7) summaryReportMap->Fill(sector,wheel,1.);
	else if(eff_result < 0.7 && eff_result > 0.5) summaryReportMap->Fill(sector,wheel,0.6);
	else if(eff_result < 0.5 && eff_result > 0.3) summaryReportMap->Fill(sector,wheel,0.4);
	else if(eff_result < 0.3 && eff_result > 0.) summaryReportMap->Fill(sector,wheel,0.15);

      }
      theSummaryContents[wheel+2]->Fill((48.-nFailingChambers)/48.);
      summaryReport->Fill(summaryReport->getFloatValue() + theSummaryContents[wheel+2]->getFloatValue()/5.);
    } else {
      LogWarning("DTDQM|DTMonitorClient|DTOfflineSummaryClients")
	<< " [DTOfflineSummaryClients] Segment Summary not found with name: " << str.str() << endl;
    }
  }

}


