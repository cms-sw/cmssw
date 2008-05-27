

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/05/06 14:02:08 $
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


DTSummaryClients::DTSummaryClients(const edm::ParameterSet& ps){

  edm::LogVerbatim ("summary") << "[DTSummaryClients]: Constructor";
  parameters = ps;
  
  dbe = edm::Service<DQMStore>().operator->();

}

DTSummaryClients::~DTSummaryClients(){

  edm::LogVerbatim ("summary") << "DTSummaryClients: analyzed " << nevents << " events";

}

void DTSummaryClients::beginRun(Run const& run, EventSetup const& eSetup) {

  edm::LogVerbatim ("summary") <<"[DTSummaryClients]: BeginRun"; 

  // book the summary histos

  dbe->setCurrentFolder("DT/EventInfo"); 
  SummaryReport = dbe->bookFloat("reportSummary");

  SummaryReportMap = dbe->book2D("reportSummaryMap","DT Report Summary Map",14,0.5,14.5,5,-2.5,2.5);
  SummaryReportMap->setBinLabel(1,"Sec1",1);
  SummaryReportMap->setBinLabel(2,"Sec2",1);
  SummaryReportMap->setBinLabel(3,"Sec3",1);
  SummaryReportMap->setBinLabel(4,"Sec4",1);
  SummaryReportMap->setBinLabel(5,"Sec5",1);
  SummaryReportMap->setBinLabel(6,"Sec6",1);
  SummaryReportMap->setBinLabel(7,"Sec7",1);
  SummaryReportMap->setBinLabel(8,"Sec8",1);
  SummaryReportMap->setBinLabel(9,"Sec9",1);
  SummaryReportMap->setBinLabel(10,"Sec10",1);
  SummaryReportMap->setBinLabel(11,"Sec11",1);
  SummaryReportMap->setBinLabel(12,"Sec12",1);
  SummaryReportMap->setBinLabel(13,"Sec13",1);
  SummaryReportMap->setBinLabel(14,"Sec14",1);
  SummaryReportMap->setBinLabel(1,"Wh-2",2);
  SummaryReportMap->setBinLabel(2,"Wh-1",2);
  SummaryReportMap->setBinLabel(3,"Wh0",2);
  SummaryReportMap->setBinLabel(4,"Wh+1",2);
  SummaryReportMap->setBinLabel(5,"Wh+2",2);

  dbe->setCurrentFolder("DT/EventInfo/reportSummaryContents");

  SummaryLocalTrigger  = dbe->bookFloat("SummaryLocalTrigger");
  SummaryOccupancy  = dbe->bookFloat("SummaryOccupancy");
  SummaryDataIntegrity  = dbe->bookFloat("SummaryDataIntegrity");

}


void DTSummaryClients::endJob(void){
  
  edm::LogVerbatim ("summary") <<"[DTSummaryClients]: endJob"; 

}


void DTSummaryClients::endRun(Run const& run, EventSetup const& eSetup) {
  
  edm::LogVerbatim ("summary") <<"[DTSummaryClients]: endRun"; 

}


void DTSummaryClients::analyze(const edm::Event& e, const edm::EventSetup& context){

   nevents++;
   edm::LogVerbatim ("summary") << "[DTSummaryClients]: "<<nevents<<" events";
   
}


void DTSummaryClients::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  edm::LogVerbatim ("summary") <<"[DTSummaryClients]: End of LS transition, performing the DQM client operation";

  // fill the summary histos

  SummaryReport->Fill(1);

  for(int sector=1; sector<=14; sector++){
    for(int wheel=-2; wheel<=2; wheel++){
      
      SummaryReportMap->Fill(sector, wheel, 1);

    }
  }

  SummaryLocalTrigger->Fill(1);
  SummaryOccupancy->Fill(1);
  SummaryDataIntegrity->Fill(1);

}


