/*
 * \file EventCoordinatesSource.cc
 * 
 * $Date: 2007/03/29 14:52:55 $
 * $Revision: 1.2 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "EventCoordinatesSource.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

EventCoordinatesSource::EventCoordinatesSource(const ParameterSet& ps){
  
  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  if ( parameters.getUntrackedParameter<bool>("enableMonitorDaemon", true) ) {
    Service<MonitorDaemon> daemon;
    daemon.operator->();
  } 
  else {
    cout<<"[EventCoordinatesSource]: Warning, MonitorDaemon service not enabled"<<endl;
  }

  dbe->setVerbose(1);
  dbe->setCurrentFolder(parameters.getUntrackedParameter<string>("eventInfoFolder", "EventInfo/")) ;
  runId = dbe->bookInt("iRun");
  eventId = dbe->bookInt("iEvent");
  timeStamp = dbe->bookFloat("timeStamp");

}

EventCoordinatesSource::~EventCoordinatesSource(){

  cout<<"[EventCoordinatesSource]: distructor"<<endl;

}

void EventCoordinatesSource::analyze(const Event& e, const EventSetup& c){

  runId->Fill(e.id().run());
  eventId->Fill(e.id().event());
  timeStamp->Fill(e.time().value());

}



