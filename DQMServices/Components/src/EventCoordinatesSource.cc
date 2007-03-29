/*
 * \file EventCoordinatesSource.cc
 * 
 * $Date: 2006/10/31 00:16:28 $
 * $Revision: 1.15 $
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

  edm::Service<MonitorDaemon> daemon; 	 
  daemon.operator->();

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



