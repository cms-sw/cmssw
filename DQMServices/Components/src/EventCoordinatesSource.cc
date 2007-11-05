/*
 * \file EventCoordinatesSource.cc
 * 
 * $Date: 2007/04/03 09:51:57 $
 * $Revision: 1.3 $
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
  
  parameters_ = ps;
  
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();

  dbe_->setVerbose(1);
  dbe_->setCurrentFolder(parameters_.getUntrackedParameter<string>("eventInfoFolder", "EventInfo/")) ;
  runId_     = dbe_->bookInt("iRun");
  lumisecId_ = dbe_->bookInt("iLumiSection");
  eventId_   = dbe_->bookInt("iEvent");
  timeStamp_ = dbe_->bookFloat("timeStamp");

}

EventCoordinatesSource::~EventCoordinatesSource(){

  cout<<"[EventCoordinatesSource]: distructor"<<endl;

}

void EventCoordinatesSource::analyze(const Event& e, const EventSetup& c){

  runId_->Fill(e.id().run());
  lumisecId_->Fill(e.luminosityBlock());
  eventId_->Fill(e.id().event());
  timeStamp_->Fill(e.time().value());

}
