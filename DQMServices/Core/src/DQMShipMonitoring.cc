// -*- C++ -*-
//
// Package:    DQMServices/Daemon
// Class:      DQMShipMonitoring
// 
/**\class DQMShipMonitoring

Description: Class shipping monitoring from DQM source to downstream collector

Implementation:
<Notes on implementation>
*/
//
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "DQMServices/Core/interface/DQMShipMonitoring.h"

//
// constructors and destructor
//
DQMShipMonitoring::DQMShipMonitoring( const edm::ParameterSet& iConfig,
				      edm::ActivityRegistry& iAR)
  : counter(0), rmt(0)
{
  getRMT();

  event_period = iConfig.getUntrackedParameter<unsigned>("period", 25);

  iAR.watchPostProcessEvent(this, &DQMShipMonitoring::postProcessEvent);
  iAR.watchPostEndJob(this, &DQMShipMonitoring::postEndJob);
}


DQMShipMonitoring::~DQMShipMonitoring()
{
  
}

// come here after all modules have successfully had endJob called
void DQMShipMonitoring::postEndJob()
{
  doTheDeed();
}

// come here after an event has been processed by all modules
void DQMShipMonitoring::postProcessEvent(const edm::Event& ie, 
					 const edm::EventSetup& ies)
{   
  ++counter;
  if(counter % event_period)
    return;

  doTheDeed();
}


