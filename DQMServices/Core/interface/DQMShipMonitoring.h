#ifndef DQMShipMonitoring_H
#define DQMShipMonitoring_H

/**\class DQMShipMonitoring

Description: Class shipping monitoring from DQM source to downstream collector
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "DQMServices/Core/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/RootMonitorThread.h"


//
// class declaration
//

class DQMShipMonitoring {
public:
  explicit DQMShipMonitoring( const edm::ParameterSet&,
			      edm::ActivityRegistry& iAR);
   ~DQMShipMonitoring();
   
  // come here after an event has been processed by all modules
  void postProcessEvent(const edm::Event&, const edm::EventSetup&);
  // come here after all modules have successfully had endJob called
  void postEndJob();

private:
      // ----------member data ---------------------------
  // event counter
  unsigned counter;
  // shipping monitoring period (# of events)
  unsigned event_period;

  RootMonitorThread * rmt;

  inline void getRMT()
  {
    rmt = MonitorDaemon::od;
  }

  inline void doTheDeed()
  {
    if(!rmt){getRMT(); if(!rmt)return;}

    rmt->sendMonitoringOnly();
  }

};


#endif // #define DQMShipMonitoring_H
