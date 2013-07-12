/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/06/10 14:56:27 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DQM/DTMonitorModule/src/DTDataErrorFilter.h"
#include "DQM/DTMonitorModule/interface/DTDataIntegrityTask.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


DTDataErrorFilter::DTDataErrorFilter(const edm::ParameterSet & config) :
  HLTFilter(config)
{
  // Get the data integrity service
  dataMonitor = edm::Service<DTDataIntegrityTask>().operator->();
}

DTDataErrorFilter::~DTDataErrorFilter(){}


bool DTDataErrorFilter::hltFilter(edm::Event& event, const edm::EventSetup& setup, trigger::TriggerFilterObjectWithRefs & filterproduct) {
  // check the event error flag 
  if (dataMonitor->eventHasErrors()) return true;
  return false;
}
