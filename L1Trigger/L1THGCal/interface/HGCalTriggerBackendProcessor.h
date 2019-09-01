#ifndef __L1Trigger_L1THGCal_HGCalTriggerBackendProcessor_h__
#define __L1Trigger_L1THGCal_HGCalTriggerBackendProcessor_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiDefs.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"

#include <vector>
#include <memory>

/*******
 *
 * class:  HGCalTriggerBackendProcessor
 * author: L.Gray (FNAL)
 * date:   12 August, 2015
 *
 * This class contains a list of backend algorithms
 * that are run in the sequence defined in python (order of pset vector).
 * This is mostly a wrapper class that manages calling the algorithms.
 *
 *******/

class HGCalTriggerBackendProcessor {
public:
  typedef std::unique_ptr<HGCalTriggerBackendAlgorithmBase> algo_ptr;

  HGCalTriggerBackendProcessor(const edm::ParameterSet& conf, edm::ConsumesCollector&& cc);

  void setGeometry(const HGCalTriggerGeometryBase* const geom);

  void setProduces(edm::stream::EDProducer<>& prod) const;

  void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, edm::Event& e);

  void putInEvent(edm::Event& evt);

  void reset();

private:
  std::vector<algo_ptr> algorithms_;
};

#endif
