#ifndef __L1Trigger_L1THGCal_HGCalTriggerClusterInterpreterBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerClusterInterpreterBase_h__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

class HGCalTriggerClusterInterpreterBase {
public:
  HGCalTriggerClusterInterpreterBase(){};
  virtual ~HGCalTriggerClusterInterpreterBase(){};
  virtual void initialize(const edm::ParameterSet& conf) = 0;
  virtual void eventSetup(const edm::EventSetup& es) = 0;
  virtual void interpret(l1t::HGCalMulticlusterBxCollection& multiclusters) const = 0;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalTriggerClusterInterpreterBase*()> HGCalTriggerClusterInterpreterFactory;

#define DEFINE_HGC_TPG_CLUSTER_INTERPRETER(type, name) \
  DEFINE_EDM_PLUGIN(HGCalTriggerClusterInterpreterFactory, type, name)

#endif
