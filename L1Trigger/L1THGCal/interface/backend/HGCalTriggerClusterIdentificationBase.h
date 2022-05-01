#ifndef __L1Trigger_L1THGCal_HGCalTriggerClusterIdentificationBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerClusterIdentificationBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

class HGCalTriggerClusterIdentificationBase {
public:
  HGCalTriggerClusterIdentificationBase(){};
  virtual ~HGCalTriggerClusterIdentificationBase(){};
  virtual void initialize(const edm::ParameterSet& conf) = 0;
  virtual float value(const l1t::HGCalMulticluster& cluster) const = 0;
  virtual bool decision(const l1t::HGCalMulticluster& cluster, unsigned wp = 0) const = 0;
  virtual const std::vector<std::string>& working_points() const = 0;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalTriggerClusterIdentificationBase*()> HGCalTriggerClusterIdentificationFactory;

#define DEFINE_HGC_TPG_CLUSTER_ID(type, name) DEFINE_EDM_PLUGIN(HGCalTriggerClusterIdentificationFactory, type, name)

#endif
