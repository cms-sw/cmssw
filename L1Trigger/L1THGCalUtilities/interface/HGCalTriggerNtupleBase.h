#ifndef __L1Trigger_L1THGCalUtilities_HGCalTriggerNtupleBase_h__
#define __L1Trigger_L1THGCalUtilities_HGCalTriggerNtupleBase_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TTree.h"

namespace HepPDT {
  class ParticleDataTable;
}
class MagneticField;
class HGCalTriggerGeometryBase;

struct HGCalTriggerNtupleEventSetup {
  edm::ESHandle<HepPDT::ParticleDataTable> pdt;
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
};

class HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleBase(const edm::ParameterSet& conf) : name_(conf.getParameter<std::string>("NtupleName")){};
  virtual ~HGCalTriggerNtupleBase(){};
  const std::string& name() const { return name_; }
  virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) = 0;
  virtual void fill(const edm::Event&, const HGCalTriggerNtupleEventSetup&) {
    edm::LogWarning("NotImplemented") << "Calling ntuplizer fill(edm::Event, HGCalTriggerNtupleEventSetup), but it is "
                                         "not implemented in the concrete class '"
                                      << name() << "'. "
                                      << "You might want to set 'accessEventSetup_' to true in order to call "
                                         "fill(edm::Event, edm::EventSetup) instead.";
  }
  // Kept for backward compatibility: used in L1Trigger/L1CaloTrigger/test
  virtual void fill(const edm::Event&, const edm::EventSetup&) {
    edm::LogWarning("NotImplemented")
        << "Calling ntuplizer fill(edm::Event, edm::EventSetup), but it is not implemented in the concrete class '"
        << name() << "'. "
        << "You might want to set 'accessEventSetup_' to false in order to call fill(edm::Event, "
           "HGCalTriggerNtupleEventSetup) instead.";
  }
  bool accessEventSetup() const { return accessEventSetup_; }

protected:
  virtual void clear() = 0;
  bool accessEventSetup_ = true;
  const std::string name_;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalTriggerNtupleBase*(const edm::ParameterSet&)> HGCalTriggerNtupleFactory;

#endif
