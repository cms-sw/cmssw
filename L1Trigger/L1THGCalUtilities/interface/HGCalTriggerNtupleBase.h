#ifndef __L1Trigger_L1THGCalUtilities_HGCalTriggerNtupleBase_h__
#define __L1Trigger_L1THGCalUtilities_HGCalTriggerNtupleBase_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"
// #include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
// #include "MagneticField/Engine/interface/MagneticField.h"
// #include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "TTree.h"

// #include "MagneticField/Engine/interface/MagneticField.h"
// #include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
// #include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
// #include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
// #include "FastSimulation/Event/interface/FSimEvent.h"
// #include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"
//
// #include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

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
  HGCalTriggerNtupleBase(const edm::ParameterSet& conf){};
  virtual ~HGCalTriggerNtupleBase(){};
  virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) = 0;
  virtual void fill(const edm::Event&, const HGCalTriggerNtupleEventSetup&) = 0;

protected:
  virtual void clear() = 0;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalTriggerNtupleBase*(const edm::ParameterSet&)> HGCalTriggerNtupleFactory;

#endif
