#include <array>
#include <iostream>
#include <tuple>
#include <utility>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/typelookup.h"
//
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHETopologyGPURcd.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHETopologyGPU.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

typedef PFHCALDenseIdNavigator<HcalDetId, HcalTopology, false> PFRecHitHCALDenseIdNavigator;

class PFHBHETopologyGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
  //class PFHBHETopologyGPUESProducer : public edm::ESProducer {
public:
  PFHBHETopologyGPUESProducer(edm::ParameterSet const&);
  ~PFHBHETopologyGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<PFHBHETopologyGPU> produce(PFHBHETopologyGPURcd const&);
  void beginRun(edm::Run const&, edm::EventSetup const&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  edm::ParameterSet const& pset_;

  // HCAL geometry/topology
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

PFHBHETopologyGPUESProducer::PFHBHETopologyGPUESProducer(edm::ParameterSet const& pset) : pset_{pset} {
  auto cc = setWhatProduced(this);

  hcalToken_ = cc.consumes();
  geomToken_ = cc.consumes();
}

void PFHBHETopologyGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                 const edm::IOVSyncValue& iTime,
                                                 edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void PFHBHETopologyGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;
  d.add<std::vector<int>>("hcalEnums", {1, 2});  // HB+HE
  desc.addWithDefaultLabel(d);
}

std::unique_ptr<PFHBHETopologyGPU> PFHBHETopologyGPUESProducer::produce(PFHBHETopologyGPURcd const& iRecord) {
  // Handles
  auto geom = iRecord.getHandle(geomToken_);
  auto topo = iRecord.getHandle(hcalToken_);

  return std::make_unique<PFHBHETopologyGPU>(pset_, *geom, *topo);
}

DEFINE_FWK_EVENTSETUP_SOURCE(PFHBHETopologyGPUESProducer);
