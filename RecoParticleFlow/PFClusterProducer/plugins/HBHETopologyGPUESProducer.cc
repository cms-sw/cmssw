#include <array>
#include <iostream>
#include <tuple>
#include <utility>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
//KH #include "FWCore/Framework/interface/EventSetup.h"
//KH #include "FWCore/Framework/interface/ESHandle.h"
//KH #include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
//KH #include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/typelookup.h"
//
//#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPURcd.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPU.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

typedef PFHCALDenseIdNavigator<HcalDetId, HcalTopology, false> PFRecHitHCALDenseIdNavigator;

class HBHETopologyGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
  //class HBHETopologyGPUESProducer : public edm::ESProducer {
public:
  HBHETopologyGPUESProducer(edm::ParameterSet const&);
  ~HBHETopologyGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<HBHETopologyGPU> produce(HBHETopologyGPURcd const&);
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
  //edm::ESHandle<CaloGeometry> geoHandle;
  //edm::ESHandle<HcalTopology> topoHandle;

  //std::unique_ptr<PFRecHitNavigatorBase> navigator_;
  //std::unique_ptr<const HcalTopology> topology_;

};

HBHETopologyGPUESProducer::HBHETopologyGPUESProducer(edm::ParameterSet const& pset) : pset_{pset} {
      //hcalToken_(esConsumes<edm::Transition::BeginRun>()),
      //geomToken_(iC.esConsumes()){

  auto cc = setWhatProduced(this);

  //
  // navigator-related parameters
  //const auto& navSet = pset.getParameterSet("navigator");
  //edm::ConsumesCollector& ccref = *cc;
  //navigator_ = PFRecHitNavigationFactory::get()->create(navSet.getParameter<std::string>("name"), navSet, cc);

  hcalToken_ = cc.consumes();
  geomToken_ = cc.consumes();

  //isUsingRecord<HBHETopologyGPURcd>();

  std::cout << "HBHETopologyGPUESProducer::HBHETopologyGPUESProducer" << std::endl;

}


void HBHETopologyGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                       const edm::IOVSyncValue& iTime,
                                                       edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void HBHETopologyGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;
  d.add<std::vector<uint32_t>>("pulseOffsets", { 3, 2, 1, 0, 1, 2, 3, 4});
  d.add<std::vector<int>>("pulseOffsets2", {-3, -2, -1, 0, 1, 2, 3, 4});
  desc.addWithDefaultLabel(d);
}

std::unique_ptr<HBHETopologyGPU> HBHETopologyGPUESProducer::produce(HBHETopologyGPURcd const& iRecord) {

  std::cout << "HBHETopologyGPUESProducer::produce" << std::endl;

  // geoHandle = iRecord.getHandle(geomToken_);
  // topoHandle = iRecord.getHandle(hcalToken_);

  // Handles
  auto geom = iRecord.getHandle(geomToken_);
  auto topo = iRecord.getHandle(hcalToken_);

  return std::make_unique<HBHETopologyGPU>(pset_,*geom,*topo);
  //return std::make_unique<SiPixelGainCalibrationForHLTGPU>(*gains, *geom);
}

DEFINE_FWK_EVENTSETUP_SOURCE(HBHETopologyGPUESProducer);
