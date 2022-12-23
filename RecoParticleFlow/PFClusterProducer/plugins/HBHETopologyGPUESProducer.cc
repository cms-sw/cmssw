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
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPU.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

typedef PFHCALDenseIdNavigator<HcalDetId, HcalTopology, false> PFRecHitHCALDenseIdNavigator;

class HBHETopologyGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HBHETopologyGPUESProducer(edm::ParameterSet const&);
  ~HBHETopologyGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<HBHETopologyGPU> produce(JobConfigurationGPURecord const&);
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
  edm::ESHandle<CaloGeometry> geoHandle;
  edm::ESHandle<HcalTopology> topoHandle;

  std::unique_ptr<PFRecHitNavigatorBase> navigator_;

  std::unique_ptr<const HcalTopology> topology_;

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

  findingRecord<JobConfigurationGPURecord>();

  std::cout << "HBHETopologyGPUESProducer::HBHETopologyGPUESProducer" << std::endl;

}

void HBHETopologyGPUESProducer::beginRun(edm::Run const& r, edm::EventSetup const& setup) {

  /*
  navigator_->init(setup);
  if (!theRecNumberWatcher_.check(setup))
    return;
  */

  topoHandle = setup.getHandle(hcalToken_);
  topology_.release();
  topology_.reset(topoHandle.product());

  //
  // Get list of valid Det Ids for HCAL barrel & endcap once
  geoHandle = setup.getHandle(geomToken_);
  // get the hcal geometry
  const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  const std::vector<DetId>& validBarrelDetIds = hcalBarrelGeo->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& validEndcapDetIds = hcalEndcapGeo->getValidDetIds(DetId::Hcal, HcalEndcap);

  std::cout << "HBHETopologyGPUESProducer test" << validBarrelDetIds.size() << " " << validEndcapDetIds.size() << std::endl;

  //
  // KH to be stored somewhere
  //
  //cudaConstants.nValidBarrelIds = validBarrelDetIds.size();
  //cudaConstants.nValidEndcapIds = validEndcapDetIds.size();
  //nValidDetIds = cudaConstants.nValidBarrelIds + cudaConstants.nValidEndcapIds;
  //cudaConstants.nValidDetIds = nValidDetIds;

  // std::cout << "Found nValidBarrelIds = " << cudaConstants.nValidBarrelIds
  //        << "\tnValidEndcapIds = " << cudaConstants.nValidEndcapIds << std::endl;

  /*

  vDenseIdHcal = reinterpret_cast<PFRecHitHCALDenseIdNavigator*>(&(*navigator_))->getValidDenseIds();
  //std::cout << "Found vDenseIdHcal->size() = " << vDenseIdHcal->size() << std::endl;

  // Fill a vector of cell neighbours
  denseIdHcalMax_ = *std::max_element(vDenseIdHcal->begin(), vDenseIdHcal->end());
  denseIdHcalMin_ = *std::min_element(vDenseIdHcal->begin(), vDenseIdHcal->end());
  //std::cout << denseIdHcalMax_ << " " << denseIdHcalMin_ << std::endl;

  nDenseIdsInRange = denseIdHcalMax_ - denseIdHcalMin_ + 1;
  cudaConstants.nDenseIdsInRange = nDenseIdsInRange;
  cudaConstants.denseIdHcalMin = denseIdHcalMin_;

  validDetIdPositions.clear();
  validDetIdPositions.reserve(nValidDetIds);
  detIdToCell.clear();
  detIdToCell.reserve(nValidDetIds);
  for (const auto& denseid : *vDenseIdHcal) {
    DetId detid_c = topology_.get()->denseId2detId(denseid);
    HcalDetId hid_c = HcalDetId(detid_c);

    //DetId detId = topology_.get()->denseId2detId(denseId);
    //HcalDetId hid(detId.rawId());

    if (hid_c.subdet() == HcalBarrel)
      validDetIdPositions.emplace_back(hcalBarrelGeo->getGeometry(detid_c)->getPosition());
    else if (hid_c.subdet() == HcalEndcap)
      validDetIdPositions.emplace_back(hcalEndcapGeo->getGeometry(detid_c)->getPosition());
    else
      std::cout << "Invalid subdetector found for detId " << hid_c.rawId() << ": " << hid_c.subdet() << std::endl;

    std::shared_ptr<const CaloCellGeometry> thisCell = nullptr;
    //PFLayer::Layer layer = PFLayer::HCAL_BARREL1;
    switch (hid_c.subdet()) {
    case HcalBarrel:
      thisCell = hcalBarrelGeo->getGeometry(hid_c);
      //layer = PFLayer::HCAL_BARREL1;
      break;

    case HcalEndcap:
      thisCell = hcalEndcapGeo->getGeometry(hid_c);
      //layer = PFLayer::HCAL_ENDCAP;
      break;
    default:
      break;
    }

    detIdToCell[hid_c.rawId()] = thisCell;

  }
  // -> vDenseIdHcal, validDetIdPositions

  */

}

void HBHETopologyGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                       const edm::IOVSyncValue& iTime,
                                                       edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void HBHETopologyGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;
  //d.add<std::vector<int>>("pulseOffsets", {-3, -2, -1, 0, 1, 2, 3, 4});
  desc.addWithDefaultLabel(d);
}

std::unique_ptr<HBHETopologyGPU> HBHETopologyGPUESProducer::produce(JobConfigurationGPURecord const&) {

  //auto geom = iRecord.getTransientHandle(geometryToken_);
  std::cout << "HBHETopologyGPUESProducer::produce" << std::endl;
  return std::make_unique<HBHETopologyGPU>(pset_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(HBHETopologyGPUESProducer);
