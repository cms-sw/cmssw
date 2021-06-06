#include <algorithm>
#include <set>
#include <vector>

#include "SeedingRegionByHF.h"

using namespace ticl;

SeedingRegionByHF::SeedingRegionByHF(const edm::ParameterSet &conf, edm::ConsumesCollector &sumes)
    : SeedingRegionAlgoBase(conf, sumes),
      hfhits_token_(sumes.consumes<HFRecHitCollection>(conf.getParameter<edm::InputTag>("hits"))),
      minAbsEta_(conf.getParameter<double>("minAbsEta")),
      maxAbsEta_(conf.getParameter<double>("maxAbsEta")),
      minEt_(conf.getParameter<double>("minEt")) {
  geo_token_ = sumes.esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>();
}

SeedingRegionByHF::~SeedingRegionByHF() {}

void SeedingRegionByHF::initialize(const edm::EventSetup &es) { geometry_ = &es.getData(geo_token_); }

void SeedingRegionByHF::makeRegions(const edm::Event &ev,
                                    const edm::EventSetup &es,
                                    std::vector<TICLSeedingRegion> &result) {
  const auto &recHits = ev.get(hfhits_token_);

  for (const auto &erh : recHits) {
    const HcalDetId &detid = (HcalDetId)erh.detid();
    if (erh.energy() < minEt_)
      continue;

    const GlobalPoint &globalPosition =
        geometry_->getSubdetectorGeometry(DetId::Hcal, HcalForward)->getGeometry(detid)->getPosition(detid);
    auto eta = globalPosition.eta();

    if (std::abs(eta) < minAbsEta_ || std::abs(eta) > maxAbsEta_)
      continue;

    int iSide = int(eta > 0);
    int idx = 0;
    edm::ProductID hfSeedId = edm::ProductID(detid.rawId());

    auto phi = globalPosition.phi();
    double theta = 2 * atan(exp(eta));
    result.emplace_back(
        globalPosition, GlobalVector(GlobalVector::Polar(theta, phi, erh.energy())), iSide, idx, hfSeedId);
  }

  // sorting seeding region by descending momentum
  std::sort(result.begin(), result.end(), [](const TICLSeedingRegion &a, const TICLSeedingRegion &b) {
    return a.directionAtOrigin.perp2() > b.directionAtOrigin.perp2();
  });
}

void SeedingRegionByHF::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<edm::InputTag>("hits", edm::InputTag("hfreco"));
  desc.add<int>("algo_verbosity", 0);
  desc.add<double>("minAbsEta", 3.0);
  desc.add<double>("maxAbsEta", 4.0);
  desc.add<double>("minEt", 5);
  SeedingRegionAlgoBase::fillPSetDescription(desc);
}
