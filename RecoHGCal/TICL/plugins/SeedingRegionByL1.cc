/*
   Author: Swagata Mukherjee

   Date: Feb 2021

   TICL is currently seeded by tracks, or just globally.
   Here, adding option to seed TICL by L1 e/gamma objects (L1 TkEm).
   This is expected to be useful for CPU timing at the HLT.
*/

#include "SeedingRegionByL1.h"

#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ticl::SeedingRegionByL1::SeedingRegionByL1(const edm::ParameterSet &conf, edm::ConsumesCollector &sumes)
    : SeedingRegionAlgoBase(conf, sumes),
      l1GTCandsToken_(sumes.consumes<l1t::P2GTCandidateCollection>(conf.getParameter<edm::InputTag>("l1GTCandColl"))),
      algoVerbosity_(conf.getParameter<int>("algo_verbosity")),
      minPt_(conf.getParameter<double>("minPt")),
      minAbsEta_(conf.getParameter<double>("minAbsEta")),
      maxAbsEta_(conf.getParameter<double>("maxAbsEta")),
      quality_(conf.getParameter<int>("quality")),
      qualityIsMask_(conf.getParameter<bool>("qualityIsMask")),
      applyQuality_(conf.getParameter<bool>("applyQuality")) {}

void ticl::SeedingRegionByL1::makeRegions(const edm::Event &ev,
                                          const edm::EventSetup &es,
                                          std::vector<TICLSeedingRegion> &result) {
  auto l1GTCands = ev.getHandle(l1GTCandsToken_);
  edm::ProductID l1gtcandsId = l1GTCands.id();

  for (size_t indx = 0; indx < (*l1GTCands).size(); indx++) {
    const auto &l1GTCand = (*l1GTCands)[indx];
    double offlinePt = l1GTCand.pt();
    bool passQuality(false);

    if (applyQuality_) {
      if (qualityIsMask_) {
        passQuality = (l1GTCand.hwQual() & quality_);
      } else {
        passQuality = (l1GTCand.hwQual() == quality_);
      }
    } else {
      passQuality = true;
    }

    if ((offlinePt < minPt_) || (std::abs(l1GTCand.eta()) < minAbsEta_) || (std::abs(l1GTCand.eta()) > maxAbsEta_) ||
        !passQuality) {
      continue;
    }

    int iSide = int(l1GTCand.eta() > 0);
    result.emplace_back(GlobalPoint(l1GTCand.p4().X(), l1GTCand.p4().Y(), l1GTCand.p4().Z()),
                        GlobalVector(l1GTCand.px(), l1GTCand.py(), l1GTCand.pz()),
                        iSide,
                        indx,
                        l1gtcandsId);
  }

  std::sort(result.begin(), result.end(), [](const TICLSeedingRegion &a, const TICLSeedingRegion &b) {
    return a.directionAtOrigin.perp2() > b.directionAtOrigin.perp2();
  });
}

void ticl::SeedingRegionByL1::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<edm::InputTag>("l1GTCandColl", edm::InputTag("L1TkPhotonsHGC", "EG"));
  desc.add<double>("minPt", 10);
  desc.add<double>("minAbsEta", 1.479);
  desc.add<double>("maxAbsEta", 4.0);
  desc.add<int>("quality", 5);
  desc.add<bool>("qualityIsMask", false);
  desc.add<bool>("applyQuality", false);
  SeedingRegionAlgoBase::fillPSetDescription(desc);
}
