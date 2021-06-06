// Author: Arabella Martelli, Felice Pantaleo, Marco Rovere
// arabella.martelli@cern.ch, felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 06/2019
#include <algorithm>
#include <set>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SeedingRegionGlobal.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

using namespace ticl;

SeedingRegionGlobal::SeedingRegionGlobal(const edm::ParameterSet &conf, edm::ConsumesCollector &sumes)
    : SeedingRegionAlgoBase(conf, sumes) {}

SeedingRegionGlobal::~SeedingRegionGlobal(){};

void SeedingRegionGlobal::makeRegions(const edm::Event &ev,
                                      const edm::EventSetup &es,
                                      std::vector<TICLSeedingRegion> &result) {
  // for unseeded iterations create 2 global seeding regions
  // one for each endcap
  for (int i = 0; i < 2; ++i) {
    result.emplace_back(GlobalPoint(0., 0., 0.), GlobalVector(0., 0., 0.), i, -1, edm::ProductID());
  }
}

void SeedingRegionGlobal::fillPSetDescription(edm::ParameterSetDescription &desc) {
  SeedingRegionAlgoBase::fillPSetDescription(desc);
}
