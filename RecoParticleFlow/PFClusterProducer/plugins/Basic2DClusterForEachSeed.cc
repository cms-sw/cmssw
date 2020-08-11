#include "Basic2DClusterForEachSeed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void Basic2DClusterForEachSeed::buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
                                                  const std::vector<bool>& rechitMask,
                                                  const std::vector<bool>& seedable,
                                                  reco::PFClusterCollection& output) {
  auto const& hits = *input;
  std::vector<bool> used(hits.size(), false);
  std::vector<unsigned int> seeds;

  // get seeds
  seeds.reserve(hits.size());
  for (unsigned int i = 0; i < hits.size(); ++i) {
    if (!rechitMask[i] || !seedable[i])
      continue;
    seeds.emplace_back(i);
  }

  // loop over seeds and make clusters
  reco::PFCluster cluster;
  for (auto seed : seeds) {
    if (!rechitMask[seed] || !seedable[seed])
      continue;
    cluster.reset();

    // seed
    auto refhit = makeRefhit(input, seed);
    auto rhf = reco::PFRecHitFraction(refhit, 1.0);

    cluster.addRecHitFraction(rhf);

    if (!cluster.recHitFractions().empty()){

      //
      const auto rh_fraction = rhf.fraction();
      const auto rh_rawenergy = refhit->energy();
      const auto rh_energy = rh_rawenergy * rh_fraction;

      // fill cluster information
      cluster.setSeed(refhit->detId());
      cluster.setEnergy(rh_energy);
      cluster.setTime(refhit->time());
      cluster.setLayer(refhit->layer());
      cluster.setPosition(math::XYZPoint(refhit->position().x(),refhit->position().y(),refhit->position().z()));
      cluster.calculatePositionREP();
      cluster.setDepth(refhit->depth());
      cluster.calculatePositionREP();

      output.push_back(cluster);
    }

  } // looping over seeds ends

}
