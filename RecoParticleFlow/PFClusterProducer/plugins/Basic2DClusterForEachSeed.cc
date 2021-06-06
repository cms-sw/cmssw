#include "Basic2DClusterForEachSeed.h"

void Basic2DClusterForEachSeed::buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
                                              const std::vector<bool>& rechitMask,
                                              const std::vector<bool>& seedable,
                                              reco::PFClusterCollection& output) {
  auto const& hits = *input;

  // loop over seeds and make clusters
  reco::PFCluster cluster;
  for (unsigned int hit = 0; hit < hits.size(); ++hit) {
    if (!rechitMask[hit] || !seedable[hit])
      continue;  // if not seed, ignore.
    cluster.reset();

    // seed
    auto refhit = makeRefhit(input, hit);
    auto rhf = reco::PFRecHitFraction(refhit, 1.0);  // entire rechit energy should go to a cluster

    // add the hit to the cluster
    cluster.addRecHitFraction(rhf);

    // extract
    const auto rh_energy = refhit->energy();

    // fill cluster information
    cluster.setSeed(refhit->detId());
    cluster.setEnergy(rh_energy);
    cluster.setTime(refhit->time());
    cluster.setLayer(refhit->layer());
    cluster.setPosition(math::XYZPoint(refhit->position().x(), refhit->position().y(), refhit->position().z()));
    cluster.calculatePositionREP();
    cluster.setDepth(refhit->depth());

    output.push_back(cluster);

  }  // looping over seeds ends
}
