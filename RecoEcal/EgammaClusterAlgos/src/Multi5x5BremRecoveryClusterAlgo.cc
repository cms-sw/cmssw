#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5BremRecoveryClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/BremRecoveryPhiRoadAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

reco::SuperClusterCollection Multi5x5BremRecoveryClusterAlgo::makeSuperClusters(
    const reco::CaloClusterPtrVector& clustersCollection) {
  const float etaBorder = 1.479;
  superclusters_v.clear();

  // create vectors of references to clusters of a specific origin...
  reco::CaloClusterPtrVector islandClustersBarrel_v;
  reco::CaloClusterPtrVector islandClustersEndCap_v;

  // ...and populate them:
  for (auto const& cluster_p : clustersCollection) {
    if (cluster_p->algo() == reco::CaloCluster::multi5x5) {
      if (fabs(cluster_p->position().eta()) < etaBorder) {
        islandClustersBarrel_v.push_back(cluster_p);
      } else {
        islandClustersEndCap_v.push_back(cluster_p);
      }
    }
  }

  // make the superclusters from the Barrel clusters - Island
  makeIslandSuperClusters(islandClustersBarrel_v, eb_rdeta_, eb_rdphi_);
  // make the superclusters from the EndCap clusters - Island
  makeIslandSuperClusters(islandClustersEndCap_v, ec_rdeta_, ec_rdphi_);

  return superclusters_v;
}

#include "DataFormats/Math/interface/Vector3D.h"

void Multi5x5BremRecoveryClusterAlgo::makeIslandSuperClusters(reco::CaloClusterPtrVector& clusters_v,
                                                              double etaRoad,
                                                              double phiRoad) {
  if (clusters_v.empty())
    return;

  const auto clustersSize = clusters_v.size();
  assert(clustersSize > 0);

  bool usedSeed[clustersSize];
  for (auto ic = 0U; ic < clustersSize; ++ic)
    usedSeed[ic] = false;

  float eta[clustersSize], phi[clustersSize], et[clustersSize];
  for (auto ic = 0U; ic < clustersSize; ++ic) {
    eta[ic] = clusters_v[ic]->eta();
    phi[ic] = clusters_v[ic]->phi();
    et[ic] = clusters_v[ic]->energy() * sin(clusters_v[ic]->position().theta());
  }

  for (auto is = 0U; is < clustersSize; ++is) {
    // check this seed was not already used
    if (usedSeed[is])
      continue;
    auto const& currentSeed = clusters_v[is];

    // Does our highest energy cluster have high enough energy?
    // changed this to continue from break (to be robust against the order of sorting of the seed clusters)
    if (et[is] < seedTransverseEnergyThreshold)
      continue;

    // if yes, make it a seed for a new SuperCluster:
    double energy = (currentSeed)->energy();
    math::XYZVector position_(
        (currentSeed)->position().X(), (currentSeed)->position().Y(), (currentSeed)->position().Z());
    position_ *= energy;
    usedSeed[is] = true;

    LogTrace("EcalClusters") << "*****************************";
    LogTrace("EcalClusters") << "******NEW SUPERCLUSTER*******";
    LogTrace("EcalClusters") << "Seed R = " << (currentSeed)->position().Rho();

    reco::CaloClusterPtrVector constituentClusters;
    constituentClusters.push_back(currentSeed);
    auto ic = is + 1;

    while (ic < clustersSize) {
      auto const& currentCluster = clusters_v[ic];

      // if dynamic phi road is enabled then compute the phi road for a cluster
      // of energy of existing clusters + the candidate cluster.
      if (dynamicPhiRoad_)
        phiRoad = phiRoadAlgo_->endcapPhiRoad(energy + (currentCluster)->energy());

      auto dphi = [](float p1, float p2) {
        auto dp = std::abs(p1 - p2);
        if (dp > float(M_PI))
          dp -= float(2 * M_PI);
        return std::abs(dp);
      };

      auto match = [&](int i, int j) {
        return (dphi(phi[i], phi[j]) < phiRoad) && (std::abs(eta[i] - eta[j]) < etaRoad);
      };

      // does the cluster match the phi road for this candidate supercluster
      if (!usedSeed[ic] && match(is, ic)) {
        // add basic cluster to supercluster constituents
        constituentClusters.push_back(currentCluster);
        energy += (currentCluster)->energy();
        position_ += (currentCluster)->energy() * math::XYZVector((currentCluster)->position().X(),
                                                                  (currentCluster)->position().Y(),
                                                                  (currentCluster)->position().Z());
        // remove cluster from vector of available clusters
        usedSeed[ic] = true;
        LogTrace("EcalClusters") << "Cluster R = " << (currentCluster)->position().Rho();
      }
      ++ic;
    }

    position_ /= energy;

    LogTrace("EcalClusters") << "Final SuperCluster R = " << position_.Rho();

    reco::SuperCluster newSuperCluster(
        energy, math::XYZPoint(position_.X(), position_.Y(), position_.Z()), currentSeed, constituentClusters);

    superclusters_v.push_back(newSuperCluster);
    LogTrace("EcalClusters") << "created a new supercluster of: ";
    LogTrace("EcalClusters") << "Energy = " << newSuperCluster.energy();
    LogTrace("EcalClusters") << "Position in (R, phi, theta) = (" << newSuperCluster.position().Rho() << ", "
                             << newSuperCluster.position().phi() << ", " << newSuperCluster.position().theta() << ")";
  }

  clusters_v.clear();
}
