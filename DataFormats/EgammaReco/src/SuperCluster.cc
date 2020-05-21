#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include <algorithm>
using namespace reco;

SuperCluster::SuperCluster(double energy, const math::XYZPoint& position)
    : CaloCluster(energy, position),
      preshowerEnergy_(0),
      rawEnergy_(0),
      phiWidth_(0),
      etaWidth_(0),
      preshowerEnergy1_(0),
      preshowerEnergy2_(0) {}

SuperCluster::SuperCluster(double energy,
                           const math::XYZPoint& position,
                           const CaloClusterPtr& seed,
                           const CaloClusterPtrVector& clusters,
                           double Epreshower,
                           double phiWidth,
                           double etaWidth,
                           double Epreshower1,
                           double Epreshower2)
    : CaloCluster(energy, position), rawEnergy_(0) {
  phiWidth_ = phiWidth;
  etaWidth_ = etaWidth;
  seed_ = seed;
  preshowerEnergy_ = Epreshower;
  preshowerEnergy1_ = Epreshower1;
  preshowerEnergy2_ = Epreshower2;

  // set references to constituent basic clusters and update list of rechits
  for (auto&& cluster : clusters) {
    clusters_.push_back((cluster));

    // updated list of used hits
    const std::vector<std::pair<DetId, float> >& v1 = (cluster)->hitsAndFractions();
    for (const auto& diIt : v1) {
      hitsAndFractions_.push_back(diIt);
    }  // loop over rechits
  }    // loop over basic clusters

  computeRawEnergy();
}

SuperCluster::SuperCluster(double energy,
                           const math::XYZPoint& position,
                           const CaloClusterPtr& seed,
                           const CaloClusterPtrVector& clusters,
                           const CaloClusterPtrVector& preshowerClusters,
                           double Epreshower,
                           double phiWidth,
                           double etaWidth,
                           double Epreshower1,
                           double Epreshower2)
    : CaloCluster(energy, position), rawEnergy_(-1.) {
  phiWidth_ = phiWidth;
  etaWidth_ = etaWidth;
  seed_ = seed;
  preshowerEnergy_ = Epreshower;
  preshowerEnergy1_ = Epreshower1;
  preshowerEnergy2_ = Epreshower2;

  // set references to constituent basic clusters and update list of rechits
  for (auto&& cluster : clusters) {
    clusters_.push_back((cluster));

    // updated list of used hits
    const std::vector<std::pair<DetId, float> >& v1 = (cluster)->hitsAndFractions();
    for (const auto& diIt : v1) {
      hitsAndFractions_.push_back(diIt);
    }  // loop over rechits
  }    // loop over basic clusters

  // set references to preshower clusters
  for (auto&& preshowerCluster : preshowerClusters) {
    preshowerClusters_.push_back((preshowerCluster));
  }
  computeRawEnergy();
}

void SuperCluster::computeRawEnergy() {
  rawEnergy_ = 0.;
  for (CaloClusterPtrVector::const_iterator bcItr = clustersBegin(); bcItr != clustersEnd(); bcItr++) {
    rawEnergy_ += (*bcItr)->energy();
  }
}
