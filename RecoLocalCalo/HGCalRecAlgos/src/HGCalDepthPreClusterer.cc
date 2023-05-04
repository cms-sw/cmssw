#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <list>

namespace {
  std::vector<size_t> sorted_indices(const reco::HGCalMultiCluster::ClusterCollection &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
      idx[i] = i;

    // sort indices based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return (*v[i1]) > (*v[i2]); });

    return idx;
  }

  float dist2(const edm::Ptr<reco::BasicCluster> &a, const edm::Ptr<reco::BasicCluster> &b) {
    return reco::deltaR2(*a, *b);
  }

  //get distance between cluster and multicluster axis (defined by remaning cluster with highest energy)
  // N.B. the order of the clusters matters
  float distAxisCluster2(const edm::Ptr<reco::BasicCluster> &a, const edm::Ptr<reco::BasicCluster> &b) {
    float tanTheta = tan(2 * atan(exp(-1 * a->eta())));
    float ax = b->z() * tanTheta * cos(a->phi());
    float ay = b->z() * tanTheta * sin(a->phi());
    return (ax - b->x()) * (ax - b->x()) + (ay - b->y()) * (ay - b->y());
  }
}  // namespace

std::vector<reco::HGCalMultiCluster> HGCalDepthPreClusterer::makePreClusters(
    const reco::HGCalMultiCluster::ClusterCollection &thecls) const {
  std::vector<reco::HGCalMultiCluster> thePreClusters;
  std::vector<size_t> es = sorted_indices(thecls);
  std::vector<int> vused(es.size(), 0);

  for (unsigned int i = 0; i < es.size(); ++i) {
    if (vused[i] == 0) {
      reco::HGCalMultiCluster temp;
      temp.push_back(thecls[es[i]]);
      vused[i] = (thecls[es[i]]->z() > 0) ? 1 : -1;
      for (unsigned int j = i + 1; j < es.size(); ++j) {
        if (vused[j] == 0) {
          float distanceCheck = 9999.;
          if (realSpaceCone)
            distanceCheck = distAxisCluster2(thecls[es[i]], thecls[es[j]]);
          else
            distanceCheck = dist2(thecls[es[i]], thecls[es[j]]);
          DetId detid = thecls[es[j]]->hitsAndFractions()[0].first();
          unsigned int layer = clusterTools->getLayer(detid);
          float radius = radii[2];
          if (layer <= rhtools_.lastLayerEE())
            radius = radii[0];
          else if (layer < rhtools_.firstLayerBH())
            radius = radii[1];
          float radius2 = radius * radius;
          if (distanceCheck<radius2 &&int(thecls[es[j]]->z() * vused[i])> 0) {
            temp.push_back(thecls[es[j]]);
            vused[j] = vused[i];
          }
        }
      }
      if (temp.size() > minClusters) {
        thePreClusters.push_back(temp);
        auto &back = thePreClusters.back();
        back.setPosition(clusterTools->getMultiClusterPosition(back));
        back.setEnergy(clusterTools->getMultiClusterEnergy(back));
      }
    }
  }

  return thePreClusters;
}
