#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCal3DClustering.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace {
  std::vector<size_t> sorted_indices(const reco::HGCalMultiCluster::ClusterCollection &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(std::begin(idx), std::end(idx), 0);

    // sort indices based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return (*v[i1]) > (*v[i2]); });

    return idx;
  }

  float distReal2(const edm::Ptr<reco::BasicCluster> &a, const std::array<double, 3> &to) {
    return (a->x() - to[0]) * (a->x() - to[0]) + (a->y() - to[1]) * (a->y() - to[1]);
  }
}  // namespace

void HGCal3DClustering::organizeByLayer(const reco::HGCalMultiCluster::ClusterCollection &thecls) {
  es = sorted_indices(thecls);
  unsigned int es_size = es.size();
  for (unsigned int i = 0; i < es_size; ++i) {
    int layer = rhtools_.getLayerWithOffset(thecls[es[i]]->hitsAndFractions()[0].first);
    layer += int(thecls[es[i]]->z() > 0) * (maxlayer + 1);
    float x = thecls[es[i]]->x();
    float y = thecls[es[i]]->y();
    float z = thecls[es[i]]->z();
    points[layer].emplace_back(ClusterRef(i, z), x, y);
    if (zees[layer] == 0.) {
      // At least one cluster for layer at z
      zees[layer] = z;
    }
    if (points[layer].empty()) {
      minpos[layer][0] = x;
      minpos[layer][1] = y;
      maxpos[layer][0] = x;
      maxpos[layer][1] = y;
    } else {
      minpos[layer][0] = std::min(x, minpos[layer][0]);
      minpos[layer][1] = std::min(y, minpos[layer][1]);
      maxpos[layer][0] = std::max(x, maxpos[layer][0]);
      maxpos[layer][1] = std::max(y, maxpos[layer][1]);
    }
  }
}
std::vector<reco::HGCalMultiCluster> HGCal3DClustering::makeClusters(
    const reco::HGCalMultiCluster::ClusterCollection &thecls) {
  reset();
  organizeByLayer(thecls);
  std::vector<reco::HGCalMultiCluster> thePreClusters;

  std::vector<KDTree> hit_kdtree(2 * (maxlayer + 1));
  for (unsigned int i = 0; i <= 2 * maxlayer + 1; ++i) {
    KDTreeBox bounds(minpos[i][0], maxpos[i][0], minpos[i][1], maxpos[i][1]);
    hit_kdtree[i].build(points[i], bounds);
  }
  std::vector<int> vused(es.size(), 0);

  unsigned int es_size = es.size();
  for (unsigned int i = 0; i < es_size; ++i) {
    if (vused[i] == 0) {
      reco::HGCalMultiCluster temp;
      temp.push_back(thecls[es[i]]);
      vused[i] = (thecls[es[i]]->z() > 0) ? 1 : -1;
      // Starting from cluster es[i] at from[0] - from[1] - from[2]
      std::array<double, 3> from{{thecls[es[i]]->x(), thecls[es[i]]->y(), thecls[es[i]]->z()}};
      unsigned int firstlayer = int(thecls[es[i]]->z() > 0) * (maxlayer + 1);
      unsigned int lastlayer = firstlayer + maxlayer + 1;
      for (unsigned int j = firstlayer; j < lastlayer; ++j) {
        if (zees[j] == 0.) {
          // layer j not yet ever reached?
          continue;
        }
        std::array<double, 3> to{{0., 0., zees[j]}};
        layerIntersection(to, from);
        unsigned int layer =
            j > maxlayer ? (j - (maxlayer + 1)) : j;  //maps back from index used for KD trees to actual layer
        float radius = radii[2];
        if (layer <= rhtools_.lastLayerEE())
          radius = radii[0];
        else if (layer < rhtools_.firstLayerBH())
          radius = radii[1];
        float radius2 = radius * radius;
        KDTreeBox search_box(
            float(to[0]) - radius, float(to[0]) + radius, float(to[1]) - radius, float(to[1]) + radius);
        std::vector<ClusterRef> found;
        // at layer j in box float(to[0])+/-radius - float(to[1])+/-radius
        hit_kdtree[j].search(search_box, found);
        // found found.size() clusters within box
        for (unsigned int k = 0; k < found.size(); k++) {
          if (vused[found[k].ind] == 0 && distReal2(thecls[es[found[k].ind]], to) < radius2) {
            temp.push_back(thecls[es[found[k].ind]]);
            vused[found[k].ind] = vused[i];
          }
        }
      }
      if (temp.size() > minClusters) {
        math::XYZPoint position = clusterTools->getMultiClusterPosition(temp);
        if (std::abs(position.z()) <= 0.)
          continue;
        // only store multiclusters that pass the energy threshold in getMultiClusterPosition
        // giving them a position inside the HGCal
        thePreClusters.push_back(temp);
        auto &back = thePreClusters.back();
        back.setPosition(position);
        back.setEnergy(clusterTools->getMultiClusterEnergy(back));
      }
    }
  }

  return thePreClusters;
}

void HGCal3DClustering::layerIntersection(std::array<double, 3> &to, const std::array<double, 3> &from) const {
  if (from[2] != 0) {
    to[0] = from[0] / from[2] * to[2];
    to[1] = from[1] / from[2] * to[2];
  } else {
    to[0] = 0;
    to[1] = 0;
  }
}
