#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalCLUEAlgo.h"

// Geometry
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "tbb/task_arena.h"
#include "tbb/tbb.h"

using namespace hgcal_clustering;

void HGCalCLUEAlgo::populate(const HGCRecHitCollection &hits) {
  // loop over all hits and create the Hexel structure, skip energies below ecut

  if (dependSensor_) {
    // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
    // once
    computeThreshold();
  }

  std::vector<bool> firstHit(2 * (maxlayer + 1), true);
  for (unsigned int i = 0; i < hits.size(); ++i) {
    const HGCRecHit &hgrh = hits[i];
    DetId detid = hgrh.detid();
    unsigned int layer = rhtools_.getLayerWithOffset(detid);
    float thickness = 0.f;
    // set sigmaNoise default value 1 to use kappa value directly in case of
    // sensor-independent thresholds
    float sigmaNoise = 1.f;
    if (dependSensor_) {
      thickness = rhtools_.getSiThickness(detid);
      int thickness_index = rhtools_.getSiThickIndex(detid);
      if (thickness_index == -1) thickness_index = 3;
      double storedThreshold = thresholds_[layer - 1][thickness_index];
      sigmaNoise = v_sigmaNoise_[layer - 1][thickness_index];

      if (hgrh.energy() < storedThreshold)
        continue;  // this sets the ZS threshold at ecut times the sigma noise
                   // for the sensor
    }
    if (!dependSensor_ && hgrh.energy() < ecut_) continue;

    // map layers from positive endcap (z) to layer + maxlayer+1 to prevent
    // mixing up hits from different sides
    layer += int(rhtools_.zside(detid) > 0) * (maxlayer + 1);

    // determine whether this is a half-hexagon
    bool isHalf = rhtools_.isHalfCell(detid);
    const GlobalPoint position(rhtools_.getPosition(detid));

    // here's were the KDNode is passed its dims arguments - note that these are
    // *copied* from the Hexel
    points_[layer].emplace_back(Hexel(hgrh, detid, isHalf, sigmaNoise, thickness, &rhtools_),
                                position.x(), position.y());

    // for each layer, store the minimum and maximum x and y coordinates for the
    // KDTreeBox boundaries
    if (firstHit[layer]) {
      minpos_[layer][0] = position.x();
      minpos_[layer][1] = position.y();
      maxpos_[layer][0] = position.x();
      maxpos_[layer][1] = position.y();
      firstHit[layer] = false;
    } else {
      minpos_[layer][0] = std::min((float)position.x(), minpos_[layer][0]);
      minpos_[layer][1] = std::min((float)position.y(), minpos_[layer][1]);
      maxpos_[layer][0] = std::max((float)position.x(), maxpos_[layer][0]);
      maxpos_[layer][1] = std::max((float)position.y(), maxpos_[layer][1]);
    }
  }  // end loop hits
}

// Create a vector of Hexels associated to one cluster from a collection of
// HGCalRecHits - this can be used directly to make the final cluster list -
// this method can be invoked multiple times for the same event with different
// input (reset should be called between events)
void HGCalCLUEAlgo::makeClusters() {
  layerClustersPerLayer_.resize(2 * maxlayer + 2);
  // assign all hits in each layer to a cluster core
  tbb::this_task_arena::isolate([&] {
    tbb::parallel_for(size_t(0), size_t(2 * maxlayer + 2), [&](size_t i) {
      KDTreeBox bounds(minpos_[i][0], maxpos_[i][0], minpos_[i][1], maxpos_[i][1]);
      KDTree hit_kdtree;
      hit_kdtree.build(points_[i], bounds);

      unsigned int actualLayer = i > maxlayer
                                     ? (i - (maxlayer + 1))
                                     : i;  // maps back from index used for KD trees to actual layer

      double maxdensity = calculateLocalDensity(points_[i], hit_kdtree,
                                                actualLayer);  // also stores rho (energy
                                                               // density) for each point (node)
      // calculate distance to nearest point with higher density storing
      // distance (delta) and point's index
      calculateDistanceToHigher(points_[i]);
      findAndAssignClusters(points_[i], hit_kdtree, maxdensity, bounds, actualLayer,
                            layerClustersPerLayer_[i]);
    });
  });
  //Now that we have the density per point we can store it
  for(auto const& p: points_) { setDensity(p); }
}

std::vector<reco::BasicCluster> HGCalCLUEAlgo::getClusters(bool) {
  reco::CaloID caloID = reco::CaloID::DET_HGCAL_ENDCAP;
  std::vector<std::pair<DetId, float>> thisCluster;
  for (const auto &clsOnLayer : layerClustersPerLayer_) {
    int index = 0;
    for (const auto &cl : clsOnLayer) {
      double energy = 0;
      Point position;
      // Will save the maximum density hit of the cluster
      size_t rsmax = max_index(cl);
      position = calculatePosition(cl);  // energy-weighted position
      for (const auto &it : cl) {
        energy += it.data.weight;
        thisCluster.emplace_back(it.data.detid, 1.f);
      }
      if (verbosity_ < pINFO) {
        LogDebug("HGCalCLUEAlgo")
          << "******** NEW CLUSTER (HGCIA) ********"
          << "Index          " << index
          << "No. of cells = " << cl.size()
          << "     Energy     = " << energy
          << "     Phi        = " << position.phi()
          << "     Eta        = " << position.eta()
          << "*****************************" << std::endl;
      }
      clusters_v_.emplace_back(energy, position, caloID, thisCluster, algoId_);
      if (!clusters_v_.empty()) {
        clusters_v_.back().setSeed(cl[rsmax].data.detid);
      }
      thisCluster.clear();
      index++;
    }
  }
  return clusters_v_;
}

math::XYZPoint HGCalCLUEAlgo::calculatePosition(const std::vector<KDNode> &v) const {
  float total_weight = 0.f;
  float x = 0.f;
  float y = 0.f;

  unsigned int v_size = v.size();
  unsigned int maxEnergyIndex = 0;
  float maxEnergyValue = 0;

  // loop over hits in cluster candidate
  // determining the maximum energy hit
  for (unsigned int i = 0; i < v_size; i++) {
    if (v[i].data.weight > maxEnergyValue) {
      maxEnergyValue = v[i].data.weight;
      maxEnergyIndex = i;
    }
  }

  // Si cell or Scintillator. Used to set approach and parameters
  int thick = rhtools_.getSiThickIndex(v[maxEnergyIndex].data.detid);

  // for hits within positionDeltaRho_c_ from maximum energy hit
  // build up weight for energy-weighted position
  // and save corresponding hits indices
  std::vector<unsigned int> innerIndices;
  for (unsigned int i = 0; i < v_size; i++) {
    if (thick == -1 || distance2(v[i].data, v[maxEnergyIndex].data) < positionDeltaRho_c_[thick]) {
      innerIndices.push_back(i);

      float rhEnergy = v[i].data.weight;
      total_weight += rhEnergy;
      // just fill x, y for scintillator
      // for Si it is overwritten later anyway
      if (thick == -1) {
        x += v[i].data.x * rhEnergy;
        y += v[i].data.y * rhEnergy;
      }
    }
  }
  // just loop on reduced vector of interesting indices
  // to compute log weighting
  if (thick != -1 && total_weight != 0.) {  // Silicon case
    float total_weight_log = 0.f;
    float x_log = 0.f;
    float y_log = 0.f;
    for (auto idx : innerIndices) {
      float rhEnergy = v[idx].data.weight;
      if (rhEnergy == 0.) continue;
      float Wi = std::max(thresholdW0_[thick] + std::log(rhEnergy / total_weight), 0.);
      x_log += v[idx].data.x * Wi;
      y_log += v[idx].data.y * Wi;
      total_weight_log += Wi;
    }
    total_weight = total_weight_log;
    x = x_log;
    y = y_log;
  }

  if (total_weight != 0.) {
    auto inv_tot_weight = 1. / total_weight;
    return math::XYZPoint(x * inv_tot_weight, y * inv_tot_weight, v[maxEnergyIndex].data.z);
  }
  return math::XYZPoint(0, 0, 0);
}

double HGCalCLUEAlgo::calculateLocalDensity(std::vector<KDNode> &nd, KDTree &lp,
                                            const unsigned int layer) const {
  double maxdensity = 0.;
  float delta_c;  // maximum search distance (critical distance) for local
                  // density calculation
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];

  // for each node calculate local density rho and store it
  for (unsigned int i = 0; i < nd.size(); ++i) {
    // speec up search by looking within +/- delta_c window only
    KDTreeBox search_box(nd[i].dims[0] - delta_c, nd[i].dims[0] + delta_c, nd[i].dims[1] - delta_c,
                         nd[i].dims[1] + delta_c);
    std::vector<KDNode> found;
    lp.search(search_box, found);
    const unsigned int found_size = found.size();
    for (unsigned int j = 0; j < found_size; j++) {
      if (distance(nd[i].data, found[j].data) < delta_c) {
        nd[i].data.rho += (nd[i].data.detid == found[j].data.detid ? 1. : 0.5) * found[j].data.weight;
        maxdensity = std::max(maxdensity, nd[i].data.rho);
      }
    }  // end loop found
  }    // end loop nodes
  return maxdensity;
}

double HGCalCLUEAlgo::calculateDistanceToHigher(std::vector<KDNode> &nd) const {
  // sort vector of Hexels by decreasing local density
  std::vector<size_t> &&rs = sorted_indices(nd);

  double maxdensity = 0.0;
  int nearestHigher = -1;

  if (!rs.empty())
    maxdensity = nd[rs[0]].data.rho;
  else
    return maxdensity;  // there are no hits
  double dist2 = 0.;
  // start by setting delta for the highest density hit to
  // the most distant hit - this is a convention

  for (const auto &j : nd) {
    double tmp = distance2(nd[rs[0]].data, j.data);
    if (tmp > dist2) dist2 = tmp;
  }
  nd[rs[0]].data.delta = std::sqrt(dist2);
  nd[rs[0]].data.nearestHigher = nearestHigher;

  // now we save the largest distance as a starting point
  const double max_dist2 = dist2;
  const unsigned int nd_size = nd.size();

  for (unsigned int oi = 1; oi < nd_size; ++oi) {  // start from second-highest density
    dist2 = max_dist2;
    unsigned int i = rs[oi];
    // we only need to check up to oi since hits
    // are ordered by decreasing density
    // and all points coming BEFORE oi are guaranteed to have higher rho
    // and the ones AFTER to have lower rho
    for (unsigned int oj = 0; oj < oi; ++oj) {
      unsigned int j = rs[oj];
      double tmp = distance2(nd[i].data, nd[j].data);
      if (tmp <= dist2) {  // this "<=" instead of "<" addresses the (rare) case
                           // when there are only two hits
        dist2 = tmp;
        nearestHigher = j;
      }
    }
    nd[i].data.delta = std::sqrt(dist2);
    nd[i].data.nearestHigher = nearestHigher;  // this uses the original unsorted hitlist
  }
  return maxdensity;
}
int HGCalCLUEAlgo::findAndAssignClusters(std::vector<KDNode> &nd, KDTree &lp, double maxdensity,
                                         KDTreeBox &bounds, const unsigned int layer,
                                         std::vector<std::vector<KDNode>> &clustersOnLayer) const {
  // this is called once per layer and endcap...
  // so when filling the cluster temporary vector of Hexels we resize each time
  // by the number  of clusters found. This is always equal to the number of
  // cluster centers...

  unsigned int nClustersOnLayer = 0;
  float delta_c;  // critical distance
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];

  std::vector<size_t> rs = sorted_indices(nd);  // indices sorted by decreasing rho
  std::vector<size_t> ds = sort_by_delta(nd);   // sort in decreasing distance to higher

  const unsigned int nd_size = nd.size();
  for (unsigned int i = 0; i < nd_size; ++i) {
    if (nd[ds[i]].data.delta < delta_c) break;  // no more cluster centers to be looked at
    if (dependSensor_) {
      float rho_c = kappa_ * nd[ds[i]].data.sigmaNoise;
      if (nd[ds[i]].data.rho < rho_c) continue;  // set equal to kappa times noise threshold

    } else if (nd[ds[i]].data.rho * kappa_ < maxdensity)
      continue;

    nd[ds[i]].data.clusterIndex = nClustersOnLayer;
    if (verbosity_ < pINFO) {
      LogDebug("HGCalCLUEAlgo")
        << "Adding new cluster with index " << nClustersOnLayer
        << "Cluster center is hit " << ds[i] << std::endl;
    }
    nClustersOnLayer++;
  }

  // at this point nClustersOnLayer is equal to the number of cluster centers -
  // if it is zero we are  done
  if (nClustersOnLayer == 0) return nClustersOnLayer;

  // assign remaining points to clusters, using the nearestHigher set from
  // previous step (always set except
  // for top density hit that is skipped...)
  for (unsigned int oi = 1; oi < nd_size; ++oi) {
    unsigned int i = rs[oi];
    int ci = nd[i].data.clusterIndex;
    if (ci == -1 && nd[i].data.delta < 2. * delta_c) {
      nd[i].data.clusterIndex = nd[nd[i].data.nearestHigher].data.clusterIndex;
    }
  }

  // make room in the temporary cluster vector for the additional clusterIndex
  // clusters
  // from this layer
  if (verbosity_ < pINFO) {
    LogDebug("HGCalCLUEAlgo")
      << "resizing cluster vector by " << nClustersOnLayer << std::endl;
  }
  clustersOnLayer.resize(nClustersOnLayer);

  // Fill the cluster vector
  for (unsigned int i = 0; i < nd_size; ++i) {
    int ci = nd[i].data.clusterIndex;
    if (ci != -1) {
      clustersOnLayer[ci].push_back(nd[i]);
      if (verbosity_ < pINFO) {
        LogDebug("HGCalCLUEAlgo")
          << "Pushing hit " << i << " into cluster with index " << ci << std::endl;
      }
    }
  }

  // prepare the offset for the next layer if there is one
  if (verbosity_ < pINFO) {
    LogDebug("HGCalCLUEAlgo") << "moving cluster offset by " << nClustersOnLayer << std::endl;
  }
  return nClustersOnLayer;
}

void HGCalCLUEAlgo::computeThreshold() {
  // To support the TDR geometry and also the post-TDR one (v9 onwards), we
  // need to change the logic of the vectors containing signal to noise and
  // thresholds. The first 3 indices will keep on addressing the different
  // thicknesses of the Silicon detectors, while the last one, number 3 (the
  // fourth) will address the Scintillators. This change will support both
  // geometries at the same time.

  if (initialized_) return;  // only need to calculate thresholds once

  initialized_ = true;

  std::vector<double> dummy;
  const unsigned maxNumberOfThickIndices = 3;
  dummy.resize(maxNumberOfThickIndices + 1, 0);  // +1 to accomodate for the Scintillators
  thresholds_.resize(maxlayer, dummy);
  v_sigmaNoise_.resize(maxlayer, dummy);

  for (unsigned ilayer = 1; ilayer <= maxlayer; ++ilayer) {
    for (unsigned ithick = 0; ithick < maxNumberOfThickIndices; ++ithick) {
      float sigmaNoise = 0.001f * fcPerEle_ * nonAgedNoises_[ithick] * dEdXweights_[ilayer] /
                         (fcPerMip_[ithick] * thicknessCorrection_[ithick]);
      thresholds_[ilayer - 1][ithick] = sigmaNoise * ecut_;
      v_sigmaNoise_[ilayer - 1][ithick] = sigmaNoise;
    }
    float scintillators_sigmaNoise = 0.001f * noiseMip_ * dEdXweights_[ilayer];
    thresholds_[ilayer - 1][maxNumberOfThickIndices] = ecut_ * scintillators_sigmaNoise;
    v_sigmaNoise_[ilayer - 1][maxNumberOfThickIndices] = scintillators_sigmaNoise;
  }
}

void HGCalCLUEAlgo::setDensity(const std::vector<KDNode> &nd){

  // for each node store the computer local density
  for (auto &i : nd){
    density_[ i.data.detid ] =  i.data.rho ;
  }
}

Density HGCalCLUEAlgo::getDensity() {
  return density_;
}
