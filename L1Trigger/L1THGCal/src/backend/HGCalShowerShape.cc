#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <unordered_map>
#include <numeric>

HGCalShowerShape::HGCalShowerShape(const edm::ParameterSet& conf)
    : threshold_(conf.getParameter<double>("shape_threshold")),
      distance_(conf.getParameter<double>("shape_distance")) {}

//Compute energy-weighted mean of any variable X in the cluster

float HGCalShowerShape::meanX(const std::vector<pair<float, float>>& energy_X_tc) const {
  float Etot = 0;
  float X_sum = 0;

  for (const auto& energy_X : energy_X_tc) {
    X_sum += energy_X.first * energy_X.second;
    Etot += energy_X.first;
  }

  float X_mean = 0;
  if (Etot > 0)
    X_mean = X_sum / Etot;
  return X_mean;
}

int HGCalShowerShape::firstLayer(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  int firstLayer = 999;

  for (const auto& id_clu : clustersPtrs) {
    if (!pass(*id_clu.second, c3d))
      continue;
    int layer = triggerTools_.layerWithOffset(id_clu.second->detId());
    if (layer < firstLayer)
      firstLayer = layer;
  }

  return firstLayer;
}

int HGCalShowerShape::maxLayer(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();
  std::unordered_map<int, float> layers_pt;
  float max_pt = 0.;
  int max_layer = 0;
  for (const auto& id_cluster : clustersPtrs) {
    if (!pass(*id_cluster.second, c3d))
      continue;
    unsigned layer = triggerTools_.layerWithOffset(id_cluster.second->detId());
    auto itr_insert = layers_pt.emplace(layer, 0.);
    itr_insert.first->second += id_cluster.second->pt();
    if (itr_insert.first->second > max_pt) {
      max_pt = itr_insert.first->second;
      max_layer = layer;
    }
  }
  return max_layer;
}

int HGCalShowerShape::lastLayer(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  int lastLayer = -999;

  for (const auto& id_clu : clustersPtrs) {
    if (!pass(*id_clu.second, c3d))
      continue;
    int layer = triggerTools_.layerWithOffset(id_clu.second->detId());
    if (layer > lastLayer)
      lastLayer = layer;
  }

  return lastLayer;
}

int HGCalShowerShape::coreShowerLength(const l1t::HGCalMulticluster& c3d,
                                       const HGCalTriggerGeometryBase& triggerGeometry) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();
  unsigned nlayers = triggerTools_.layers(ForwardSubdetector::ForwardEmpty);
  std::vector<bool> layers(nlayers);
  for (const auto& id_cluster : clustersPtrs) {
    if (!pass(*id_cluster.second, c3d))
      continue;
    unsigned layer = triggerGeometry.triggerLayer(id_cluster.second->detId());
    if (triggerTools_.isNose(id_cluster.second->detId()))
      nlayers = triggerTools_.layers(ForwardSubdetector::HFNose);
    else {
      nlayers = triggerTools_.layers(ForwardSubdetector::ForwardEmpty);
    }
    if (layer == 0 || layer > nlayers)
      continue;
    layers[layer - 1] = true;  //layer 0 doesn't exist, so shift by -1
  }
  int length = 0;
  int maxlength = 0;
  for (bool layer : layers) {
    if (layer)
      length++;
    else
      length = 0;
    if (length > maxlength)
      maxlength = length;
  }
  return maxlength;
}

float HGCalShowerShape::percentileLayer(const l1t::HGCalMulticluster& c3d,
                                        const HGCalTriggerGeometryBase& triggerGeometry,
                                        float quantile) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();
  unsigned nlayers = triggerTools_.layers(ForwardSubdetector::ForwardEmpty);
  std::vector<double> layers(nlayers, 0);
  for (const auto& id_clu : clustersPtrs) {
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      unsigned layer = triggerGeometry.triggerLayer(id_tc.second->detId());
      if (triggerTools_.isNose(id_tc.second->detId()))
        nlayers = triggerTools_.layers(ForwardSubdetector::HFNose);
      else {
        nlayers = triggerTools_.layers(ForwardSubdetector::ForwardEmpty);
      }
      if (layer == 0 || layer > nlayers)
        continue;
      layers[layer - 1] += id_tc.second->pt();  //layer 0 doesn't exist, so shift by -1
    }
  }
  std::partial_sum(layers.begin(), layers.end(), layers.begin());
  double pt_threshold = layers.back() * quantile;
  unsigned percentile = 0;
  for (double pt : layers) {
    if (pt > pt_threshold) {
      break;
    }
    percentile++;
  }
  // Linear interpolation of percentile value
  double pt0 = (percentile > 0 ? layers[percentile - 1] : 0.);
  double pt1 = (percentile < layers.size() ? layers[percentile] : layers.back());
  return percentile + (pt1 - pt0 > 0. ? (pt_threshold - pt0) / (pt1 - pt0) : 0.);
}

float HGCalShowerShape::percentileTriggerCells(const l1t::HGCalMulticluster& c3d, float quantile) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();
  std::set<double> ordered_tcs;
  double pt_sum = 0.;
  for (const auto& id_clu : clustersPtrs) {
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      ordered_tcs.emplace(id_tc.second->pt());
      pt_sum += id_tc.second->pt();
    }
  }
  double pt_threshold = pt_sum * quantile;
  double partial_sum = 0.;
  double partial_sum_prev = 0.;
  int ntc = 0;
  for (auto itr = ordered_tcs.rbegin(); itr != ordered_tcs.rend(); ++itr) {
    partial_sum_prev = partial_sum;
    partial_sum += *itr;
    ntc++;
    if (partial_sum > pt_threshold) {
      break;
    }
  }
  // Linear interpolation of ntc
  return ntc - 1 +
         (partial_sum - partial_sum_prev > 0. ? (pt_threshold - partial_sum_prev) / (partial_sum - partial_sum_prev)
                                              : 0.);
}

float HGCalShowerShape::sigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  std::vector<std::pair<float, float>> tc_energy_eta;

  for (const auto& id_clu : clustersPtrs) {
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      tc_energy_eta.emplace_back(std::make_pair(id_tc.second->energy(), id_tc.second->eta()));
    }
  }

  float SeeTot = sigmaXX(tc_energy_eta, c3d.eta());

  return SeeTot;
}

float HGCalShowerShape::sigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  std::vector<std::pair<float, float>> tc_energy_phi;

  for (const auto& id_clu : clustersPtrs) {
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      tc_energy_phi.emplace_back(std::make_pair(id_tc.second->energy(), id_tc.second->phi()));
    }
  }

  float SppTot = sigmaPhiPhi(tc_energy_phi, c3d.phi());

  return SppTot;
}

float HGCalShowerShape::sigmaRRTot(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  std::vector<std::pair<float, float>> tc_energy_r;

  for (const auto& id_clu : clustersPtrs) {
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      float r = (id_tc.second->position().z() != 0.
                     ? std::sqrt(pow(id_tc.second->position().x(), 2) + pow(id_tc.second->position().y(), 2)) /
                           std::abs(id_tc.second->position().z())
                     : 0.);
      tc_energy_r.emplace_back(std::make_pair(id_tc.second->energy(), r));
    }
  }

  float r_mean = meanX(tc_energy_r);
  float Szz = sigmaXX(tc_energy_r, r_mean);

  return Szz;
}

float HGCalShowerShape::sigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d) const {
  std::unordered_map<int, std::vector<std::pair<float, float>>> tc_layer_energy_eta;
  std::unordered_map<int, LorentzVector> layer_LV;

  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  for (const auto& id_clu : clustersPtrs) {
    unsigned layer = triggerTools_.layerWithOffset(id_clu.second->detId());

    layer_LV[layer] += id_clu.second->p4();

    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      tc_layer_energy_eta[layer].emplace_back(std::make_pair(id_tc.second->energy(), id_tc.second->eta()));
    }
  }

  float SigmaEtaEtaMax = 0;

  for (auto& tc_iter : tc_layer_energy_eta) {
    const std::vector<std::pair<float, float>>& energy_eta_layer = tc_iter.second;
    const LorentzVector& LV_layer = layer_LV[tc_iter.first];
    float SigmaEtaEtaLayer = sigmaXX(energy_eta_layer, LV_layer.eta());  //RMS wrt layer eta, not wrt c3d eta
    if (SigmaEtaEtaLayer > SigmaEtaEtaMax)
      SigmaEtaEtaMax = SigmaEtaEtaLayer;
  }

  return SigmaEtaEtaMax;
}

float HGCalShowerShape::sigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d) const {
  std::unordered_map<int, std::vector<std::pair<float, float>>> tc_layer_energy_phi;
  std::unordered_map<int, LorentzVector> layer_LV;

  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  for (const auto& id_clu : clustersPtrs) {
    unsigned layer = triggerTools_.layerWithOffset(id_clu.second->detId());

    layer_LV[layer] += id_clu.second->p4();

    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      tc_layer_energy_phi[layer].emplace_back(std::make_pair(id_tc.second->energy(), id_tc.second->phi()));
    }
  }

  float SigmaPhiPhiMax = 0;

  for (auto& tc_iter : tc_layer_energy_phi) {
    const std::vector<std::pair<float, float>>& energy_phi_layer = tc_iter.second;
    const LorentzVector& LV_layer = layer_LV[tc_iter.first];
    float SigmaPhiPhiLayer = sigmaPhiPhi(energy_phi_layer, LV_layer.phi());  //RMS wrt layer phi, not wrt c3d phi
    if (SigmaPhiPhiLayer > SigmaPhiPhiMax)
      SigmaPhiPhiMax = SigmaPhiPhiLayer;
  }

  return SigmaPhiPhiMax;
}

float HGCalShowerShape::sigmaRRMax(const l1t::HGCalMulticluster& c3d) const {
  std::unordered_map<int, std::vector<std::pair<float, float>>> tc_layer_energy_r;

  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  for (const auto& id_clu : clustersPtrs) {
    unsigned layer = triggerTools_.layerWithOffset(id_clu.second->detId());

    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      float r = (id_tc.second->position().z() != 0.
                     ? std::sqrt(pow(id_tc.second->position().x(), 2) + pow(id_tc.second->position().y(), 2)) /
                           std::abs(id_tc.second->position().z())
                     : 0.);
      tc_layer_energy_r[layer].emplace_back(std::make_pair(id_tc.second->energy(), r));
    }
  }

  float SigmaRRMax = 0;

  for (auto& tc_iter : tc_layer_energy_r) {
    const std::vector<std::pair<float, float>>& energy_r_layer = tc_iter.second;
    float r_mean_layer = meanX(energy_r_layer);
    float SigmaRRLayer = sigmaXX(energy_r_layer, r_mean_layer);
    if (SigmaRRLayer > SigmaRRMax)
      SigmaRRMax = SigmaRRLayer;
  }

  return SigmaRRMax;
}

float HGCalShowerShape::sigmaRRMean(const l1t::HGCalMulticluster& c3d, float radius) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();
  // group trigger cells by layer
  std::unordered_map<int, std::vector<edm::Ptr<l1t::HGCalTriggerCell>>> layers_tcs;
  for (const auto& id_clu : clustersPtrs) {
    unsigned layer = triggerTools_.layerWithOffset(id_clu.second->detId());
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();
    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      layers_tcs[layer].emplace_back(id_tc.second);
    }
  }

  // Select trigger cells within X cm of the max TC in the layer
  std::unordered_map<int, std::vector<std::pair<float, float>>> tc_layers_energy_r;
  for (const auto& layer_tcs : layers_tcs) {
    int layer = layer_tcs.first;
    edm::Ptr<l1t::HGCalTriggerCell> max_tc = layer_tcs.second.front();
    for (const auto& tc : layer_tcs.second) {
      if (tc->energy() > max_tc->energy())
        max_tc = tc;
    }
    for (const auto& tc : layer_tcs.second) {
      double dx = tc->position().x() - max_tc->position().x();
      double dy = tc->position().y() - max_tc->position().y();
      double distance_to_max = std::sqrt(dx * dx + dy * dy);
      if (distance_to_max < radius) {
        float r = (tc->position().z() != 0.
                       ? std::sqrt(tc->position().x() * tc->position().x() + tc->position().y() * tc->position().y()) /
                             std::abs(tc->position().z())
                       : 0.);
        tc_layers_energy_r[layer].emplace_back(std::make_pair(tc->energy(), r));
      }
    }
  }

  // Compute srr layer by layer
  std::vector<std::pair<float, float>> layers_energy_srr2;
  for (const auto& layer_energy_r : tc_layers_energy_r) {
    const auto& energies_r = layer_energy_r.second;
    float r_mean_layer = meanX(energies_r);
    float srr = sigmaXX(energies_r, r_mean_layer);
    double energy_sum = 0.;
    for (const auto& energy_r : energies_r) {
      energy_sum += energy_r.first;
    }
    layers_energy_srr2.emplace_back(std::make_pair(energy_sum, srr * srr));
  }
  // Combine all layer srr
  float srr2_mean = meanX(layers_energy_srr2);
  return std::sqrt(srr2_mean);
}

float HGCalShowerShape::eMax(const l1t::HGCalMulticluster& c3d) const {
  std::unordered_map<int, float> layer_energy;

  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  for (const auto& id_clu : clustersPtrs) {
    if (!pass(*id_clu.second, c3d))
      continue;
    unsigned layer = triggerTools_.layerWithOffset(id_clu.second->detId());
    layer_energy[layer] += id_clu.second->energy();
  }

  float EMax = 0;

  for (const auto& layer : layer_energy) {
    if (layer.second > EMax)
      EMax = layer.second;
  }

  return EMax;
}

float HGCalShowerShape::meanZ(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  std::vector<std::pair<float, float>> tc_energy_z;

  for (const auto& id_clu : clustersPtrs) {
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      tc_energy_z.emplace_back(id_tc.second->energy(), id_tc.second->position().z());
    }
  }

  return meanX(tc_energy_z);
}

float HGCalShowerShape::sigmaZZ(const l1t::HGCalMulticluster& c3d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clustersPtrs = c3d.constituents();

  std::vector<std::pair<float, float>> tc_energy_z;

  for (const auto& id_clu : clustersPtrs) {
    const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& triggerCells = id_clu.second->constituents();

    for (const auto& id_tc : triggerCells) {
      if (!pass(*id_tc.second, c3d))
        continue;
      tc_energy_z.emplace_back(std::make_pair(id_tc.second->energy(), id_tc.second->position().z()));
    }
  }

  float z_mean = meanX(tc_energy_z);
  float Szz = sigmaXX(tc_energy_z, z_mean);

  return Szz;
}

float HGCalShowerShape::sigmaEtaEtaTot(const l1t::HGCalCluster& c2d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& cellsPtrs = c2d.constituents();

  std::vector<std::pair<float, float>> tc_energy_eta;

  for (const auto& id_cell : cellsPtrs) {
    if (!pass(*id_cell.second, c2d))
      continue;
    tc_energy_eta.emplace_back(std::make_pair(id_cell.second->energy(), id_cell.second->eta()));
  }

  float See = sigmaXX(tc_energy_eta, c2d.eta());

  return See;
}

float HGCalShowerShape::sigmaPhiPhiTot(const l1t::HGCalCluster& c2d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& cellsPtrs = c2d.constituents();

  std::vector<std::pair<float, float>> tc_energy_phi;

  for (const auto& id_cell : cellsPtrs) {
    if (!pass(*id_cell.second, c2d))
      continue;
    tc_energy_phi.emplace_back(std::make_pair(id_cell.second->energy(), id_cell.second->phi()));
  }

  float Spp = sigmaPhiPhi(tc_energy_phi, c2d.phi());

  return Spp;
}

float HGCalShowerShape::sigmaRRTot(const l1t::HGCalCluster& c2d) const {
  const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& cellsPtrs = c2d.constituents();

  std::vector<std::pair<float, float>> tc_energy_r;

  for (const auto& id_cell : cellsPtrs) {
    if (!pass(*id_cell.second, c2d))
      continue;
    float r = (id_cell.second->position().z() != 0.
                   ? std::sqrt(pow(id_cell.second->position().x(), 2) + pow(id_cell.second->position().y(), 2)) /
                         std::abs(id_cell.second->position().z())
                   : 0.);
    tc_energy_r.emplace_back(std::make_pair(id_cell.second->energy(), r));
  }

  float r_mean = meanX(tc_energy_r);
  float Srr = sigmaXX(tc_energy_r, r_mean);

  return Srr;
}

void HGCalShowerShape::fillShapes(l1t::HGCalMulticluster& c3d, const HGCalTriggerGeometryBase& triggerGeometry) const {
  c3d.showerLength(showerLength(c3d));
  c3d.coreShowerLength(coreShowerLength(c3d, triggerGeometry));
  c3d.firstLayer(firstLayer(c3d));
  c3d.maxLayer(maxLayer(c3d));
  c3d.sigmaEtaEtaTot(sigmaEtaEtaTot(c3d));
  c3d.sigmaEtaEtaMax(sigmaEtaEtaMax(c3d));
  c3d.sigmaPhiPhiTot(sigmaPhiPhiTot(c3d));
  c3d.sigmaPhiPhiMax(sigmaPhiPhiMax(c3d));
  c3d.sigmaZZ(sigmaZZ(c3d));
  c3d.sigmaRRTot(sigmaRRTot(c3d));
  c3d.sigmaRRMax(sigmaRRMax(c3d));
  c3d.sigmaRRMean(sigmaRRMean(c3d));
  c3d.eMax(eMax(c3d));
  c3d.zBarycenter(meanZ(c3d));
  c3d.layer10percent(percentileLayer(c3d, triggerGeometry, 0.10));
  c3d.layer50percent(percentileLayer(c3d, triggerGeometry, 0.50));
  c3d.layer90percent(percentileLayer(c3d, triggerGeometry, 0.90));
  c3d.triggerCells67percent(percentileTriggerCells(c3d, 0.67));
  c3d.triggerCells90percent(percentileTriggerCells(c3d, 0.90));
}
