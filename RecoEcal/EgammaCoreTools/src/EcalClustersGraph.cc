#include "RecoEcal/EgammaCoreTools/interface/EcalClustersGraph.h"
#include <cmath>

using namespace std;
using namespace reco;
using namespace reco::DeepSCInputs;

typedef std::vector<CalibratedPFCluster> CalibratedPFClusterVector;

EcalClustersGraph::EcalClustersGraph(CalibratedPFClusterVector clusters,
                                     int nSeeds,
                                     const CaloTopology* topology,
                                     const CaloSubdetectorGeometry* ebGeom,
                                     const CaloSubdetectorGeometry* eeGeom,
                                     const EcalRecHitCollection* recHitsEB,
                                     const EcalRecHitCollection* recHitsEE,
                                     const SCProducerCache* cache)
    : clusters_(clusters),
      nSeeds_(nSeeds),
      nCls_(clusters_.size()),
      topology_(topology),
      ebGeom_(ebGeom),
      eeGeom_(eeGeom),
      recHitsEB_(recHitsEB),
      recHitsEE_(recHitsEE),
      scProducerCache_(cache),
      graphMap_(clusters.size()) {
  // Prepare the batch size of the tensor inputs == number of windows
  inputs_.clustersX.resize(nSeeds_);
  inputs_.windowX.resize(nSeeds_);
  inputs_.hitsX.resize(nSeeds_);
  inputs_.isSeed.resize(nSeeds_);

  // Init the graph nodes
  for (size_t i = 0; i < nCls_; i++) {
    if (i < nSeeds_)
      graphMap_.addNode(i, GraphMap::NodeCategory::kSeed);
    else
      graphMap_.addNode(i, GraphMap::NodeCategory::kNode);
  }

  // Select the collection strategy from the config
  if (scProducerCache_->config.collectionStrategy == "Cascade") {
    strategy_ = GraphMap::CollectionStrategy::Cascade;
  } else if (scProducerCache_->config.collectionStrategy == "CollectAndMerge") {
    strategy_ = GraphMap::CollectionStrategy::CollectAndMerge;
  } else if (scProducerCache_->config.collectionStrategy == "SeedsFirst") {
    strategy_ = GraphMap::CollectionStrategy::SeedsFirst;
  } else if (scProducerCache_->config.collectionStrategy == "CascadeHighest") {
    strategy_ = GraphMap::CollectionStrategy::CascadeHighest;
  } else {
    edm::LogWarning("EcalClustersGraph") << "GraphMap::CollectionStrategy not recognized. Default to Cascade";
    strategy_ = GraphMap::CollectionStrategy::Cascade;
  }

  LogTrace("EcalClustersGraph") << "EcalClustersGraph created. nSeeds " << nSeeds_ << ", nClusters " << nCls_ << endl;
}

std::array<int, 3> EcalClustersGraph::clusterPosition(const CaloCluster* cluster) const {
  std::array<int, 3> coordinates;
  int ieta = -999;
  int iphi = -999;
  int iz = -99;

  const math::XYZPoint& caloPos = cluster->position();
  if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL) {
    EBDetId eb_id(ebGeom_->getClosestCell(GlobalPoint(caloPos.x(), caloPos.y(), caloPos.z())));
    ieta = eb_id.ieta();
    iphi = eb_id.iphi();
    iz = 0;
  } else if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP) {
    EEDetId ee_id(eeGeom_->getClosestCell(GlobalPoint(caloPos.x(), caloPos.y(), caloPos.z())));
    ieta = ee_id.ix();
    iphi = ee_id.iy();
    if (ee_id.zside() < 0)
      iz = -1;
    if (ee_id.zside() > 0)
      iz = 1;
  }

  coordinates[0] = ieta;
  coordinates[1] = iphi;
  coordinates[2] = iz;
  return coordinates;
}

std::array<double, 3> EcalClustersGraph::dynamicWindow(double seedEta) const {
  // The dEta-dPhi detector window dimension is chosen to that the algorithm is always larger than
  // the Mustache dimension
  std::array<double, 3> window;

  double eta = std::abs(seedEta);
  double deta_down = 0.;
  double deta_up = 0.;
  double dphi = 0.;

  //deta_down
  constexpr float deta_down_bins[2] = {2.1, 2.5};
  if (eta < deta_down_bins[0])
    deta_down = -0.075;
  else if (eta >= deta_down_bins[0] && eta < deta_down_bins[1])
    deta_down = -0.1875 * eta + 0.31875;
  else if (eta >= deta_down_bins[1])
    deta_down = -0.15;

  //deta_up
  constexpr float deta_up_bins[4] = {0.1, 1.3, 1.7, 1.9};
  if (eta < deta_up_bins[0])
    deta_up = 0.075;
  else if (eta >= deta_up_bins[0] && eta < deta_up_bins[1])
    deta_up = 0.0758929 - 0.0178571 * eta + 0.0892857 * (eta * eta);
  else if (eta >= deta_up_bins[1] && eta < deta_up_bins[2])
    deta_up = 0.2;
  else if (eta >= deta_up_bins[2] && eta < deta_up_bins[3])
    deta_up = 0.625 - 0.25 * eta;
  else if (eta >= deta_up_bins[3])
    deta_up = 0.15;

  //dphi
  constexpr float dphi_bins[2] = {1.9, 2.7};
  if (eta < dphi_bins[0])
    dphi = 0.6;
  else if (eta >= dphi_bins[0] && eta < dphi_bins[1])
    dphi = 1.075 - 0.25 * eta;
  else if (eta >= dphi_bins[1])
    dphi = 0.4;

  window[0] = deta_down;
  window[1] = deta_up;
  window[2] = dphi;

  return window;
}

void EcalClustersGraph::initWindows() {
  for (uint is = 0; is < nSeeds_; is++) {
    const auto& seedLocal = clusterPosition((clusters_[is]).ptr().get());
    double seed_eta = clusters_[is].eta();
    double seed_phi = clusters_[is].phi();
    const auto& width = dynamicWindow(seed_eta);
    // Add a self loop on the seed node
    graphMap_.addEdge(is, is);

    // The graph associated to each seed includes only other seeds if they have a smaller energy.
    // This is imposed to be consistent with the current trained model, which has been training on "non-overalapping windows".
    // The effect of connecting all the seeds, and not only the smaller energy ones has been already tested: the reconstruction
    // differences are negligible (tested with Cascade collection algo).
    // In the next version of the training this requirement will be relaxed to have a model that fully matches the reconstruction
    // mechanism in terms of overlapping seeds.
    for (uint icl = is + 1; icl < nCls_; icl++) {
      if (is == icl)
        continue;
      const auto& clusterLocal = clusterPosition((clusters_[icl]).ptr().get());
      double cl_eta = clusters_[icl].eta();
      double cl_phi = clusters_[icl].phi();
      double dphi = deltaPhi(seed_phi, cl_phi);
      double deta = deltaEta(seed_eta, cl_eta);

      if (seedLocal[2] == clusterLocal[2] && deta >= width[0] && deta <= width[1] && std::abs(dphi) <= width[2]) {
        graphMap_.addEdge(is, icl);
      }
    }
  }
}

std::vector<std::vector<float>> EcalClustersGraph::fillHits(const CaloCluster* cluster) const {
  const std::vector<std::pair<DetId, float>>& hitsAndFractions = cluster->hitsAndFractions();
  std::vector<std::vector<float>> out(hitsAndFractions.size());
  if (hitsAndFractions.empty()) {
    edm::LogError("EcalClustersGraph") << "No hits in cluster!!";
  }
  // Map containing the available features for the rechits
  DeepSCInputs::FeaturesMap rechitsFeatures;
  for (unsigned int i = 0; i < hitsAndFractions.size(); i++) {
    rechitsFeatures.clear();
    if (hitsAndFractions[i].first.subdetId() == EcalBarrel) {
      double energy = (*recHitsEB_->find(hitsAndFractions[i].first)).energy();
      EBDetId eb_id(hitsAndFractions[i].first);
      rechitsFeatures["ieta"] = eb_id.ieta();                                //ieta
      rechitsFeatures["iphi"] = eb_id.iphi();                                //iphi
      rechitsFeatures["iz"] = 0.;                                            //iz
      rechitsFeatures["en_withfrac"] = energy * hitsAndFractions[i].second;  //energy * fraction
    } else if (hitsAndFractions[i].first.subdetId() == EcalEndcap) {
      double energy = (*recHitsEE_->find(hitsAndFractions[i].first)).energy();
      EEDetId ee_id(hitsAndFractions[i].first);
      rechitsFeatures["ieta"] = ee_id.ix();  //ix
      rechitsFeatures["iphi"] = ee_id.iy();  //iy
      if (ee_id.zside() < 0)
        rechitsFeatures["iz"] = -1.;  //iz
      if (ee_id.zside() > 0)
        rechitsFeatures["iz"] = +1.;                                         //iz
      rechitsFeatures["en_withfrac"] = energy * hitsAndFractions[i].second;  //energy * fraction
    } else {
      edm::LogError("EcalClustersGraph") << "Rechit is not either EB or EE!!";
    }
    // Use the method in DeepSCGraphEvaluation to get only the requested variables and possible a rescaling
    // (depends on configuration)
    out[i] = scProducerCache_->deepSCEvaluator->getScaledInputs(rechitsFeatures,
                                                                scProducerCache_->deepSCEvaluator->inputFeaturesHits);
  }
  return out;
}

DeepSCInputs::FeaturesMap EcalClustersGraph::computeVariables(const CaloCluster* seed,
                                                              const CaloCluster* cluster) const {
  DeepSCInputs::FeaturesMap clFeatures;
  const auto& clusterLocal = clusterPosition(cluster);
  double cl_energy = cluster->energy();
  double cl_eta = cluster->eta();
  double cl_phi = cluster->phi();
  double seed_energy = seed->energy();
  double seed_eta = seed->eta();
  double seed_phi = seed->phi();
  clFeatures["cl_energy"] = cl_energy;                                                                //cl_energy
  clFeatures["cl_et"] = cl_energy / std::cosh(cl_eta);                                                //cl_et
  clFeatures["cl_eta"] = cl_eta;                                                                      //cl_eta
  clFeatures["cl_phi"] = cl_phi;                                                                      //cl_phi
  clFeatures["cl_ieta"] = clusterLocal[0];                                                            //cl_ieta/ix
  clFeatures["cl_iphi"] = clusterLocal[1];                                                            //cl_iphi/iy
  clFeatures["cl_iz"] = clusterLocal[2];                                                              //cl_iz
  clFeatures["cl_seed_dEta"] = deltaEta(seed_eta, cl_eta);                                            //cl_dEta
  clFeatures["cl_seed_dPhi"] = deltaPhi(seed_phi, cl_phi);                                            //cl_dPhi
  clFeatures["cl_seed_dEnergy"] = seed_energy - cl_energy;                                            //cl_dEnergy
  clFeatures["cl_seed_dEt"] = (seed_energy / std::cosh(seed_eta)) - (cl_energy / std::cosh(cl_eta));  //cl_dEt
  clFeatures["cl_nxtals"] = cluster->hitsAndFractions().size();                                       // nxtals
  return clFeatures;
}

DeepSCInputs::FeaturesMap EcalClustersGraph::computeWindowVariables(
    const std::vector<DeepSCInputs::FeaturesMap>& clusters) const {
  size_t nCls = clusters.size();
  std::map<std::string, float> min;
  std::map<std::string, float> max;
  std::map<std::string, float> avg;
  for (const auto& clFeatures : clusters) {
    for (auto const& [key, val] : clFeatures) {
      avg[key] += (val / nCls);
      if (val < min[key])
        min[key] = val;
      if (val > max[key])
        max[key] = val;
    }
  }
  DeepSCInputs::FeaturesMap windFeatures;
  for (auto const& el : clusters.front()) {
    windFeatures["max_" + el.first] = max[el.first];
    windFeatures["min_" + el.first] = min[el.first];
    windFeatures["avg_" + el.first] = avg[el.first];
  }
  return windFeatures;
}

std::pair<double, double> EcalClustersGraph::computeCovariances(const CaloCluster* cluster) {
  double numeratorEtaWidth = 0;
  double numeratorPhiWidth = 0;
  double denominator = cluster->energy();
  double clEta = cluster->position().eta();
  double clPhi = cluster->position().phi();
  std::shared_ptr<const CaloCellGeometry> this_cell;
  EcalRecHitCollection::const_iterator rHit;

  const std::vector<std::pair<DetId, float>>& detId = cluster->hitsAndFractions();
  // Loop over recHits associated with the given SuperCluster
  for (const auto& hit : detId) {
    if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL) {
      rHit = recHitsEB_->find(hit.first);
      if (rHit == recHitsEB_->end()) {
        continue;
      }
      this_cell = ebGeom_->getGeometry(rHit->id());
    } else if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP) {
      rHit = recHitsEE_->find(hit.first);
      if (rHit == recHitsEE_->end()) {
        continue;
      }
      this_cell = eeGeom_->getGeometry(rHit->id());
    } else {
      continue;
    }

    GlobalPoint position = this_cell->getPosition();
    //take into account energy fractions
    double energyHit = rHit->energy() * hit.second;
    //form differences
    double dPhi = deltaPhi(position.phi(), clPhi);
    double dEta = position.eta() - clEta;
    if (energyHit > 0) {
      numeratorEtaWidth += energyHit * dEta * dEta;
      numeratorPhiWidth += energyHit * dPhi * dPhi;
    }
  }
  double etaWidth = sqrt(numeratorEtaWidth / denominator);
  double phiWidth = sqrt(numeratorPhiWidth / denominator);

  return std::make_pair(etaWidth, phiWidth);
}

std::vector<double> EcalClustersGraph::computeShowerShapes(const CaloCluster* cluster, bool full5x5 = false) {
  std::vector<double> showerVars_;
  showerVars_.resize(8);
  widths_ = computeCovariances(cluster);
  float e1 = 1.;
  float e4 = 0.;
  float r9 = 0.;

  if (full5x5) {
    if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL) {
      locCov_ = noZS::EcalClusterTools::localCovariances(*cluster, recHitsEB_, topology_);
      e1 = noZS::EcalClusterTools::eMax(*cluster, recHitsEB_);
      e4 = noZS::EcalClusterTools::eTop(*cluster, recHitsEB_, topology_) +
           noZS::EcalClusterTools::eRight(*cluster, recHitsEB_, topology_) +
           noZS::EcalClusterTools::eBottom(*cluster, recHitsEB_, topology_) +
           noZS::EcalClusterTools::eLeft(*cluster, recHitsEB_, topology_);
      r9 = noZS::EcalClusterTools::e3x3(*cluster, recHitsEB_, topology_) / cluster->energy();  //r9

    } else if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP) {
      locCov_ = noZS::EcalClusterTools::localCovariances(*cluster, recHitsEE_, topology_);
      e1 = noZS::EcalClusterTools::eMax(*cluster, recHitsEE_);
      e4 = noZS::EcalClusterTools::eTop(*cluster, recHitsEE_, topology_) +
           noZS::EcalClusterTools::eRight(*cluster, recHitsEE_, topology_) +
           noZS::EcalClusterTools::eBottom(*cluster, recHitsEE_, topology_) +
           noZS::EcalClusterTools::eLeft(*cluster, recHitsEE_, topology_);
      r9 = noZS::EcalClusterTools::e3x3(*cluster, recHitsEE_, topology_) / cluster->energy();  //r9
    }
  } else {
    if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL) {
      locCov_ = EcalClusterTools::localCovariances(*cluster, recHitsEB_, topology_);
      e1 = EcalClusterTools::eMax(*cluster, recHitsEB_);
      e4 = EcalClusterTools::eTop(*cluster, recHitsEB_, topology_) +
           EcalClusterTools::eRight(*cluster, recHitsEB_, topology_) +
           EcalClusterTools::eBottom(*cluster, recHitsEB_, topology_) +
           EcalClusterTools::eLeft(*cluster, recHitsEB_, topology_);
      r9 = EcalClusterTools::e3x3(*cluster, recHitsEB_, topology_) / cluster->energy();  //r9
    } else if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP) {
      locCov_ = EcalClusterTools::localCovariances(*cluster, recHitsEE_, topology_);
      e1 = EcalClusterTools::eMax(*cluster, recHitsEE_);
      e4 = EcalClusterTools::eTop(*cluster, recHitsEE_, topology_) +
           EcalClusterTools::eRight(*cluster, recHitsEE_, topology_) +
           EcalClusterTools::eBottom(*cluster, recHitsEE_, topology_) +
           EcalClusterTools::eLeft(*cluster, recHitsEE_, topology_);
      r9 = EcalClusterTools::e3x3(*cluster, recHitsEE_, topology_) / cluster->energy();
    }
  }
  showerVars_[0] = r9;
  showerVars_[1] = sqrt(locCov_[0]);                                      //sigmaietaieta
  showerVars_[2] = locCov_[1];                                            //sigmaietaiphi
  showerVars_[3] = (!edm::isFinite(locCov_[2])) ? 0. : sqrt(locCov_[2]);  //sigmaiphiiphi
  showerVars_[4] = (e1 != 0.) ? 1. - e4 / e1 : -999.;                     //swiss_cross
  showerVars_[5] = cluster->hitsAndFractions().size();                    //nXtals
  showerVars_[6] = widths_.first;                                         //etaWidth
  showerVars_[7] = widths_.second;                                        //phiWidth

  return showerVars_;
}

void EcalClustersGraph::fillVariables() {
  LogDebug("EcalClustersGraph") << "Fill tensorflow input vector";
  const auto& deepSCEval = scProducerCache_->deepSCEvaluator;
  // Reserving the batch dimension
  inputs_.clustersX.reserve(nSeeds_);
  inputs_.hitsX.reserve(nSeeds_);
  inputs_.windowX.reserve(nSeeds_);
  inputs_.isSeed.reserve(nSeeds_);

  // Looping on all the seeds (window)
  for (uint is = 0; is < nSeeds_; is++) {
    const auto seedPointer = (clusters_[is]).ptr().get();
    std::vector<DeepSCInputs::FeaturesMap> unscaledClusterFeatures;
    const auto& outEdges = graphMap_.getOutEdges(is);
    size_t ncls = outEdges.size();
    // Reserve the vector size
    inputs_.clustersX[is].reserve(ncls);
    inputs_.hitsX[is].reserve(ncls);
    inputs_.isSeed[is].reserve(ncls);
    unscaledClusterFeatures.reserve(ncls);
    // Loop on all the clusters
    for (const auto ic : outEdges) {
      LogTrace("EcalClustersGraph") << "seed: " << is << ", out edge --> " << ic;
      const auto clPointer = (clusters_[ic]).ptr().get();
      const auto& clusterFeatures = computeVariables(seedPointer, clPointer);
      for (const auto& [key, val] : clusterFeatures) {
        LogTrace("EcalCluster") << key << "=" << val;
      }
      unscaledClusterFeatures.push_back(clusterFeatures);
      // Select and scale only the requested variables for the tensorflow model input
      inputs_.clustersX[is].push_back(deepSCEval->getScaledInputs(clusterFeatures, deepSCEval->inputFeaturesClusters));
      // The scaling and feature selection on hits is performed inside the function for each hit
      inputs_.hitsX[is].push_back(fillHits(clPointer));
      inputs_.isSeed[is].push_back(ic == is);
    }
    // For window we need the unscaled cluster features and then we select them
    inputs_.windowX[is] =
        deepSCEval->getScaledInputs(computeWindowVariables(unscaledClusterFeatures), deepSCEval->inputFeaturesWindows);
  }
  LogTrace("EcalClustersGraph") << "N. Windows: " << inputs_.clustersX.size();
}

void EcalClustersGraph::evaluateScores() {
  // Evaluate model
  const auto& scores = scProducerCache_->deepSCEvaluator->evaluate(inputs_);
  for (uint i = 0; i < nSeeds_; ++i) {
    uint k = 0;
    LogTrace("EcalClustersGraph") << "Score) seed: " << i << ":";
    for (auto const& j : graphMap_.getOutEdges(i)) {
      // Fill the scores from seed --> node (i --> j)
      // Not symmetrically, in order to save multiple values for seeds in other
      // seeds windows.
      graphMap_.setAdjMatrix(i, j, scores[i][k]);
      LogTrace("EcalClustersGraph") << "\t" << i << "-->" << j << ": " << scores[i][k];
      k++;
    }
  }
}

void EcalClustersGraph::setThresholds() {
  // Simple global threshold for the moment
  threshold_ = 0.5;
}

void EcalClustersGraph::selectClusters() {
  // Collect the final superClusters as subgraphs
  graphMap_.collectNodes(strategy_, threshold_);
}

EcalClustersGraph::EcalGraphOutput EcalClustersGraph::getGraphOutput() {
  EcalClustersGraph::EcalGraphOutput finalWindows_;
  const auto& finalSuperClusters_ = graphMap_.getGraphOutput();
  for (const auto& [is, cls] : finalSuperClusters_) {
    CalibratedPFCluster seed = clusters_[is];
    CalibratedPFClusterVector clusters_inWindow;
    for (const auto& ic : cls) {
      clusters_inWindow.push_back(clusters_[ic]);
    }
    finalWindows_.push_back({seed, clusters_inWindow});
  }
  return finalWindows_;
}
