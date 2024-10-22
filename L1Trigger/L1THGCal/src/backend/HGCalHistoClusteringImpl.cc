#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoClusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HGCalHistoClusteringImpl::HGCalHistoClusteringImpl(const edm::ParameterSet& conf)
    : dr_(conf.getParameter<double>("dR_multicluster")),
      dr_byLayer_coefficientA_(conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientA")
                                   ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientA")
                                   : std::vector<double>()),
      dr_byLayer_coefficientB_(conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientB")
                                   ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientB")
                                   : std::vector<double>()),
      ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
      cluster_association_input_(conf.getParameter<string>("cluster_association")),
      shape_(conf) {
  if (cluster_association_input_ == "NearestNeighbour") {
    cluster_association_strategy_ = NearestNeighbour;
  } else if (cluster_association_input_ == "EnergySplit") {
    cluster_association_strategy_ = EnergySplit;
  } else {
    throw cms::Exception("HGCTriggerParameterError")
        << "Unknown cluster association strategy'" << cluster_association_strategy_;
  }

  edm::LogInfo("HGCalMulticlusterParameters")
      << "Multicluster dR: " << dr_ << "\nMulticluster minimum transverse-momentum: " << ptC3dThreshold_;

  id_ = std::unique_ptr<HGCalTriggerClusterIdentificationBase>{
      HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT")};
  id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification"));
}

float HGCalHistoClusteringImpl::dR(const l1t::HGCalCluster& clu, const GlobalPoint& seed) const {
  return (seed - clu.centreProj()).mag();
}

std::vector<l1t::HGCalMulticluster> HGCalHistoClusteringImpl::clusterSeedMulticluster(
    const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
    const std::vector<std::pair<GlobalPoint, double>>& seeds,
    std::vector<l1t::HGCalCluster>& rejected_clusters) const {
  std::map<int, l1t::HGCalMulticluster> mapSeedMulticluster;
  std::vector<l1t::HGCalMulticluster> multiclustersOut;

  for (const auto& clu : clustersPtrs) {
    int z_side = triggerTools_.zside(clu->detId());

    double radiusCoefficientA =
        dr_byLayer_coefficientA_.empty() ? dr_ : dr_byLayer_coefficientA_[triggerTools_.layerWithOffset(clu->detId())];
    double radiusCoefficientB =
        dr_byLayer_coefficientB_.empty() ? 0 : dr_byLayer_coefficientB_[triggerTools_.layerWithOffset(clu->detId())];

    double minDist = radiusCoefficientA + radiusCoefficientB * (kMidRadius_ - std::abs(clu->eta()));

    std::vector<pair<int, double>> targetSeedsEnergy;

    for (unsigned int iseed = 0; iseed < seeds.size(); iseed++) {
      GlobalPoint seedPosition = seeds[iseed].first;
      double seedEnergy = seeds[iseed].second;

      if (z_side * seedPosition.z() < 0)
        continue;
      double d = this->dR(*clu, seeds[iseed].first);

      if (d < minDist) {
        if (cluster_association_strategy_ == EnergySplit) {
          targetSeedsEnergy.emplace_back(iseed, seedEnergy);
        } else if (cluster_association_strategy_ == NearestNeighbour) {
          minDist = d;

          if (targetSeedsEnergy.empty()) {
            targetSeedsEnergy.emplace_back(iseed, seedEnergy);
          } else {
            targetSeedsEnergy.at(0).first = iseed;
            targetSeedsEnergy.at(0).second = seedEnergy;
          }
        }
      }
    }

    if (targetSeedsEnergy.empty()) {
      rejected_clusters.emplace_back(*clu);
      continue;
    }
    //Loop over target seeds and divide up the clusters energy
    double totalTargetSeedEnergy = 0;
    for (const auto& energy : targetSeedsEnergy) {
      totalTargetSeedEnergy += energy.second;
    }

    for (const auto& energy : targetSeedsEnergy) {
      double seedWeight = 1;
      if (cluster_association_strategy_ == EnergySplit)
        seedWeight = energy.second / totalTargetSeedEnergy;
      if (mapSeedMulticluster[energy.first].size() == 0) {
        mapSeedMulticluster[energy.first] = l1t::HGCalMulticluster(clu, seedWeight);
      } else {
        mapSeedMulticluster[energy.first].addConstituent(clu, true, seedWeight);
      }
    }
  }

  multiclustersOut.reserve(mapSeedMulticluster.size());
  for (const auto& mclu : mapSeedMulticluster)
    multiclustersOut.emplace_back(mclu.second);

  return multiclustersOut;
}

void HGCalHistoClusteringImpl::clusterizeHisto(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
                                               const std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy,
                                               const HGCalTriggerGeometryBase& triggerGeometry,
                                               l1t::HGCalMulticlusterBxCollection& multiclusters,
                                               l1t::HGCalClusterBxCollection& rejected_clusters) const {
  /* clusterize clusters around seeds */
  std::vector<l1t::HGCalCluster> rejected_clusters_vec;
  std::vector<l1t::HGCalMulticluster> multiclusters_vec =
      clusterSeedMulticluster(clustersPtrs, seedPositionsEnergy, rejected_clusters_vec);
  /* making the collection of multiclusters */
  finalizeClusters(multiclusters_vec, rejected_clusters_vec, multiclusters, rejected_clusters, triggerGeometry);
}

void HGCalHistoClusteringImpl::finalizeClusters(std::vector<l1t::HGCalMulticluster>& multiclusters_in,
                                                const std::vector<l1t::HGCalCluster>& rejected_clusters_in,
                                                l1t::HGCalMulticlusterBxCollection& multiclusters_out,
                                                l1t::HGCalClusterBxCollection& rejected_clusters_out,
                                                const HGCalTriggerGeometryBase& triggerGeometry) const {
  for (const auto& tc : rejected_clusters_in) {
    rejected_clusters_out.push_back(0, tc);
  }

  for (auto& multicluster : multiclusters_in) {
    // compute the eta, phi from its barycenter
    // + pT as scalar sum of pT of constituents
    double sumPt = multicluster.sumPt();

    math::PtEtaPhiMLorentzVector multiclusterP4(sumPt, multicluster.centre().eta(), multicluster.centre().phi(), 0.);
    multicluster.setP4(multiclusterP4);

    if (multicluster.pt() > ptC3dThreshold_) {
      //compute shower shapes
      shape_.fillShapes(multicluster, triggerGeometry);
      // fill quality flag
      unsigned hwQual = 0;
      for (unsigned wp = 0; wp < id_->working_points().size(); wp++) {
        hwQual |= (id_->decision(multicluster, wp) << wp);
      }
      multicluster.setHwQual(hwQual);
      // fill H/E
      multicluster.saveHOverE();

      multiclusters_out.push_back(0, multicluster);
    } else {
      for (const auto& tc : multicluster.constituents()) {
        rejected_clusters_out.push_back(0, *(tc.second));
      }
    }
  }
}
