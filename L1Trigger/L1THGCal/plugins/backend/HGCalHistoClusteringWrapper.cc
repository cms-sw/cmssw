#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoClusteringImpl_SA.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalCluster_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalSeed_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticluster_SA.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalHistoClusteringWrapper : public HGCalHistoClusteringWrapperBase {
public:
  HGCalHistoClusteringWrapper(const edm::ParameterSet& conf);
  ~HGCalHistoClusteringWrapper() override {}

  void configure(
      const std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>& configuration) override;

  void process(const std::pair<const std::vector<edm::Ptr<l1t::HGCalCluster>>,
                               const std::vector<std::pair<GlobalPoint, double>>>& inputClustersAndSeeds,
               std::pair<l1t::HGCalMulticlusterBxCollection&, l1t::HGCalClusterBxCollection&>&
                   outputMulticlustersAndRejectedClusters) const override;

private:
  void convertCMSSWInputs(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
                          l1thgcfirmware::HGCalClusterSACollection& clusters_SA,
                          const std::vector<std::pair<GlobalPoint, double>>& seeds,
                          l1thgcfirmware::HGCalSeedSACollection& seeds_SA) const;
  void convertAlgorithmOutputs(const l1thgcfirmware::HGCalMulticlusterSACollection& multiclusters_out,
                               const l1thgcfirmware::HGCalClusterSACollection& rejected_clusters_out,
                               const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
                               l1t::HGCalMulticlusterBxCollection& multiclusters,
                               l1t::HGCalClusterBxCollection& rejected_clusters) const;

  void clusterizeHisto(const l1thgcfirmware::HGCalClusterSACollection& inputClusters,
                       const l1thgcfirmware::HGCalSeedSACollection& inputSeeds,
                       l1thgcfirmware::HGCalMulticlusterSACollection& outputMulticlusters,
                       l1thgcfirmware::HGCalClusterSACollection& outputRejectedClusters) const;

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

  HGCalTriggerTools triggerTools_;

  HGCalHistoClusteringImplSA theAlgo_;

  l1thgcfirmware::ClusterAlgoConfig theConfiguration_;

  static constexpr double kMidRadius_ = 2.3;
};

HGCalHistoClusteringWrapper::HGCalHistoClusteringWrapper(const edm::ParameterSet& conf)
    : HGCalHistoClusteringWrapperBase(conf),
      theAlgo_(),
      theConfiguration_(kMidRadius_,
                        conf.getParameter<double>("dR_multicluster"),
                        conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientA")
                            ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientA")
                            : std::vector<double>(),
                        conf.existsAs<std::vector<double>>("dR_multicluster_byLayer_coefficientB")
                            ? conf.getParameter<std::vector<double>>("dR_multicluster_byLayer_coefficientB")
                            : std::vector<double>(),
                        conf.getParameter<double>("minPt_multicluster")) {}

void HGCalHistoClusteringWrapper::convertCMSSWInputs(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
                                                     std::vector<l1thgcfirmware::HGCalCluster>& clusters_SA,
                                                     const std::vector<std::pair<GlobalPoint, double>>& seeds,
                                                     std::vector<l1thgcfirmware::HGCalSeed>& seeds_SA) const {
  clusters_SA.clear();
  clusters_SA.reserve(clustersPtrs.size());
  unsigned int clusterIndex = 0;
  for (const auto& cluster : clustersPtrs) {
    clusters_SA.emplace_back(cluster->centreProj().x(),
                             cluster->centreProj().y(),
                             cluster->centre().z(),
                             triggerTools_.zside(cluster->detId()),
                             triggerTools_.layerWithOffset(cluster->detId()),
                             cluster->eta(),
                             cluster->phi(),
                             cluster->pt(),
                             cluster->mipPt(),
                             clusterIndex);
    ++clusterIndex;
  }

  seeds_SA.clear();
  seeds_SA.reserve(seeds.size());
  for (const auto& seed : seeds) {
    seeds_SA.emplace_back(seed.first.x(), seed.first.y(), seed.first.z(), seed.second);
  }
}

void HGCalHistoClusteringWrapper::convertAlgorithmOutputs(
    const std::vector<l1thgcfirmware::HGCalMulticluster>& multiclusters_out,
    const std::vector<l1thgcfirmware::HGCalCluster>& rejected_clusters_out,
    const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
    l1t::HGCalMulticlusterBxCollection& multiclustersBXCollection,
    l1t::HGCalClusterBxCollection& rejected_clusters) const {
  // Not doing completely the correct thing here
  // Taking the multiclusters from the stand alone emulation
  // Getting their consistuent clusters (stand alone objects)
  // Linking back to the original CMSSW-type cluster
  // And creating a CMSSW-type multicluster based from these clusters
  // So the output multiclusters will not be storing bit accurate quantities (or whatever was derived by the stand along emulation)
  // As these inherit from L1Candidate, could set their HW quantities to the bit accurate ones
  for (const auto& rejected_cluster : rejected_clusters_out) {
    rejected_clusters.push_back(0, *clustersPtrs.at(rejected_cluster.index_cmssw()));
  }

  std::vector<l1t::HGCalMulticluster> multiclusters;
  multiclusters.reserve(multiclusters_out.size());
  for (unsigned int imulticluster = 0; imulticluster < multiclusters_out.size(); ++imulticluster) {
    bool firstConstituent = true;
    for (const auto& constituent : multiclusters_out[imulticluster].constituents()) {
      if (firstConstituent) {
        multiclusters.emplace_back(clustersPtrs.at(constituent.index_cmssw()), 1.);
      } else {
        multiclusters.at(imulticluster).addConstituent(clustersPtrs.at(constituent.index_cmssw()), 1.);
      }
      firstConstituent = false;
    }
  }

  for (const auto& multicluster : multiclusters) {
    multiclustersBXCollection.push_back(0, multicluster);
  }
}

void HGCalHistoClusteringWrapper::process(
    const std::pair<const std::vector<edm::Ptr<l1t::HGCalCluster>>, const std::vector<std::pair<GlobalPoint, double>>>&
        inputClustersAndSeeds,
    std::pair<l1t::HGCalMulticlusterBxCollection&, l1t::HGCalClusterBxCollection&>&
        outputMulticlustersAndRejectedClusters) const {
  l1thgcfirmware::HGCalClusterSACollection clusters_SA;
  l1thgcfirmware::HGCalSeedSACollection seeds_SA;
  convertCMSSWInputs(inputClustersAndSeeds.first, clusters_SA, inputClustersAndSeeds.second, seeds_SA);

  l1thgcfirmware::HGCalClusterSACollection rejected_clusters_finalized_SA;
  l1thgcfirmware::HGCalMulticlusterSACollection multiclusters_finalized_SA;
  clusterizeHisto(clusters_SA, seeds_SA, multiclusters_finalized_SA, rejected_clusters_finalized_SA);

  convertAlgorithmOutputs(multiclusters_finalized_SA,
                          rejected_clusters_finalized_SA,
                          inputClustersAndSeeds.first,
                          outputMulticlustersAndRejectedClusters.first,
                          outputMulticlustersAndRejectedClusters.second);
}

void HGCalHistoClusteringWrapper::clusterizeHisto(
    const l1thgcfirmware::HGCalClusterSACollection& inputClusters,
    const l1thgcfirmware::HGCalSeedSACollection& inputSeeds,
    l1thgcfirmware::HGCalMulticlusterSACollection& outputMulticlusters,
    l1thgcfirmware::HGCalClusterSACollection& outputRejectedClusters) const {
  // Call SA clustering
  std::vector<l1thgcfirmware::HGCalCluster> rejected_clusters_vec_SA;
  std::vector<l1thgcfirmware::HGCalMulticluster> multiclusters_vec_SA =
      theAlgo_.clusterSeedMulticluster_SA(inputClusters, inputSeeds, rejected_clusters_vec_SA, theConfiguration_);

  theAlgo_.finalizeClusters_SA(
      multiclusters_vec_SA, rejected_clusters_vec_SA, outputMulticlusters, outputRejectedClusters, theConfiguration_);
}

void HGCalHistoClusteringWrapper::configure(
    const std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>& configuration) {
  setGeometry(configuration.first);

  // theConfiguration_.setParameters( ... );

  if ((!theConfiguration_.dr_byLayer_coefficientA().empty() &&
       (theConfiguration_.dr_byLayer_coefficientA().size() - 1) < triggerTools_.lastLayerBH()) ||
      (!theConfiguration_.dr_byLayer_coefficientB().empty() &&
       (theConfiguration_.dr_byLayer_coefficientB().size() - 1) < triggerTools_.lastLayerBH())) {
    throw cms::Exception("Configuration")
        << "The per-layer dR values go up to " << (theConfiguration_.dr_byLayer_coefficientA().size() - 1) << "(A) and "
        << (theConfiguration_.dr_byLayer_coefficientB().size() - 1) << "(B), while layers go up to "
        << triggerTools_.lastLayerBH() << "\n";
  }
};

DEFINE_EDM_PLUGIN(HGCalHistoClusteringWrapperBaseFactory, HGCalHistoClusteringWrapper, "HGCalHistoClusteringWrapper");
