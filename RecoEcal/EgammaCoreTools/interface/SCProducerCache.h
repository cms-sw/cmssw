#ifndef RecoEcal_EgammaCoreTools_SCProducerCache_h
#define RecoEcal_EgammaCoreTools_SCProducerCache_h

#include "RecoEcal/EgammaCoreTools/interface/DeepSCGraphEvaluation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {

  class SCProducerCache {
  public:
    // Cache for SuperCluster Producer containing Tensorflow objects
    SCProducerCache(const edm::ParameterSet& conf) {
      // Here we will have to load the DNN PFID if present in the config
      auto clustering_type = conf.getParameter<std::string>("ClusteringType");

      if (clustering_type == "DeepSC") {
        const auto& pset_dnn = conf.getParameter<edm::ParameterSet>("deepSuperClusterConfig");
        config.modelFile = pset_dnn.getParameter<std::string>("modelFile");
        config.configFileClusterFeatures = pset_dnn.getParameter<std::string>("configFileClusterFeatures");
        config.configFileWindowFeatures = pset_dnn.getParameter<std::string>("configFileWindowFeatures");
        config.configFileHitsFeatures = pset_dnn.getParameter<std::string>("configFileHitsFeatures");
        config.nClusterFeatures = pset_dnn.getParameter<uint>("nClusterFeatures");
        config.nWindowFeatures = pset_dnn.getParameter<uint>("nWindowFeatures");
        config.nHitsFeatures = pset_dnn.getParameter<uint>("nHitsFeatures");
        config.maxNClusters = pset_dnn.getParameter<uint>("maxNClusters");
        config.maxNRechits = pset_dnn.getParameter<uint>("maxNRechits");
        config.batchSize = pset_dnn.getParameter<uint>("batchSize");
        config.collectionStrategy = pset_dnn.getParameter<std::string>("collectionStrategy");
        deepSCEvaluator = std::make_unique<DeepSCGraphEvaluation>(config);
      }
    };

    std::unique_ptr<const DeepSCGraphEvaluation> deepSCEvaluator;
    reco::DeepSCConfiguration config;
  };
}  // namespace reco

#endif
