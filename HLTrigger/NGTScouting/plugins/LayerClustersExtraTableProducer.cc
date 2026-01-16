#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

class LayerClustersExtraTableProducer : public edm::stream::EDProducer<> {
public:
  LayerClustersExtraTableProducer(edm::ParameterSet const& params)
      : skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")),
        tableName_(params.getParameter<std::string>("tableName")),
        precision_(params.getParameter<int>("precision")),
        clustersTime_token_(consumes<edm::ValueMap<std::pair<float, float>>>(
            params.getParameter<edm::InputTag>("time_layerclusters"))) {
    produces<nanoaod::FlatTable>(tableName_);
  }

  void produce(edm::Event& iEvent, const edm::EventSetup&) override {
    //Layer Clusters time value map
    auto clustersTime_h = iEvent.getHandle(clustersTime_token_);
    const auto nClusters = clustersTime_h.isValid() ? clustersTime_h->size() : 0;

    static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();

    std::vector<float> time(nClusters, default_value);
    std::vector<float> timeError(nClusters, default_value);

    // initialize to quiet Nans
    if (clustersTime_h.isValid() or !(skipNonExistingSrc_)) {
      const auto& clustersTime = *clustersTime_h;
      for (size_t i = 0; i < clustersTime.size(); ++i) {
        float t = clustersTime.get(i).first;
        float tE = clustersTime.get(i).first;
        time[i] = t;
        timeError[i] = tE;
      }
    }

    auto layerClustersTable =
        std::make_unique<nanoaod::FlatTable>(nClusters, tableName_, /*singleton*/ false, /*extension*/ true);
    layerClustersTable->addColumn<float>("time", time, "LayerCluster time [ns]", precision_);
    layerClustersTable->addColumn<float>("timeError", timeError, "LayerCluster timeError [ns]", precision_);
    iEvent.put(std::move(layerClustersTable), tableName_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("skipNonExistingSrc", false)
        ->setComment("whether or not to skip producing the table on absent input product");
    desc.add<std::string>("tableName", "hltMergeLayerClusters")->setComment("name of the flat table ouput");
    desc.add<edm::InputTag>("time_layerclusters", edm::InputTag("hltMergeLayerClusters", "timeLayerCluster"));
    desc.add<int>("precision", 7);
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const bool skipNonExistingSrc_;
  const std::string tableName_;
  const unsigned int precision_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClusters_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LayerClustersExtraTableProducer);
