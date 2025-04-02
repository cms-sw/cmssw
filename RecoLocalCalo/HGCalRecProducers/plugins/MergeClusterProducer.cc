// Authors: Olivie Franklova - olivie.abigail.franklova@cern.ch
// Date: 03/2023
// @file merge layer clusters which were produce by HGCalLayerClusterProducer

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"

class MergeClusterProducer : public edm::stream::EDProducer<> {
public:
  /**
   * @brief Constructor with parameter settings - which can be changed in  ...todo.
   * Constructor will set all variables by input param ps.
   *
   * @param[in] ps parametr set to set variables
  */
  MergeClusterProducer(const edm::ParameterSet &);
  ~MergeClusterProducer() override {}
  /**
   * @brief Method fill description which will be used in pyhton file.
   *
   * @param[out] description to be fill
  */
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  /**
   * @brief Method will merge the producers and put them back to event
   *
   * @param[in, out] evt from get info and put result
   * @param[in] es to get event setup info
  */
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  std::vector<edm::EDGetTokenT<std::vector<reco::CaloCluster>>> tokens_;

  std::vector<edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>>> time_tokens_;
  std::string timeClname_;
  /**
   * @brief method merge three vectors of reco::CaloCluster to one
   *
   * @param[out] merge the vector into which others vectors will be merge
   * @param[in] EE vector for Electromagnetic silicon
   * @param[in] HSi vector for Hardon silicon
   * @param[in] ESci vector for hadron scintillator
  */
  void mergeTogether(std::vector<reco::CaloCluster> &merge, std::vector<std::vector<reco::CaloCluster>> &clusters);

  /**
   * @brief copy all values from vm to to
   *
   * @param[in] vm Value map with values
   * @param[out] to vector to will be copy value map
  */
  void addTo(std::vector<std::pair<float, float>> &to, const edm::ValueMap<std::pair<float, float>> &vm) {
    size_t size = vm.size();
    for (size_t i = 0; i < size; ++i) {
      to.push_back(vm.get(i));
    }
  }
  /**
   * @brief Merge value map of time for all parts of detector together  to vector times
   *
   * @param[in] evt Event to get time value maps
   * @param[in] size of all 3 value maps
   * @param[out] times vector of merged time vectors
  */
  void mergeTime(edm::Event &evt, size_t size, std::vector<std::pair<float, float>> &times) {
    // get values from all three part of detectors
    std::vector<edm::ValueMap<std::pair<float, float>>> time_maps;
    for (auto token : time_tokens_) {
      edm::Handle<edm::ValueMap<std::pair<float, float>>> handle;
      evt.getByToken(token, handle);
      time_maps.push_back(*handle);
    }
    times.reserve(size);
    for (const auto &vm : time_maps) {
      addTo(times, vm);
    }
  }
  /**
   * @brief get info form event and then call merge
   *
   * it is used for merge and clusters and time
   *
   * @param[in] evt Event
   * @param[in] EE_token token for Electromagnetic silicon
   * @param[in] HSi_token token for Hardon silicon
   * @param[in] ESci_token token for hadron scintillator
   * @return merged result
  */
  template <typename T>
  void createMerge(edm::Event &evt, std::vector<edm::EDGetTokenT<T>> &tokens, T &merge) {
    std::vector<T> handles;
    for (auto token : tokens) {
      edm::Handle<T> handle;
      evt.getByToken(token, handle);
      handles.push_back(*handle);
    }
    mergeTogether(merge, handles);
  }
};

MergeClusterProducer::MergeClusterProducer(const edm::ParameterSet &ps)
    : timeClname_(ps.getParameter<std::string>("timeClname")) {
  std::vector<edm::InputTag> tags = ps.getParameter<std::vector<edm::InputTag>>("layerClusters");
  for (auto &tag : tags) {
    tokens_.push_back(consumes<std::vector<reco::CaloCluster>>(tag));
  }
  std::vector<edm::InputTag> time_tags = ps.getParameter<std::vector<edm::InputTag>>("time_layerclusters");
  for (auto &tag : time_tags) {
    time_tokens_.push_back(consumes<edm::ValueMap<std::pair<float, float>>>(tag));
  }

  produces<std::vector<float>>("InitialLayerClustersMask");
  produces<std::vector<reco::BasicCluster>>();
  produces<std::vector<reco::BasicCluster>>("sharing");
  //time for layer clusters
  produces<edm::ValueMap<std::pair<float, float>>>(timeClname_);
}

void MergeClusterProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // hgcalMergeLayerClusters
  edm::ParameterSetDescription desc;
  //layer clusters
  desc.add<std::vector<edm::InputTag>>("layerClusters",
                                       {edm::InputTag("hgcalLayerClustersEE"),
                                        edm::InputTag("hgcalLayerClustersHSi"),
                                        edm::InputTag("hgcalLayerClustersHSci")});
  //time
  desc.add<std::vector<edm::InputTag>>("time_layerclusters",
                                       {edm::InputTag("hgcalLayerClustersEE:timeLayerCluster"),
                                        edm::InputTag("hgcalLayerClustersHSi:timeLayerCluster"),
                                        edm::InputTag("hgcalLayerClustersHSci:timeLayerCluster")});
  desc.add<std::string>("timeClname", "timeLayerCluster");
  descriptions.add("hgcalMergeLayerClusters", desc);
}

void MergeClusterProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  //merge clusters
  std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>);
  createMerge(evt, tokens_, *clusters);
  //put new clusters to event
  auto clusterHandle = evt.put(std::move(clusters));

  //create layer cluster mask
  std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
  layerClustersMask->resize(clusterHandle->size(), 1.0);
  //put it into event
  evt.put(std::move(layerClustersMask), "InitialLayerClustersMask");

  //time
  std::vector<std::pair<float, float>> times;
  mergeTime(evt, clusterHandle->size(), times);

  auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
  filler.insert(clusterHandle, times.begin(), times.end());
  filler.fill();
  evt.put(std::move(timeCl), timeClname_);
}

void MergeClusterProducer::mergeTogether(std::vector<reco::CaloCluster> &merge,
                                         std::vector<std::vector<reco::CaloCluster>> &clusters) {
  size_t clusterSize = 0;
  for (auto &cl : clusters) {
    clusterSize += cl.size();
  }
  merge.reserve(clusterSize);

  for (auto &cl : clusters) {
    merge.insert(merge.end(), cl.begin(), cl.end());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MergeClusterProducer);
