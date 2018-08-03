#ifndef __RecoHGCal_TICL_FilteredLayerClustersProducer_H__
#define __RecoHGCal_TICL_FilteredLayerClustersProducer_H__
#include <string>
#include "RecoHGCal/TICL/interface/ClusterFilter.h"
class FilteredLayerClustersProducer : public edm::stream::EDProducer<> {
public:
  FilteredLayerClustersProducer(const edm::ParameterSet &);
  ~FilteredLayerClustersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;


private:

  edm::EDGetTokenT<ClusterFilter> clusterFilterToken;
  std::string iterationLabel;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token;
  edm::EDGetTokenT<std::vector<float>> clustersMask_token;
  const ClusterFilter* theFilter = nullptr;
};


#endif
