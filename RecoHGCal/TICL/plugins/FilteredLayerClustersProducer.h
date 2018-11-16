#ifndef __RecoHGCal_TICL_FilteredLayerClustersProducer_H__
#define __RecoHGCal_TICL_FilteredLayerClustersProducer_H__
#include <string>
#include "RecoHGCal/TICL/interface/ClusterFilterBase.h"

class FilteredLayerClustersProducer : public edm::stream::EDProducer<> {
public:
  FilteredLayerClustersProducer(const edm::ParameterSet &);
  ~FilteredLayerClustersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;


private:

  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token_;
  edm::EDGetTokenT<std::vector<float>> clustersMask_token_;
  std::string clusterFilter_;
  std::string iteration_label_;
  const ClusterFilterBase* theFilter_ = nullptr;
};


#endif
