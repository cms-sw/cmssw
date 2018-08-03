#ifndef __RecoHGCal_TICL_TrackstersProducer_H__
#define __RecoHGCal_TICL_TrackstersProducer_H__
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"

class TrackstersProducer : public edm::stream::EDProducer<> {
public:
  TrackstersProducer(const edm::ParameterSet &);
  ~TrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::Event &, const edm::EventSetup &) override;


private:

  edm::EDGetTokenT<PatternRecognitionAlgoBase> patterRecognitionAlgoToken;

  edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters_token;
  edm::EDGetTokenT<std::vector<std::pair<unsigned int, float>>> filteredClustersMask_token;

  std::unique_ptr<PatternRecognitionAlgoBase> myAlgo_;

};


#endif
