#ifndef SHALLOW_RECHITCLUSTERS_PRODUCER
#define SHALLOW_RECHITCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

class ShallowRechitClustersProducer : public edm::EDProducer {
public:
  explicit ShallowRechitClustersProducer(const edm::ParameterSet&);
private:
  std::string Suffix;
  std::string Prefix;
  const edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > clusters_token_;
	std::vector< edm::EDGetTokenT<SiStripRecHit2DCollection> > rec_hits_tokens_;
  //std::vector<edm::InputTag> inputTags;
  void produce( edm::Event &, const edm::EventSetup & ) override;
};

#endif
