#ifndef SHALLOW_SIMHITCLUSTERS_PRODUCER
#define SHALLOW_SIMHITCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

class ShallowSimhitClustersProducer : public edm::EDProducer {
 public:
  explicit ShallowSimhitClustersProducer(const edm::ParameterSet&);
 private:
  //std::vector<edm::InputTag> inputTags;
	std::vector< edm::EDGetTokenT< std::vector<PSimHit> > > simhits_tokens_;
  const edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  std::string Prefix;
	std::string runningmode_;

  void produce( edm::Event &, const edm::EventSetup & ) override;
  shallow::CLUSTERMAP::const_iterator match_cluster(const unsigned&, 
						    const float&, 
						    const shallow::CLUSTERMAP&, 
						    const edmNew::DetSetVector<SiStripCluster>& ) const;
};
#endif
