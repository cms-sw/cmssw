#ifndef SHALLOW_SIMHITCLUSTERS_PRODUCER
#define SHALLOW_SIMHITCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"
class SiStripCluster;

class ShallowSimhitClustersProducer : public edm::EDProducer {
 public:
  explicit ShallowSimhitClustersProducer(const edm::ParameterSet&);
 private:
  std::vector<edm::InputTag> inputTags;
  edm::InputTag theClustersLabel;
  std::string Prefix;

  void produce( edm::Event &, const edm::EventSetup & );
  shallow::CLUSTERMAP::const_iterator match_cluster(const unsigned&, 
						    const float&, 
						    const shallow::CLUSTERMAP&, 
						    const edmNew::DetSetVector<SiStripCluster>& ) const;
};
#endif
