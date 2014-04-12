#ifndef SHALLOW_RECHITCLUSTERS_PRODUCER
#define SHALLOW_RECHITCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ShallowRechitClustersProducer : public edm::EDProducer {
public:
  explicit ShallowRechitClustersProducer(const edm::ParameterSet&);
private:
  std::string Suffix;
  std::string Prefix;
  edm::InputTag theClustersLabel;
  std::vector<edm::InputTag> inputTags;
  void produce( edm::Event &, const edm::EventSetup & );
};

#endif
