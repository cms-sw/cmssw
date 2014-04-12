#ifndef SHALLOW_TRACKCLUSTERS_PRODUCER
#define SHALLOW_TRACKCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ShallowTrackClustersProducer : public edm::EDProducer {
public:
  explicit ShallowTrackClustersProducer(const edm::ParameterSet&);
private:
  edm::InputTag theTracksLabel;
  edm::InputTag theClustersLabel;
  std::string Suffix;
  std::string Prefix;

  void produce( edm::Event &, const edm::EventSetup & );
};
#endif
