#ifndef RecoLocalTracker_SiStripClusterizer_h
#define RecoLocalTracker_SiStripClusterizer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"

class SiStripClusterizer : public edm::EDProducer  {

public:

  explicit SiStripClusterizer(const edm::ParameterSet& conf);
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  template<class T> bool findInput(const edm::InputTag&, edm::Handle<T>&, const edm::Event&);
  const std::vector<edm::InputTag> inputTags;
  std::auto_ptr<StripClusterizerAlgorithm> algorithm;

};

#endif
