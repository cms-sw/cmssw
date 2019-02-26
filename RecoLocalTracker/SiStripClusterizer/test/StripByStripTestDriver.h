#ifndef StripByStripTestDriver_h
#define StripByStripTestDriver_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
//#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerFactory.h"

#include <memory>

class StripByStripTestDriver : public edm::EDProducer {
  
  typedef edmNew::DetSetVector<SiStripCluster> output_t;
  
public:  

  StripByStripTestDriver(const edm::ParameterSet&);
  ~StripByStripTestDriver();

private:

  void produce(edm::Event&, const edm::EventSetup&);

  const edm::InputTag inputTag;
  const bool hlt;

  //SiStripClusterizerFactory*               hltFactory;
  std::unique_ptr<StripClusterizerAlgorithm> algorithm;

};
#endif
