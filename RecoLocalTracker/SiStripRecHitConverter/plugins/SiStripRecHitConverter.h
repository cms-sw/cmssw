#ifndef SiStripRecHitConverter_h
#define SiStripRecHitConverter_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"

class SiStripRecHitConverter : public edm::EDProducer
{
  
 public:
  
  explicit SiStripRecHitConverter(const edm::ParameterSet&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  SiStripRecHitConverterAlgorithm recHitConverterAlgorithm;
  std::string matchedRecHitsTag, rphiRecHitsTag, stereoRecHitsTag;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterProducer;

};
#endif
