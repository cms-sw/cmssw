#ifndef SiStripRecHitConverter_h
#define SiStripRecHitConverter_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"

class SiStripRecHitConverter : public edm::EDProducer
{
  
 public:
  
  explicit SiStripRecHitConverter(const edm::ParameterSet& conf);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  
  SiStripRecHitConverterAlgorithm recHitConverterAlgorithm_;
  edm::ParameterSet conf_;
  std::string matchedRecHitsTag_, rphiRecHitsTag_, stereoRecHitsTag_;
  std::string np_;
  bool m_newCont; // save also in emdNew::DetSetVector
  
};
#endif
