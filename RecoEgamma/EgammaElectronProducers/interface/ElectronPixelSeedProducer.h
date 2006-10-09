#ifndef ElectronPixelSeedProducer_h
#define ElectronPixelSeedProducer_h
  
//
// Package:         RecoEgamma/ElectronTrackSeedProducers
// Class:           ElectronPixelSeedProducer
// 
// Description:     Calls ElectronPixelSeedGenerator
//                  to find TrackingSeeds.
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class  ElectronPixelSeedGenerator;
 
class ElectronPixelSeedProducer : public edm::EDProducer
{
 public:
  
  explicit ElectronPixelSeedProducer(const edm::ParameterSet& conf);
  
  virtual ~ElectronPixelSeedProducer();
  
  virtual void beginJob(edm::EventSetup const&iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  std::string label_[2];
  std::string instanceName_[2];
  const edm::ParameterSet conf_;
  ElectronPixelSeedGenerator *matcher_;
 
  };
  
#endif
 


