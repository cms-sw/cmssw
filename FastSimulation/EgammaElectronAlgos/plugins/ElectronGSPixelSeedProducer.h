#ifndef ElectronGSPixelSeedProducer_h
#define ElectronGSPixelSeedProducer_h
  
//
// Package:         FastSimulation/EgammaElectronAlgos
// Class:           ElectronGSPixelSeedProducer
// 
// Description:     Calls ElectronGSPixelSeedGenerator
//                  to find TrackingSeeds.
  
  
#include "FWCore/Framework/interface/EDProducer.h"
  
class  ElectronGSPixelSeedGenerator;

namespace edm { 
  class EventSetup;
  class Event;
  class ParameterSet;
}
 
class ElectronGSPixelSeedProducer : public edm::EDProducer
{
 public:
  
  explicit ElectronGSPixelSeedProducer(const edm::ParameterSet& conf);
  
  virtual ~ElectronGSPixelSeedProducer();
  
  virtual void beginJob(edm::EventSetup const& iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  edm::InputTag clusters_[2];
  edm::InputTag simTracks_;
  edm::InputTag trackerHits_;
  ElectronGSPixelSeedGenerator *matcher_;
 
  };
  
#endif
 


