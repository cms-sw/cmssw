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
  std::string label_[2];
  std::string instanceName_[2];
  ElectronGSPixelSeedGenerator *matcher_;
 
  };
  
#endif
 


