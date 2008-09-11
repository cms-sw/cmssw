#ifndef ElectronSiStripSeedProducer_h
#define ElectronSiStripSeedProducer_h
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class ElectronSiStripSeedGenerator;
 
class ElectronSiStripSeedProducer : public edm::EDProducer
{
 public:
  
  explicit ElectronSiStripSeedProducer(const edm::ParameterSet& conf);
  
  virtual ~ElectronSiStripSeedProducer();
  
  virtual void beginJob(edm::EventSetup const&iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  edm::InputTag superClusters_[2];
  const edm::ParameterSet conf_;
  ElectronSiStripSeedGenerator *matcher_;
  };
  
#endif
