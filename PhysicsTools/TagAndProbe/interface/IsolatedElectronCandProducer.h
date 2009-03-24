#ifndef PhysicsTools_TagAndProbe__IsolatedElectronCandProducer_h
#define PhysicsTools_TagAndProbe__IsolatedElectronCandProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations

class IsolatedElectronCandProducer : public edm::EDProducer {
 public:
  explicit IsolatedElectronCandProducer(const edm::ParameterSet&);
  ~IsolatedElectronCandProducer();
  

 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
      
  // ----------member data ---------------------------
      
  edm::InputTag electronProducer_;
  edm::InputTag trackProducer_;
  edm::InputTag beamspotProducer_;

  double ptMin_;
  double intRadius_;
  double extRadius_;
  double maxVtxDist_;
  double isoCut_;
  double drb_;

  bool absolut_;
  
};


#endif
