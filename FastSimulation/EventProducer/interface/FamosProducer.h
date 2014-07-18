#ifndef FastSimulation_EventProducer_FamosProducer_H
#define FastSimulation_EventProducer_FamosProducer_H

#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h" // future obsolete
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"    // obsolete

class FamosManager;
class ParameterSet;
class Event;
class EventSetup;

namespace HepMC { 
  class GenEvent;
}

class FamosProducer : public edm::EDProducer
{

 public:

  explicit FamosProducer(edm::ParameterSet const & p);
  virtual ~FamosProducer();
  virtual void beginRun(edm::Run const& run, const edm::EventSetup & es) override;
  virtual void endJob() override;
  virtual void produce(edm::Event & e, const edm::EventSetup & c) override;

 private:

  FamosManager * famosManager_;
  HepMC::GenEvent * evt_;
  bool simulateMuons;

  // labels
  edm::InputTag sourceLabel; // FUTURE OBSOLETE
  edm::InputTag genParticleLabel;
  edm::InputTag beamSpotLabel;

  // tokens
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken;
  // FUTURE OBSOLETE CODE
  edm::EDGetTokenT<edm::HepMCProduct> sourceToken;
  edm::EDGetTokenT<edm::HepMCProduct> puToken;
  // OBSOLETE CODE
  edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct> > mixSourceToken;
  edm::EDGetTokenT<reco::GenParticleCollection> mixGenParticleToken;
};

#endif
