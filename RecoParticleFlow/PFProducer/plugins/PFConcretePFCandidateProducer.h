#ifndef RecoParticleFlow_PFProducer_PFPConcretePFCandidateProducer_h_
#define RecoParticleFlow_PFProducer_PFPConcretePFCandidateProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// useful?
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFConcretePFCandidateProducer : public edm::EDProducer {
 public:
  explicit PFConcretePFCandidateProducer(const edm::ParameterSet&);
  ~PFConcretePFCandidateProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  edm::InputTag  inputColl_;
};

#endif
