#ifndef RecoParticleFlow_PFProducer_PFPConcretePFCandidateProducer_h_
#define RecoParticleFlow_PFProducer_PFPConcretePFCandidateProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// useful?
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFConcretePFCandidateProducer : public edm::stream::EDProducer<> {
 public:
  explicit PFConcretePFCandidateProducer(const edm::ParameterSet&);
  ~PFConcretePFCandidateProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  edm::InputTag  inputColl_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFConcretePFCandidateProducer);

#endif
