#ifndef ThreeJetVarProducer_h
#define ThreeJetVarProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TLorentzVector.h"

#include <vector>

class ThreeJetVarProducer : public edm::EDProducer {
 public: 
  explicit ThreeJetVarProducer(const edm::ParameterSet&);
  ~ThreeJetVarProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  edm::InputTag inputJetTag_; // input tag identifying product
};

#endif //ThreeJetVarProducer_h
