#ifndef DiJetPairsVarProducer_h
#define DiJetPairsVarProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TLorentzVector.h"

#include <vector>

class DiJetPairsVarProducer : public edm::EDProducer {
 public: 
  explicit DiJetPairsVarProducer(const edm::ParameterSet&);
  ~DiJetPairsVarProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  edm::InputTag inputJetTag_; // input tag identifying product
};

#endif //DiJetPairsVarProducer_h
