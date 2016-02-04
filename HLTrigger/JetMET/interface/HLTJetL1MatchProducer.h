#ifndef HLTJetL1MatchProducer_h
#define HLTJetL1MatchProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

class HLTJetL1MatchProducer : public edm::EDProducer {
 public:
  explicit HLTJetL1MatchProducer(const edm::ParameterSet&);
  ~HLTJetL1MatchProducer();
  virtual void beginJob() ; 
  virtual void produce(edm::Event &, const edm::EventSetup&);
 private:
  edm::InputTag jetsInput_;
  edm::InputTag L1TauJets_;
  edm::InputTag L1CenJets_;
  edm::InputTag L1ForJets_;
  double DeltaR_;         // DeltaR(HLT,L1)
};

#endif
