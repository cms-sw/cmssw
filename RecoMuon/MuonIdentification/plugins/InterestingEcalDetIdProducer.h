#ifndef MuonIdentification_InterestingEcalDetIdProducer_h
#define MuonIdentification_InterestingEcalDetIdProducer_h
// -*- C++ -*-
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class CaloTopology;
class InterestingEcalDetIdProducer : public edm::EDProducer {
 public:
  explicit InterestingEcalDetIdProducer(const edm::ParameterSet&);
  ~InterestingEcalDetIdProducer();
  void produce(edm::Event&, const edm::EventSetup&);
  void beginRun(edm::Run&, const edm::EventSetup&);

 private:
  edm::InputTag inputCollection_;
  const CaloTopology* caloTopology_;
};

#endif
