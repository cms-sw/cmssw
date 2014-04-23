#ifndef MuonIdentification_InterestingEcalDetIdProducer_h
#define MuonIdentification_InterestingEcalDetIdProducer_h
// -*- C++ -*-
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class CaloTopology;
class InterestingEcalDetIdProducer : public edm::stream::EDProducer<> {
 public:
  explicit InterestingEcalDetIdProducer(const edm::ParameterSet&);
  ~InterestingEcalDetIdProducer();
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

 private:
  edm::InputTag inputCollection_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const CaloTopology* caloTopology_;
};

#endif
