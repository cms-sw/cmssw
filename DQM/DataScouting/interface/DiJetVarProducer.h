#ifndef DiJetVarProducer_h
#define DiJetVarProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TLorentzVector.h"
#include "TVector3.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <vector>

class DiJetVarProducer : public edm::EDProducer {
public:
  explicit DiJetVarProducer(const edm::ParameterSet &);
  ~DiJetVarProducer() override;
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::InputTag inputJetTag_;  // input tag jets
  double wideJetDeltaR_;       // Radius parameter for wide jets

  // set Token(-s)
  edm::EDGetTokenT<reco::CaloJetCollection> inputJetTagToken_;
};

#endif  // DiJetVarProducer_h
