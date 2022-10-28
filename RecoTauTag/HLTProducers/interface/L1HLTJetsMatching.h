#ifndef L1HLTJETSMATCHING_H
#define L1HLTJETSMATCHING_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include <map>
#include <vector>

class L1HLTJetsMatching : public edm::global::EDProducer<> {
public:
  explicit L1HLTJetsMatching(const edm::ParameterSet&);
  ~L1HLTJetsMatching() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::EDGetTokenT<edm::View<reco::Candidate> > jetSrc;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauTrigger;
  double mEt_Min;
};
#endif
