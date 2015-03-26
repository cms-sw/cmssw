#ifndef PFJETSMATCHEDTOFILTEREDCALOJETSPRODUCER_H
#define PFJETSMATCHEDTOFILTEREDCALOJETSPRODUCER_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


#include <map>
#include <vector>

class PFJetsMatchedToFilteredCaloJetsProducer: public edm::stream::EDProducer<> {
 public:
  explicit PFJetsMatchedToFilteredCaloJetsProducer(const edm::ParameterSet&);
  ~PFJetsMatchedToFilteredCaloJetsProducer();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  std::vector<reco::CaloJetRef> jetRefVec;

  std::vector<l1extra::L1JetParticleRef> tauCandRefVec;
  std::vector<l1extra::L1JetParticleRef> jetCandRefVec;
  std::vector<l1extra::L1JetParticleRef> objL1CandRefVec;
  l1extra::L1JetParticleRef tauCandRef;
  
  edm::EDGetTokenT<edm::View<reco::Candidate>> m_thePFJetToken;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_theTriggerJetToken;

  edm::InputTag PFJetSrc;
  edm::InputTag CaloJetFilter;
  double DeltaR_;         // DeltaR(HLT,L1)
  int TriggerType_;

};
#endif
