#ifndef HLTJetL1TMatchProducer_h
#define HLTJetL1TMatchProducer_h

#include <string>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"

template <typename T>
class HLTJetL1TMatchProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTJetL1TMatchProducer(const edm::ParameterSet&);
  ~HLTJetL1TMatchProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::InputTag jetsInput_;
  edm::InputTag L1Jets_;
  double DeltaR2_;  // DeltaR2(HLT,L1) with sign
  edm::EDGetTokenT<std::vector<T>> m_theJetToken;
  edm::EDGetTokenT<l1t::JetBxCollection> m_theL1JetToken;
};

#endif
