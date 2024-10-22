#ifndef HLTrigger_JetMET_HLTJetCollForElePlusJets_h
#define HLTrigger_JetMET_HLTJetCollForElePlusJets_h

/** \class HLTJetCollForElePlusJets
 *
 *
 *  This class is an EDProducer implementing an HLT
 *  trigger for electron and jet objects, cutting on
 *  variables relating to the jet 4-momentum representation.
 *  The producer checks for overlaps between electrons and jets and if a
 *  combination of one electron + jets cleaned against this electrons satisfy the cuts.
 *  These jets are then added to a cleaned jet collection which is put into the event.
 *
 *
 *  \author Lukasz Kreczko
 *
 */

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

template <typename T>
class HLTJetCollForElePlusJets : public edm::stream::EDProducer<> {
public:
  explicit HLTJetCollForElePlusJets(const edm::ParameterSet&);
  ~HLTJetCollForElePlusJets() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_theElectronToken;
  edm::EDGetTokenT<std::vector<T>> m_theJetToken;
  edm::InputTag hltElectronTag;
  edm::InputTag sourceJetTag;

  double minJetPt_;        // jet pt threshold in GeV
  double maxAbsJetEta_;    // jet |eta| range
  unsigned int minNJets_;  // number of required jets passing cuts after cleaning
  double minDeltaR2_;      // min dR^2 (with sign) for jets and electrons not to match
  double minSoftJetPt_;    // jet pt threshold for the soft jet in the VBF pair
  double minDeltaEta_;     // pseudorapidity separation for the VBF pair
};

#endif
