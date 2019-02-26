/*
 * =====================================================================================
 *       Filename:  RecoTauPiZeroFlattener.cc
 *
 *    Description:  Produce a plain vector<RecoTauPizero> from the the PiZeros in
 *    a JetPiZeroAssociation associated to the input jets.
 *        Created:  10/31/2010 12:33:41
 *
 *         Author:  Evan K. Friis (UC Davis), evan.klose.friis@cern.ch
 * =====================================================================================
 */

#include <algorithm>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

class RecoTauPiZeroFlattener : public edm::EDProducer {
  public:
    explicit RecoTauPiZeroFlattener(const edm::ParameterSet &pset);
    ~RecoTauPiZeroFlattener() override {}
    void produce(edm::Event& evt, const edm::EventSetup& es) override;
  private:
    edm::InputTag jetSrc_;
    edm::InputTag piZeroSrc_;
};

RecoTauPiZeroFlattener::RecoTauPiZeroFlattener(const edm::ParameterSet& pset) {
  jetSrc_ = pset.getParameter<edm::InputTag>("jetSrc");
  piZeroSrc_ = pset.getParameter<edm::InputTag>("piZeroSrc");
  produces<std::vector<reco::RecoTauPiZero> >();
}

void
RecoTauPiZeroFlattener::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Get the jet input collection via a view of Candidates
  edm::Handle<reco::CandidateView> jetView;
  evt.getByLabel(jetSrc_, jetView);

  // Convert to a vector of PFJetRefs
  reco::PFJetRefVector jets =
      reco::tau::castView<reco::PFJetRefVector>(jetView);

  // Get the pizero input collection
  edm::Handle<reco::JetPiZeroAssociation> piZeroAssoc;
  evt.getByLabel(piZeroSrc_, piZeroAssoc);

  // Create output collection
  auto output = std::make_unique<std::vector<reco::RecoTauPiZero>>();

  // Loop over the jets and append the pizeros for each jet to our output
  // collection.
  for(auto const& jetRef : jets) {
    const std::vector<reco::RecoTauPiZero>& pizeros = (*piZeroAssoc)[jetRef];
    output->reserve(output->size() + pizeros.size());
    output->insert(output->end(), pizeros.begin(), pizeros.end());
  }

  evt.put(std::move(output));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPiZeroFlattener);
