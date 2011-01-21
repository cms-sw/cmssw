/*
 * RecoTauJetRegionProducer
 *
 * Given a set of PFJets, make new jets with the same p4 but collect all the
 * PFCandidates from a cone of a given size into the constituents.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include <boost/bind.hpp>

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Common/interface/Association.h"

#include "RecoTauTag/RecoTau/interface/ConeTools.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RecoTauJetRegionProducer : public edm::EDProducer {
  public:
    typedef edm::Association<reco::PFJetCollection> PFJetMatchMap;
    explicit RecoTauJetRegionProducer(const edm::ParameterSet& pset);
    ~RecoTauJetRegionProducer() {}
    void produce(edm::Event& evt, const edm::EventSetup& es);
  private:
    double deltaR_;
    edm::InputTag inputJets_;
    edm::InputTag pfSrc_;
};

RecoTauJetRegionProducer::RecoTauJetRegionProducer(
    const edm::ParameterSet& pset) {
  deltaR_ = pset.getParameter<double>("deltaR");
  inputJets_ = pset.getParameter<edm::InputTag>("src");
  pfSrc_ = pset.getParameter<edm::InputTag>("pfSrc");
  produces<reco::PFJetCollection>("jets");
  produces<PFJetMatchMap>();
}

void RecoTauJetRegionProducer::produce(edm::Event& evt,
    const edm::EventSetup& es) {

  edm::Handle<reco::PFCandidateCollection> pfCandsHandle;
  evt.getByLabel(pfSrc_, pfCandsHandle);

  // Build Ptrs for all the PFCandidates
  typedef edm::Ptr<reco::PFCandidate> PFCandPtr;
  std::vector<PFCandPtr> pfCands;
  pfCands.reserve(pfCandsHandle->size());
  for (size_t icand = 0; icand < pfCandsHandle->size(); ++icand) {
    pfCands.push_back(PFCandPtr(pfCandsHandle, icand));
  }

  // Get the jets
  edm::Handle<reco::PFJetCollection> jetHandle;
  evt.getByLabel(inputJets_, jetHandle);

  std::auto_ptr<reco::PFJetCollection> newJets(new reco::PFJetCollection);

  size_t nJets = jetHandle->size();
  for (size_t ijet = 0; ijet < nJets; ++ijet) {
    // Get a ref to jet
    reco::PFJetRef jetRef(jetHandle, ijet);
    // Make an initial copy.
    reco::PFJet newJet(*jetRef);
    // Clear out all the constituents
    newJet.clearDaughters();
    // Build a DR cone filter about our jet
    reco::tau::cone::DeltaRPtrFilter<PFCandPtr>
      filter(jetRef->p4(), 0, deltaR_);

    // Loop over all the PFCands
    std::for_each(
        // filtering those that don't pass our filter
        boost::make_filter_iterator(filter,
          pfCands.begin(), pfCands.end()),
        boost::make_filter_iterator(filter,
          pfCands.end(), pfCands.end()),
        // For the ones that do, call newJet.addDaughter(..) on them
        boost::bind(&reco::PFJet::addDaughter, boost::ref(newJet), _1));
    newJets->push_back(newJet);
  }

  // Put our new jets into the event
  edm::OrphanHandle<reco::PFJetCollection> newJetsInEvent =
    evt.put(newJets, "jets");

  // Create a matching between original jets -> extra collection
  std::auto_ptr<PFJetMatchMap> matching(new PFJetMatchMap(newJetsInEvent));
  PFJetMatchMap::Filler filler(*matching);

  // Create a matchign between indices of the two collections
  // This is trivial, as they are one to one.
  std::vector<int> matchInfo(nJets);
  for (size_t i = 0; i < nJets; ++i) {
    matchInfo[i] = i;
  }
  filler.insert(jetHandle, matchInfo.begin(), matchInfo.end());
  filler.fill();
  evt.put(matching);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauJetRegionProducer);
