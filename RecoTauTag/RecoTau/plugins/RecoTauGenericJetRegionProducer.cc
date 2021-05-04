/*
 * RecoTauJetRegionProducer
 *
 * Given a set of Jets, make new jets with the same p4 but collect all the
 * particle-flow candidates from a cone of a given size into the constituents.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "RecoTauTag/RecoTau/interface/ConeTools.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include <string>
#include <iostream>

template <class JetType, class CandType>
class RecoTauGenericJetRegionProducer : public edm::stream::EDProducer<> {
public:
  typedef edm::AssociationMap<edm::OneToOne<reco::JetView, reco::JetView> > JetMatchMap;
  typedef edm::AssociationMap<edm::OneToMany<std::vector<JetType>, std::vector<CandType>, unsigned int> >
      JetToCandidateAssociation;
  explicit RecoTauGenericJetRegionProducer(const edm::ParameterSet& pset);
  ~RecoTauGenericJetRegionProducer() override {}

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void fillDescriptionsBase(edm::ConfigurationDescriptions& descriptions, const std::string& name);

private:
  std::string moduleLabel_;

  edm::InputTag inputJets_;
  edm::InputTag pfCandSrc_;
  edm::InputTag pfCandAssocMapSrc_;

  edm::EDGetTokenT<std::vector<CandType> > pf_token;
  edm::EDGetTokenT<reco::CandidateView> Jets_token;
  edm::EDGetTokenT<JetToCandidateAssociation> pfCandAssocMap_token;

  double minJetPt_;
  double maxJetAbsEta_;
  double deltaR2_;

  int verbosity_;
};

template <class JetType, class CandType>
RecoTauGenericJetRegionProducer<JetType, CandType>::RecoTauGenericJetRegionProducer(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
  inputJets_ = cfg.getParameter<edm::InputTag>("src");
  pfCandSrc_ = cfg.getParameter<edm::InputTag>("pfCandSrc");
  pfCandAssocMapSrc_ = cfg.getParameter<edm::InputTag>("pfCandAssocMapSrc");

  pf_token = consumes<std::vector<CandType> >(pfCandSrc_);
  Jets_token = consumes<reco::CandidateView>(inputJets_);
  pfCandAssocMap_token = consumes<JetToCandidateAssociation>(pfCandAssocMapSrc_);

  double deltaR = cfg.getParameter<double>("deltaR");
  deltaR2_ = deltaR * deltaR;
  minJetPt_ = cfg.getParameter<double>("minJetPt");
  maxJetAbsEta_ = cfg.getParameter<double>("maxJetAbsEta");

  verbosity_ = cfg.getParameter<int>("verbosity");

  produces<std::vector<JetType> >("jets");
  produces<JetMatchMap>();
}

template <class JetType, class CandType>
void RecoTauGenericJetRegionProducer<JetType, CandType>::produce(edm::Event& evt, const edm::EventSetup& es) {
  if (verbosity_) {
    std::cout << "<RecoTauJetRegionProducer::produce (moduleLabel = " << moduleLabel_ << ")>:" << std::endl;
    std::cout << " inputJets = " << inputJets_ << std::endl;
    std::cout << " pfCandSrc = " << pfCandSrc_ << std::endl;
    std::cout << " pfCandAssocMapSrc_ = " << pfCandAssocMapSrc_ << std::endl;
  }

  edm::Handle<std::vector<CandType> > pfCandsHandle;
  evt.getByToken(pf_token, pfCandsHandle);

  // Build Ptrs for all the PFCandidates
  typedef edm::Ptr<CandType> CandPtr;
  std::vector<CandPtr> pfCands;
  pfCands.reserve(pfCandsHandle->size());
  for (size_t icand = 0; icand < pfCandsHandle->size(); ++icand) {
    pfCands.push_back(CandPtr(pfCandsHandle, icand));
  }

  // Get the jets
  edm::Handle<reco::CandidateView> jetView;
  evt.getByToken(Jets_token, jetView);
  // Convert to a vector of JetRefs
  edm::RefVector<std::vector<JetType> > jets = reco::tau::castView<edm::RefVector<std::vector<JetType> > >(jetView);
  size_t nJets = jets.size();

  // Get the association map matching jets to Candidates
  // (needed for reconstruction of boosted taus)
  edm::Handle<JetToCandidateAssociation> jetToPFCandMap;
  std::vector<std::unordered_set<unsigned> > fastJetToPFCandMap;
  if (!pfCandAssocMapSrc_.label().empty()) {
    evt.getByToken(pfCandAssocMap_token, jetToPFCandMap);
    fastJetToPFCandMap.resize(nJets);
    for (size_t ijet = 0; ijet < nJets; ++ijet) {
      // Get a ref to jet
      const edm::Ref<std::vector<JetType> >& jetRef = jets[ijet];
      const auto& pfCandsMappedToJet = (*jetToPFCandMap)[jetRef];
      for (const auto& pfCandMappedToJet : pfCandsMappedToJet) {
        fastJetToPFCandMap[ijet].emplace(pfCandMappedToJet.key());
      }
    }
  }

  // Get the original product, so we can match against it - otherwise the
  // indices don't match up.
  edm::ProductID originalId = jets.id();
  edm::Handle<std::vector<JetType> > originalJets;
  size_t nOriginalJets = 0;
  // We have to make sure that we have some selected jets, otherwise we don't
  // actually have a valid product ID to the original jets.
  if (nJets) {
    try {
      evt.get(originalId, originalJets);
    } catch (const cms::Exception& e) {
      edm::LogError("MissingOriginalCollection") << "Can't get the original jets that made: " << inputJets_
                                                 << " that have product ID: " << originalId << " from the event!!";
      throw e;
    }
    nOriginalJets = originalJets->size();
  }

  auto newJets = std::make_unique<std::vector<JetType> >();

  // Keep track of the indices of the current jet and the old (original) jet
  // -1 indicates no match.
  std::vector<int> matchInfo(nOriginalJets, -1);
  newJets->reserve(nJets);
  size_t nNewJets = 0;
  for (size_t ijet = 0; ijet < nJets; ++ijet) {
    // Get a ref to jet
    const edm::Ref<std::vector<JetType> >& jetRef = jets[ijet];
    if (jetRef->pt() - minJetPt_ < 1e-5)
      continue;
    if (std::abs(jetRef->eta()) - maxJetAbsEta_ > -1e-5)
      continue;
    // Make an initial copy.
    newJets->emplace_back(*jetRef);
    JetType& newJet = newJets->back();
    // Clear out all the constituents
    newJet.clearDaughters();
    // Loop over all the PFCands
    for (const auto& pfCand : pfCands) {
      bool isMappedToJet = false;
      if (jetToPFCandMap.isValid()) {
        auto temp = jetToPFCandMap->find(jetRef);
        if (temp == jetToPFCandMap->end()) {
          edm::LogWarning("WeirdCandidateMap") << "Candidate map for jet " << jetRef.key() << " is empty!";
          continue;
        }
        isMappedToJet = fastJetToPFCandMap[ijet].count(pfCand.key());
      } else {
        isMappedToJet = true;
      }
      if (reco::deltaR2(*jetRef, *pfCand) < deltaR2_ && isMappedToJet)
        newJet.addDaughter(pfCand);
    }
    if (verbosity_) {
      std::cout << "jet #" << ijet << ": Pt = " << jetRef->pt() << ", eta = " << jetRef->eta()
                << ", phi = " << jetRef->eta() << ","
                << " mass = " << jetRef->mass() << ", area = " << jetRef->jetArea() << std::endl;
      auto jetConstituents = newJet.daughterPtrVector();
      int idx = 0;
      for (const auto& jetConstituent : jetConstituents) {
        std::cout << " constituent #" << idx << ": Pt = " << jetConstituent->pt() << ", eta = " << jetConstituent->eta()
                  << ", phi = " << jetConstituent->phi() << std::endl;
        ++idx;
      }
    }
    // Match the index of the jet we just made to the index into the original
    // collection.
    //matchInfo[jetRef.key()] = ijet;
    matchInfo[jetRef.key()] = nNewJets;
    nNewJets++;
  }

  // Put our new jets into the event
  edm::OrphanHandle<std::vector<JetType> > newJetsInEvent = evt.put(std::move(newJets), "jets");

  // Create a matching between original jets -> extra collection
  auto matching = (nJets != 0)
                      ? std::make_unique<JetMatchMap>(
                            edm::makeRefToBaseProdFrom(edm::RefToBase<reco::Jet>(jets[0]), evt), newJetsInEvent)
                      : std::make_unique<JetMatchMap>();
  for (size_t ijet = 0; ijet < nJets; ++ijet) {
    matching->insert(edm::RefToBase<reco::Jet>(jets[ijet]),
                     edm::RefToBase<reco::Jet>(edm::Ref<std::vector<JetType> >(newJetsInEvent, matchInfo[ijet])));
  }
  evt.put(std::move(matching));
}

template <class JetType, class CandType>
void RecoTauGenericJetRegionProducer<JetType, CandType>::fillDescriptionsBase(
    edm::ConfigurationDescriptions& descriptions, const std::string& name) {
  // RecoTauGenericJetRegionProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("ak4PFJets"));
  desc.add<double>("deltaR", 0.8);
  desc.add<edm::InputTag>("pfCandAssocMapSrc", edm::InputTag(""));
  desc.add<int>("verbosity", 0);
  desc.add<double>("maxJetAbsEta", 2.5);
  desc.add<double>("minJetPt", 14.0);
  desc.add<edm::InputTag>("pfCandSrc", edm::InputTag("particleFlow"));
  descriptions.add(name, desc);
}

template <>
void RecoTauGenericJetRegionProducer<reco::PFJet, reco::PFCandidate>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // RecoTauGenericJetRegionProducer
  RecoTauGenericJetRegionProducer::fillDescriptionsBase(descriptions, "RecoTauJetRegionProducer");
}

template <>
void RecoTauGenericJetRegionProducer<pat::Jet, pat::PackedCandidate>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // RecoTauGenericJetRegionProducer
  RecoTauGenericJetRegionProducer::fillDescriptionsBase(descriptions, "RecoTauPatJetRegionProducer");
}

typedef RecoTauGenericJetRegionProducer<reco::PFJet, reco::PFCandidate> RecoTauJetRegionProducer;
typedef RecoTauGenericJetRegionProducer<pat::Jet, pat::PackedCandidate> RecoTauPatJetRegionProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauJetRegionProducer);
DEFINE_FWK_MODULE(RecoTauPatJetRegionProducer);
