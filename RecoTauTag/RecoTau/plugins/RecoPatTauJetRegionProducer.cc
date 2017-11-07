/*
 * RecoTauPatJetRegionProducer
 *
 * Given a set of PFJets, make new jets with the same p4 but collect all the
 * PFCandidates from a cone of a given size into the constituents.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include <boost/bind.hpp>

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "RecoTauTag/RecoTau/interface/ConeTools.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <iostream>

class RecoTauPatJetRegionProducer : public edm::stream::EDProducer<> 
{
 public:
  typedef edm::Association<pat::JetCollection> PatJetMatchMap;
  typedef edm::AssociationMap<edm::OneToMany<std::vector<pat::Jet>, std::vector<pat::PackedCandidate>, unsigned int> > JetToPackedCandidateAssociation;
  explicit RecoTauPatJetRegionProducer(const edm::ParameterSet& pset);
  ~RecoTauPatJetRegionProducer() {}

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

 private:
  std::string moduleLabel_;

  edm::InputTag inputJets_;
  edm::InputTag pfCandSrc_;
  edm::InputTag pfCandAssocMapSrc_;

  edm::EDGetTokenT<pat::PackedCandidateCollection> pf_token;
  edm::EDGetTokenT<reco::CandidateView> Jets_token;
  edm::EDGetTokenT<JetToPackedCandidateAssociation> pfCandAssocMap_token;

  double minJetPt_;
  double maxJetAbsEta_;
  double deltaR2_;

  int verbosity_;
};

RecoTauPatJetRegionProducer::RecoTauPatJetRegionProducer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  inputJets_ = cfg.getParameter<edm::InputTag>("src");
  pfCandSrc_ = cfg.getParameter<edm::InputTag>("pfCandSrc");
  pfCandAssocMapSrc_ = cfg.getParameter<edm::InputTag>("pfCandAssocMapSrc");

  pf_token = consumes<pat::PackedCandidateCollection>(pfCandSrc_); 
  Jets_token = consumes<reco::CandidateView>(inputJets_);
  pfCandAssocMap_token =  consumes<JetToPackedCandidateAssociation>(pfCandAssocMapSrc_);
  
  double deltaR = cfg.getParameter<double>("deltaR"); 
  deltaR2_ = deltaR*deltaR;
  minJetPt_ = ( cfg.exists("minJetPt") ) ? cfg.getParameter<double>("minJetPt") : -1.0;
  maxJetAbsEta_ = ( cfg.exists("maxJetAbsEta") ) ? cfg.getParameter<double>("maxJetAbsEta") : 99.0;
  
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
  
  produces<pat::JetCollection>("jets");
  produces<PatJetMatchMap>();
}

void RecoTauPatJetRegionProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  if ( verbosity_ ) {
    std::cout << "<RecoTauPatJetRegionProducer::produce (moduleLabel = " << moduleLabel_ << ")>:" << std::endl;
    std::cout << " inputJets = " << inputJets_ << std::endl;
    std::cout << " pfCandSrc = " << pfCandSrc_ << std::endl;
    std::cout << " pfCandAssocMapSrc_ = " << pfCandAssocMapSrc_ << std::endl;
  }

  edm::Handle<pat::PackedCandidateCollection> pfCandsHandle;
  evt.getByToken(pf_token, pfCandsHandle);

  // Build Ptrs for all the PFCandidates
  typedef edm::Ptr<pat::PackedCandidate> PackedCandPtr;
  std::vector<PackedCandPtr> pfCands;
  pfCands.reserve(pfCandsHandle->size());
  for ( size_t icand = 0; icand < pfCandsHandle->size(); ++icand ) {
    pfCands.push_back(PackedCandPtr(pfCandsHandle, icand));
  }

  // Get the jets
  edm::Handle<reco::CandidateView> jetView;
  evt.getByToken(Jets_token, jetView);
  // Convert to a vector of PFJetRefs
  pat::JetRefVector jets = reco::tau::castView<pat::JetRefVector>(jetView);
  size_t nJets = jets.size();

  // Get the association map matching jets to PFCandidates
  // (needed for recinstruction of boosted taus)
  edm::Handle<JetToPackedCandidateAssociation> jetToPFCandMap;
  std::vector<std::unordered_set<unsigned> > fastJetToPFCandMap;
  if ( pfCandAssocMapSrc_.label() != "" ) {
    evt.getByToken(pfCandAssocMap_token, jetToPFCandMap);
    fastJetToPFCandMap.resize(nJets);
    for ( size_t ijet = 0; ijet < nJets; ++ijet ) {
      // Get a ref to jet
      const pat::JetRef& jetRef = jets[ijet];
      const auto& pfCandsMappedToJet = (*jetToPFCandMap)[jetRef];
      for ( const auto& pfCandMappedToJet : pfCandsMappedToJet ) {
	fastJetToPFCandMap[ijet].emplace(pfCandMappedToJet.key());
      }
    }
  }

  // Get the original product, so we can match against it - otherwise the
  // indices don't match up.
  edm::ProductID originalId = jets.id();
  edm::Handle<pat::JetCollection> originalJets;
  size_t nOriginalJets = 0;
  // We have to make sure that we have some selected jets, otherwise we don't
  // actually have a valid product ID to the original jets.
  if ( nJets ) {
    try {
      evt.get(originalId, originalJets);
    } catch(const cms::Exception &e) {
      edm::LogError("MissingOriginalCollection")
        << "Can't get the original jets that made: " << inputJets_
        << " that have product ID: " << originalId
        << " from the event!!";
      throw e;
    }
    nOriginalJets = originalJets->size();
  }

  auto newJets = std::make_unique<pat::JetCollection>();

  // Keep track of the indices of the current jet and the old (original) jet
  // -1 indicates no match.
  std::vector<int> matchInfo(nOriginalJets, -1);
  newJets->reserve(nJets);
  size_t nNewJets = 0;
  for ( size_t ijet = 0; ijet < nJets; ++ijet ) {
    // Get a ref to jet
    const pat::JetRef& jetRef = jets[ijet];
    if(jetRef->pt() - minJetPt_ < 1e-5) continue;
    if(std::abs(jetRef->eta()) - maxJetAbsEta_ > -1e-5) continue;
    // Make an initial copy.
    newJets->emplace_back(*jetRef);
    pat::Jet& newJet = newJets->back();
    // Clear out all the constituents
    newJet.clearDaughters();
    // Loop over all the PFCands
    for ( const auto& pfCand : pfCands ) {
      bool isMappedToJet = false;
      if ( jetToPFCandMap.isValid() ) {
	auto temp = jetToPFCandMap->find(jetRef);
	if( temp == jetToPFCandMap->end() ) {
	  edm::LogWarning("WeirdCandidateMap") << "Candidate map for jet " << jetRef.key() << " is empty!";
	  continue;
	}
	isMappedToJet = fastJetToPFCandMap[ijet].count(pfCand.key());
      } else {
	isMappedToJet = true;
      }
      if ( reco::deltaR2(*jetRef, *pfCand) < deltaR2_ && isMappedToJet ) newJet.addDaughter(pfCand);
    }
    if ( verbosity_ ) {
      std::cout << "jet #" << ijet << ": Pt = " << jetRef->pt() << ", eta = " << jetRef->eta() << ", phi = " << jetRef->eta() << ","
		<< " mass = " << jetRef->mass() << ", area = " << jetRef->jetArea() << std::endl;
      auto jetConstituents = newJet.daughterPtrVector();
      int idx = 0;
      for ( const auto& jetConstituent : jetConstituents) {
	std::cout << " constituent #" << idx << ": Pt = " << jetConstituent->pt() << ", eta = " << jetConstituent->eta() << ", phi = " << jetConstituent->phi() << std::endl;
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
  edm::OrphanHandle<pat::JetCollection> newJetsInEvent = evt.put(std::move(newJets), "jets");

  // Create a matching between original jets -> extra collection
  auto matching = std::make_unique<PatJetMatchMap>(newJetsInEvent);
  if ( nJets ) {
    PatJetMatchMap::Filler filler(*matching);
    filler.insert(originalJets, matchInfo.begin(), matchInfo.end());
    filler.fill();
  }
  evt.put(std::move(matching));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPatJetRegionProducer);
