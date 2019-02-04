
/* =============================================================================
 *       Filename:  BoostedTauSeedsProducer.cc
 *
 *    Description:  Take the two subjets found by CMSBoostedTauSeedingAlgorithm
 *                  and add the data-formats for 
 *                 o seeding the reconstruction of hadronic taus (PFJets, collection of PFCandidates not within subjet)
 *                 o computation of electron and muon isolation Pt-sums, excluding the particles in the other subjet (collection of PFCandidates within subjet, used to define "Vetos")
 *
 *        Created:  10/22/2013 16:05:00
 *
 *         Authors:  Christian Veelken (LLR)
 *
 * =============================================================================
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TMath.h>

#include <string>
#include <iostream>
#include <iomanip>

class BoostedTauSeedsProducer : public edm::stream::EDProducer<>
{
 public:
  explicit BoostedTauSeedsProducer(const edm::ParameterSet&);
  ~BoostedTauSeedsProducer() override {}
  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  typedef edm::AssociationMap<edm::OneToMany<std::vector<reco::PFJet>, std::vector<reco::PFCandidate>, unsigned int> > JetToPFCandidateAssociation;

  std::string moduleLabel_;

  typedef edm::View<reco::Jet> JetView;
  edm::EDGetTokenT<JetView> srcSubjets_;
  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidates_;

  int verbosity_;
};

BoostedTauSeedsProducer::BoostedTauSeedsProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  srcSubjets_ = consumes<JetView>(cfg.getParameter<edm::InputTag>("subjetSrc"));
  srcPFCandidates_ = consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("pfCandidateSrc"));

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<reco::PFJetCollection>();
  produces<JetToPFCandidateAssociation>("pfCandAssocMapForIsolation");
  //produces<JetToPFCandidateAssociation>("pfCandAssocMapForIsoDepositVetos");
}

namespace
{
  typedef std::vector<std::unordered_set<uint32_t> > JetToConstitMap;
  
  reco::PFJet convertToPFJet(const reco::Jet& jet, const reco::Jet::Constituents& jetConstituents)
  {    
    // CV: code for filling pfJetSpecific objects taken from
    //        RecoParticleFlow/PFRootEvent/src/JetMaker.cc
    double chargedHadronEnergy = 0.;
    double neutralHadronEnergy = 0.;
    double chargedEmEnergy     = 0.;
    double neutralEmEnergy     = 0.;
    double chargedMuEnergy     = 0.;
    int    chargedMultiplicity = 0;
    int    neutralMultiplicity = 0;
    int    muonMultiplicity    = 0;
    for ( reco::Jet::Constituents::const_iterator jetConstituent = jetConstituents.begin();
	  jetConstituent != jetConstituents.end(); ++jetConstituent ) {
      const reco::PFCandidate* pfCandidate = dynamic_cast<const reco::PFCandidate*>(jetConstituent->get());
      if ( pfCandidate ) {
	switch ( pfCandidate->particleId() ) {
	case reco::PFCandidate::h :          // charged hadron
	  chargedHadronEnergy += pfCandidate->energy();
	  ++chargedMultiplicity;
	  break;           
	case reco::PFCandidate::e :          // electron 
	  chargedEmEnergy += pfCandidate->energy(); 
	  ++chargedMultiplicity;
	  break;
	case reco::PFCandidate::mu :         // muon
	  chargedMuEnergy += pfCandidate->energy();
	  ++chargedMultiplicity;
	  ++muonMultiplicity;
	  break;
	case reco::PFCandidate::gamma :     // photon
	case reco::PFCandidate::egamma_HF : // electromagnetic in HF
	  neutralEmEnergy += pfCandidate->energy();
	  ++neutralMultiplicity;
	  break;
	case reco::PFCandidate::h0 :        // neutral hadron
	case reco::PFCandidate::h_HF :      // hadron in HF
	  neutralHadronEnergy += pfCandidate->energy();
	  ++neutralMultiplicity;
	  break;
	default:
	  edm::LogWarning("convertToPFJet") 
	    << "PFCandidate: Pt = " << pfCandidate->pt() << ", eta = " << pfCandidate->eta() << ", phi = " << pfCandidate->phi() 
	    << " has invalid particleID = " << pfCandidate->particleId() << " !!" << std::endl;
	  break;
	}
      } else {
	edm::LogWarning("convertToPFJet") 
	  << "Jet constituent: Pt = " << pfCandidate->pt() << ", eta = " << pfCandidate->eta() << ", phi = " << pfCandidate->phi() 
	  << " is not of type PFCandidate !!" << std::endl;
      }
    }
    reco::PFJet::Specific pfJetSpecific;
    pfJetSpecific.mChargedHadronEnergy = chargedHadronEnergy;
    pfJetSpecific.mNeutralHadronEnergy = neutralHadronEnergy;
    pfJetSpecific.mChargedEmEnergy     = chargedEmEnergy;
    pfJetSpecific.mChargedMuEnergy     = chargedMuEnergy;
    pfJetSpecific.mNeutralEmEnergy     = neutralEmEnergy;
    pfJetSpecific.mChargedMultiplicity = chargedMultiplicity;
    pfJetSpecific.mNeutralMultiplicity = neutralMultiplicity;
    pfJetSpecific.mMuonMultiplicity    = muonMultiplicity;

    reco::PFJet pfJet(jet.p4(), jet.vertex(), pfJetSpecific, jetConstituents);
    pfJet.setJetArea(jet.jetArea());

    return pfJet;
  }

  void getJetConstituents(const reco::Jet& jet, reco::Jet::Constituents& jet_and_subjetConstituents)
  {
    reco::Jet::Constituents jetConstituents = jet.getJetConstituents();
    for ( reco::Jet::Constituents::const_iterator jetConstituent = jetConstituents.begin();
	  jetConstituent != jetConstituents.end(); ++jetConstituent ) {
      const reco::Jet* subjet = dynamic_cast<const reco::Jet*>(jetConstituent->get());
      if ( subjet ) {
	getJetConstituents(*subjet, jet_and_subjetConstituents);
      } else { 
	jet_and_subjetConstituents.push_back(*jetConstituent);
      }
    }
  }

  std::vector<reco::PFCandidateRef> getPFCandidates_exclJetConstituents(const reco::Jet& jet, const edm::Handle<reco::PFCandidateCollection>& pfCandidates, const JetToConstitMap::value_type& constitmap, const reco::Jet::Constituents& jetConstituents, double /*dRmatch*/, bool invert)
  { 
    //const double dRmatch2 = dRmatch*dRmatch; // comment out for now in case someone needs a dR-based search again
    auto const & collection_cand = (*pfCandidates);
    std::vector<reco::PFCandidateRef> pfCandidates_exclJetConstituents;
    size_t numPFCandidates = pfCandidates->size();
    for ( size_t pfCandidateIdx = 0; pfCandidateIdx < numPFCandidates; ++pfCandidateIdx ) {
      if(!(deltaR2(collection_cand[pfCandidateIdx], jet)<1.0)) continue;
      bool isJetConstituent = constitmap.count(pfCandidateIdx);      
      if ( !(isJetConstituent^invert) ) {
	reco::PFCandidateRef pfCandidate(pfCandidates, pfCandidateIdx);
	pfCandidates_exclJetConstituents.push_back(pfCandidate);
      }
    }
    return pfCandidates_exclJetConstituents;
  }

  void printJetConstituents(const std::string& label, const reco::Jet::Constituents& jetConstituents)
  {
    std::cout << "#" << label << " = " << jetConstituents.size() << ":" << std::endl;
    int idx = 0;
    for ( reco::Jet::Constituents::const_iterator jetConstituent = jetConstituents.begin();
	  jetConstituent != jetConstituents.end(); ++jetConstituent ) {
      std::cout << " jetConstituent #" << idx << ": Pt = " << (*jetConstituent)->pt() << ", eta = " << (*jetConstituent)->eta() << ", phi = " << (*jetConstituent)->phi() << std::endl;
      ++idx;
    }
  }
}

void BoostedTauSeedsProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ >= 1 ) {
    std::cout << "<BoostedTauSeedsProducer::produce (moduleLabel = " << moduleLabel_ << ")>:" << std::endl;
  }

  edm::Handle<JetView> subjets;
  evt.getByToken(srcSubjets_, subjets);
  if ( verbosity_ >= 1 ) {
    std::cout << "#subjets = " << subjets->size() << std::endl;
  }
  assert((subjets->size() % 2) == 0); // CV: ensure that subjets come in pairs
  
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByToken(srcPFCandidates_, pfCandidates);
  if ( verbosity_ >= 1 ) {
    std::cout << "#pfCandidates = " << pfCandidates->size() << std::endl;
  }
  
  auto selectedSubjets = std::make_unique<reco::PFJetCollection>();
  edm::RefProd<reco::PFJetCollection> selectedSubjetRefProd = evt.getRefBeforePut<reco::PFJetCollection>();

  auto selectedSubjetPFCandidateAssociationForIsolation = std::make_unique<JetToPFCandidateAssociation>(&evt.productGetter());
  //auto selectedSubjetPFCandidateAssociationForIsoDepositVetos = std::make_unique<JetToPFCandidateAssociation>(&evt.productGetter());

  // cache for jet->pfcandidate
  JetToConstitMap constitmap(subjets->size());

  // fill constituents map
  const auto& thesubjets = *subjets;
  for( unsigned i = 0; i < thesubjets.size(); ++i ) {
    for ( unsigned j = 0; j < thesubjets[i].numberOfDaughters(); ++j ) {
      constitmap[i].emplace(thesubjets[i].daughterPtr(j).key());
    }
  }

  for ( size_t idx = 0; idx < (subjets->size() / 2); ++idx ) {
    const reco::Jet* subjet1 = &subjets->at(2*idx);
    const reco::Jet* subjet2 = &subjets->at(2*idx + 1);
    assert(subjet1 && subjet2);
    if ( verbosity_ >= 1 ) {
      std::cout << "processing jet #" << idx << ":" << std::endl;
      std::cout << " subjet1: Pt = " << subjet1->pt() << ", eta = " << subjet1->eta() << ", phi = " << subjet1->phi() << ", mass = " << subjet1->mass() 
		<< " (#constituents = " << subjet1->nConstituents() << ", area = " << subjet1->jetArea() << ")" << std::endl;
      std::cout << " subjet2: Pt = " << subjet2->pt() << ", eta = " << subjet2->eta() << ", phi = " << subjet2->phi() << ", mass = " << subjet2->mass() 
		<< " (#constituents = " << subjet2->nConstituents() << ", area = " << subjet2->jetArea() << ")" << std::endl;
    }

    if ( !(subjet1->nConstituents() >= 1 && subjet1->pt() > 1. &&
	   subjet2->nConstituents() >= 1 && subjet2->pt() > 1.) ) continue; // CV: skip pathological cases

    // find PFCandidate constituents of each subjet
    reco::Jet::Constituents subjetConstituents1;
    getJetConstituents(*subjet1, subjetConstituents1);    
    reco::Jet::Constituents subjetConstituents2;
    getJetConstituents(*subjet2, subjetConstituents2);
    if ( verbosity_ >= 1 ) {
      printJetConstituents("subjetConstituents1", subjetConstituents1);
      printJetConstituents("subjetConstituents2", subjetConstituents2);
    }

    selectedSubjets->push_back(convertToPFJet(*subjet1, subjetConstituents1));
    edm::Ref<reco::PFJetCollection> subjetRef1(selectedSubjetRefProd, selectedSubjets->size() - 1);
    selectedSubjets->push_back(convertToPFJet(*subjet2, subjetConstituents2));
    edm::Ref<reco::PFJetCollection> subjetRef2(selectedSubjetRefProd, selectedSubjets->size() - 1);
        
    // find all PFCandidates that are not constituents of the **other** subjet
    std::vector<reco::PFCandidateRef> pfCandidatesNotInSubjet1 = getPFCandidates_exclJetConstituents(*subjet1, pfCandidates, constitmap[2*idx], subjetConstituents2, 1.e-4, false);
    std::vector<reco::PFCandidateRef> pfCandidatesNotInSubjet2 = getPFCandidates_exclJetConstituents(*subjet2, pfCandidates, constitmap[2*idx+1], subjetConstituents1, 1.e-4, false);
    if ( verbosity_ >= 1 ) {
      std::cout << "#pfCandidatesNotInSubjet1 = " << pfCandidatesNotInSubjet1.size() << std::endl;
      std::cout << "#pfCandidatesNotInSubjet2 = " << pfCandidatesNotInSubjet2.size() << std::endl;
    }

    // build JetToPFCandidateAssociation 
    // (key = subjet, value = collection of PFCandidates that are not constituents of subjet)
    for(auto const& pfCandidate : pfCandidatesNotInSubjet1 ) {
      selectedSubjetPFCandidateAssociationForIsolation->insert(subjetRef1, pfCandidate);
    }
    for(auto const& pfCandidate : pfCandidatesNotInSubjet2 ) {
      selectedSubjetPFCandidateAssociationForIsolation->insert(subjetRef2, pfCandidate);
    }
  }

  evt.put(std::move(selectedSubjets));
  evt.put(std::move(selectedSubjetPFCandidateAssociationForIsolation), "pfCandAssocMapForIsolation");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BoostedTauSeedsProducer);
