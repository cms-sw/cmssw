#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>
#include <iostream>

class GenVisTauProducer : public edm::global::EDProducer<> 
{
 public:
  GenVisTauProducer(const edm::ParameterSet& params) 
    : src_(consumes<reco::GenJetCollection>(params.getParameter<edm::InputTag>("src")))
    , srcGenParticles_(consumes<reco::GenParticleCollection>(params.getParameter<edm::InputTag>("srcGenParticles")))
  {
    produces<reco::GenParticleCollection>();
  }
  
  ~GenVisTauProducer() override {}

  void produce(edm::StreamID id, edm::Event& evt, const edm::EventSetup& es) const override 
  {
    edm::Handle<reco::GenJetCollection> genTauJets;
    evt.getByToken(src_, genTauJets);

    edm::Handle<reco::GenParticleCollection> genParticles;
    evt.getByToken(srcGenParticles_, genParticles);
    size_t numGenParticles = genParticles->size();

    auto genVisTaus = std::make_unique<reco::GenParticleCollection>();

    for (const auto & genTauJet : *genTauJets) {
      std::string decayMode_string = JetMCTagUtils::genTauDecayMode(genTauJet);
      // CV: store hadronic tau decays only
      if ( decayMode_string == "electron" || decayMode_string == "muon" ) continue;
      int decayMode = reco::PFTau::kNull;
      if      ( decayMode_string == "oneProng0Pi0"   ) decayMode = reco::PFTau::kOneProng0PiZero;
      else if ( decayMode_string == "oneProng1Pi0"   ) decayMode = reco::PFTau::kOneProng1PiZero;
      else if ( decayMode_string == "oneProng2Pi0"   ) decayMode = reco::PFTau::kOneProng2PiZero;
      else if ( decayMode_string == "threeProng0Pi0" ) decayMode = reco::PFTau::kThreeProng0PiZero;
      else if ( decayMode_string == "threeProng1Pi0" ) decayMode = reco::PFTau::kThreeProng1PiZero;
      else                                             decayMode = reco::PFTau::kRareDecayMode;

      int pdgId = ( genTauJet.charge() > 0 ) ? -15 : +15;

      // CV: store decayMode in status flag of GenParticle object
      reco::GenParticle genVisTau(genTauJet.charge(), genTauJet.p4(), genTauJet.vertex(), pdgId, decayMode, true);

      // CV: find tau lepton "mother" particle
      for ( size_t idxGenParticle = 0; idxGenParticle < numGenParticles; ++idxGenParticle ) {
	const reco::GenParticle & genTau = (*genParticles)[idxGenParticle];
	if ( abs(genTau.pdgId()) == 15 && genTau.status() == 2 ) {
	  reco::Candidate::LorentzVector daughterVisP4;
          for (const reco::GenParticleRef & daughter : genTau.daughterRefVector()) {
	    int abs_pdgId = abs(daughter->pdgId());
	    // CV: skip neutrinos
	    if ( abs_pdgId == 12 || abs_pdgId == 14 || abs_pdgId == 16 ) continue;
	    daughterVisP4 += daughter->p4();
	  }
	  double dR2 = deltaR2(daughterVisP4, genVisTau);
	  if ( dR2 < 1.e-4 ) {	  
	    genVisTau.addMother(reco::GenParticleRef(genParticles, idxGenParticle));
	    break;
	  }
	}
      }

      genVisTaus->push_back(genVisTau);
    }

    evt.put(std::move(genVisTaus));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src")->setComment("collection of visible gen taus (as reco::GenJetCollection)");
      desc.add<edm::InputTag>("srcGenParticles")->setComment("collections of gen particles");
      descriptions.add("genVisTaus", desc);
  }


 private:
  const edm::EDGetTokenT<reco::GenJetCollection> src_;
  const edm::EDGetTokenT<reco::GenParticleCollection> srcGenParticles_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenVisTauProducer);

