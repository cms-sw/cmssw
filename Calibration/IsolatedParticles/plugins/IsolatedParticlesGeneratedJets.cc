// -*- C++ -*-
//
// Package:    IsolatedParticles
// Class:      IsolatedParticlesGeneratedJets
//
/**\class IsolatedParticlesGeneratedJets IsolatedParticlesGeneratedJets.cc Calibration/IsolatedParticles/plugins/IsolatedParticlesGeneratedJets.cc

 Description: Studies properties of jets at generator level in context of
              isolated particles

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seema Sharma
//         Created:  Thu Mar  4 18:52:02 CST 2010
//
//

// user include files
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "RecoJets/JetProducers/interface/JetMatchingTools.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

class IsolatedParticlesGeneratedJets : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit IsolatedParticlesGeneratedJets(const edm::ParameterSet &);
  ~IsolatedParticlesGeneratedJets() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override {}

  void bookHistograms();
  void clearTreeVectors();

  const bool debug_;
  TTree *tree_;

  const edm::EDGetTokenT<reco::GenJetCollection> tok_jets_;
  const edm::EDGetTokenT<reco::GenParticleCollection> tok_parts_;

  std::vector<int> *t_gjetN;
  std::vector<double> *t_gjetE, *t_gjetPt, *t_gjetEta, *t_gjetPhi;
  std::vector<std::vector<double> > *t_jetTrkP;
  std::vector<std::vector<double> > *t_jetTrkPt;
  std::vector<std::vector<double> > *t_jetTrkEta;
  std::vector<std::vector<double> > *t_jetTrkPhi;
  std::vector<std::vector<double> > *t_jetTrkPdg;
  std::vector<std::vector<double> > *t_jetTrkCharge;
};

IsolatedParticlesGeneratedJets::IsolatedParticlesGeneratedJets(const edm::ParameterSet &iConfig)
    : debug_(iConfig.getUntrackedParameter<bool>("Debug", false)),
      tok_jets_(consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("JetSource"))),
      tok_parts_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("ParticleSource"))) {
  usesResource(TFileService::kSharedResource);
}

void IsolatedParticlesGeneratedJets::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("Debug", true);
  desc.add<edm::InputTag>("JetSource", edm::InputTag("ak5GenJets"));
  desc.add<edm::InputTag>("ParticleSource", edm::InputTag("genParticles"));
  descriptions.add("isolatedParticlesGeneratedJets", desc);
}

void IsolatedParticlesGeneratedJets::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  //using namespace edm;
  clearTreeVectors();

  //=== genJet information
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(tok_jets_, genJets);

  //=== genJet information
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(tok_parts_, genParticles);

  JetMatchingTools jetMatching(iEvent, consumesCollector());
  std::vector<std::vector<const reco::GenParticle *> > genJetConstituents(genJets->size());

  int njets = 0;
  for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
    const reco::GenJet &genJet = (*genJets)[iGenJet];

    double genJetE = genJet.energy();
    double genJetPt = genJet.pt();
    double genJetEta = genJet.eta();
    double genJetPhi = genJet.phi();

    if (genJetPt > 30.0 && std::abs(genJetEta) < 3.0) {
      njets++;

      std::vector<const reco::GenParticle *> genJetConstituents = jetMatching.getGenParticles((*genJets)[iGenJet]);
      std::vector<double> v_trkP, v_trkPt, v_trkEta, v_trkPhi, v_trkPdg, v_trkCharge;

      if (debug_)
        edm::LogVerbatim("IsoTrack") << "Jet(pt,Eta,Phi) " << genJetPt << " " << genJetEta << " " << genJetPhi;
      for (unsigned int ic = 0; ic < genJetConstituents.size(); ic++) {
        if (debug_)
          edm::LogVerbatim("IsoTrack") << "p,pt,eta,phi " << genJetConstituents[ic]->p() << " "
                                       << genJetConstituents[ic]->pt() << " " << genJetConstituents[ic]->eta() << " "
                                       << genJetConstituents[ic]->phi();

        v_trkP.push_back(genJetConstituents[ic]->p());
        v_trkPt.push_back(genJetConstituents[ic]->pt());
        v_trkEta.push_back(genJetConstituents[ic]->eta());
        v_trkPhi.push_back(genJetConstituents[ic]->phi());
        v_trkPdg.push_back(genJetConstituents[ic]->pdgId());
        v_trkCharge.push_back(genJetConstituents[ic]->charge());

      }  //loop over genjet constituents

      t_gjetE->push_back(genJetE);
      t_gjetPt->push_back(genJetPt);
      t_gjetEta->push_back(genJetEta);
      t_gjetPhi->push_back(genJetPhi);

      t_jetTrkP->push_back(v_trkP);
      t_jetTrkPt->push_back(v_trkPt);
      t_jetTrkEta->push_back(v_trkEta);
      t_jetTrkPhi->push_back(v_trkPhi);
      t_jetTrkPdg->push_back(v_trkPdg);
      t_jetTrkCharge->push_back(v_trkCharge);

    }  // if jetPt>30

  }  //loop over genjets

  t_gjetN->push_back(njets);

  if (debug_) {
    unsigned int indx = 0;
    reco::GenParticleCollection::const_iterator ig = genParticles->begin();
    for (; ig != genParticles->end(); ++ig, ++indx) {
      edm::LogVerbatim("IsoTrack") << "Track " << indx << " Status " << ig->status() << " charge " << ig->charge()
                                   << " pdgId " << ig->pdgId() << " mass " << ig->mass() << " P " << ig->momentum()
                                   << " E " << ig->energy() << " Origin " << ig->vertex();
    }
  }

  tree_->Fill();
}

void IsolatedParticlesGeneratedJets::beginJob() { bookHistograms(); }

void IsolatedParticlesGeneratedJets::clearTreeVectors() {
  t_gjetN->clear();
  t_gjetE->clear();
  t_gjetPt->clear();
  t_gjetEta->clear();
  t_gjetPhi->clear();

  t_jetTrkP->clear();
  t_jetTrkPt->clear();
  t_jetTrkEta->clear();
  t_jetTrkPhi->clear();
  t_jetTrkPdg->clear();
  t_jetTrkCharge->clear();
}

void IsolatedParticlesGeneratedJets::bookHistograms() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("tree", "tree");

  t_gjetN = new std::vector<int>();
  t_gjetE = new std::vector<double>();
  t_gjetPt = new std::vector<double>();
  t_gjetEta = new std::vector<double>();
  t_gjetPhi = new std::vector<double>();

  t_jetTrkP = new std::vector<std::vector<double> >();
  t_jetTrkPt = new std::vector<std::vector<double> >();
  t_jetTrkEta = new std::vector<std::vector<double> >();
  t_jetTrkPhi = new std::vector<std::vector<double> >();
  t_jetTrkPdg = new std::vector<std::vector<double> >();
  t_jetTrkCharge = new std::vector<std::vector<double> >();

  tree_->Branch("t_gjetN", "std::vector<int>", &t_gjetN);
  tree_->Branch("t_gjetE", "std::vector<double>", &t_gjetE);
  tree_->Branch("t_gjetPt", "std::vector<double>", &t_gjetPt);
  tree_->Branch("t_gjetEta", "std::vector<double>", &t_gjetEta);
  tree_->Branch("t_gjetPhi", "std::vector<double>", &t_gjetPhi);

  tree_->Branch("t_jetTrkP", "std::vector<vector<double> >", &t_jetTrkP);
  tree_->Branch("t_jetTrkPt", "std::vector<vector<double> >", &t_jetTrkPt);
  tree_->Branch("t_jetTrkEta", "std::vector<vector<double> >", &t_jetTrkEta);
  tree_->Branch("t_jetTrkPhi", "std::vector<vector<double> >", &t_jetTrkPhi);
  tree_->Branch("t_jetTrkPdg", "std::vector<vector<double> >", &t_jetTrkPdg);
  tree_->Branch("t_jetTrkCharge", "std::vector<vector<double> >", &t_jetTrkCharge);
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedParticlesGeneratedJets);
