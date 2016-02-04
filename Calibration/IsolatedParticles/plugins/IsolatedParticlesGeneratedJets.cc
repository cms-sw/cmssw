// -*- C++ -*-
//
// Package:    IsolatedParticlesGeneratedJets
// Class:      IsolatedParticlesGeneratedJets
// 
/**\class IsolatedParticlesGeneratedJets IsolatedParticlesGeneratedJets.cc Calibration/IsolatedParticles/plugins/IsolatedParticlesGeneratedJets.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seema Sharma
//         Created:  Thu Mar  4 18:52:02 CST 2010
//
//

#include "Calibration/IsolatedParticles/plugins/IsolatedParticlesGeneratedJets.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

IsolatedParticlesGeneratedJets::IsolatedParticlesGeneratedJets(const edm::ParameterSet& iConfig) {

  debug   = iConfig.getUntrackedParameter<bool>  ("Debug", false);
  jetSrc  = iConfig.getParameter<edm::InputTag>("JetSource");
  partSrc = iConfig.getParameter<edm::InputTag>("ParticleSource");
}


IsolatedParticlesGeneratedJets::~IsolatedParticlesGeneratedJets() {

}

void IsolatedParticlesGeneratedJets::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  //using namespace edm;
  clearTreeVectors();

  //=== genJet information
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByLabel(jetSrc, genJets);

  //=== genJet information
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByLabel(partSrc, genParticles);

  JetMatchingTools jetMatching (iEvent);
  std::vector <std::vector <const reco::GenParticle*> > genJetConstituents (genJets->size());

  int njets = 0;
  for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
    const reco::GenJet& genJet = (*genJets) [iGenJet];

    double genJetE   = genJet.energy();
    double genJetPt  = genJet.pt();
    double genJetEta = genJet.eta();
    double genJetPhi = genJet.phi();

    if( genJetPt> 30.0 && std::abs(genJetEta)<3.0 ) {

      njets++;

      std::vector <const reco::GenParticle*> genJetConstituents = jetMatching.getGenParticles ((*genJets) [iGenJet]);

      std::vector<double> v_trkP, v_trkPt, v_trkEta, v_trkPhi, v_trkPdg, v_trkCharge;
    
      if(debug) std::cout<<"Jet(pt,Eta,Phi) "<<genJetPt<<" "<<genJetEta<<" "<<genJetPhi <<std::endl;
      for(unsigned int ic=0; ic<genJetConstituents.size(); ic++) {

	if(debug) {
	  std::cout << "p,pt,eta,phi "<<genJetConstituents[ic]->p()<<" "<<genJetConstituents[ic]->pt()
		    <<" "<<genJetConstituents[ic]->eta()<<" "<<genJetConstituents[ic]->phi()
		    <<std::endl;
	}

	v_trkP.push_back(genJetConstituents[ic]->p());
	v_trkPt.push_back(genJetConstituents[ic]->pt());
	v_trkEta.push_back(genJetConstituents[ic]->eta());
	v_trkPhi.push_back(genJetConstituents[ic]->phi());
	v_trkPdg.push_back(genJetConstituents[ic]->pdgId());
	v_trkCharge.push_back(genJetConstituents[ic]->charge());

      } //loop over genjet constituents

      t_gjetE   ->push_back(genJetE  );
      t_gjetPt  ->push_back(genJetPt );
      t_gjetEta ->push_back(genJetEta);
      t_gjetPhi ->push_back(genJetPhi);
      
      t_jetTrkP   ->push_back(v_trkP  );
      t_jetTrkPt  ->push_back(v_trkPt );
      t_jetTrkEta ->push_back(v_trkEta);
      t_jetTrkPhi ->push_back(v_trkPhi);
      t_jetTrkPdg ->push_back(v_trkPdg);
      t_jetTrkCharge ->push_back(v_trkCharge);

    } // if jetPt>30

  } //loop over genjets

  t_gjetN->push_back(njets);

  unsigned int indx = 0;
  for(reco::GenParticleCollection::const_iterator ig = genParticles->begin(); ig!= genParticles->end(); ++ig,++indx) {
 
    if (debug)
      std::cout << "Track " << indx << " Status " << ig->status() << " charge "
		<< ig->charge() << " pdgId " << ig->pdgId() << " mass "
		<< ig->mass() << " P " << ig->momentum() << " E "
		<< ig->energy() << " Origin " << ig->vertex() << std::endl;
  }


  tree->Fill();
}

void IsolatedParticlesGeneratedJets::beginJob() {

  BookHistograms();

}

void IsolatedParticlesGeneratedJets::clearTreeVectors() {
  t_gjetN     ->clear();
  t_gjetE     ->clear();
  t_gjetPt    ->clear();
  t_gjetEta   ->clear();
  t_gjetPhi   ->clear();

  t_jetTrkP   ->clear();
  t_jetTrkPt  ->clear();
  t_jetTrkEta ->clear();
  t_jetTrkPhi ->clear();
  t_jetTrkPdg ->clear();
  t_jetTrkCharge ->clear();
}

void IsolatedParticlesGeneratedJets::BookHistograms(){

  tree = fs->make<TTree>("tree", "tree");

  t_gjetN     = new std::vector<int>   ();
  t_gjetE     = new std::vector<double>();
  t_gjetPt    = new std::vector<double>();
  t_gjetEta   = new std::vector<double>();
  t_gjetPhi   = new std::vector<double>();

  t_jetTrkP   = new std::vector<std::vector<double> >();
  t_jetTrkPt  = new std::vector<std::vector<double> >();
  t_jetTrkEta = new std::vector<std::vector<double> >();
  t_jetTrkPhi = new std::vector<std::vector<double> >();
  t_jetTrkPdg = new std::vector<std::vector<double> >();
  t_jetTrkCharge = new std::vector<std::vector<double> >();

  tree->Branch("t_gjetN",     "vector<int>",             &t_gjetN);
  tree->Branch("t_gjetE",     "vector<double>",          &t_gjetE);
  tree->Branch("t_gjetPt",    "vector<double>",          &t_gjetPt);
  tree->Branch("t_gjetEta",   "vector<double>",          &t_gjetEta);
  tree->Branch("t_gjetPhi",   "vector<double>",          &t_gjetPhi);

  tree->Branch("t_jetTrkP",   "vector<vector<double> >", &t_jetTrkP);
  tree->Branch("t_jetTrkPt",  "vector<vector<double> >", &t_jetTrkPt);
  tree->Branch("t_jetTrkEta", "vector<vector<double> >", &t_jetTrkEta);
  tree->Branch("t_jetTrkPhi", "vector<vector<double> >", &t_jetTrkPhi);
  tree->Branch("t_jetTrkPdg", "vector<vector<double> >", &t_jetTrkPdg);
  tree->Branch("t_jetTrkCharge", "vector<vector<double> >", &t_jetTrkCharge);

}

void IsolatedParticlesGeneratedJets::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedParticlesGeneratedJets);
