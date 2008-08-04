////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TH1F.h"

class SDDijetsAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  SDDijetsAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~SDDijetsAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob(const edm::EventSetup& eventSetup) ;
  virtual void endJob() ;

private:
  // Input from cfg file
  edm::InputTag genParticlesTag_;
  edm::InputTag genJetsTag_;

  // Histograms
  TH1F* hJet1Pt;
  TH1F* hJet1Eta;
  TH1F* hJet1Phi;
  TH1F* hJet2Pt;
  TH1F* hJet2Eta;
  TH1F* hJet2Phi;
  TH1F* hEnergyvsEta;	

  int nevents;
};

////////// Source code ////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

/// Constructor
SDDijetsAnalyzer::SDDijetsAnalyzer(const edm::ParameterSet& pset)
{
  genParticlesTag_ = pset.getParameter<edm::InputTag>("GenParticleTag");
  genJetsTag_ = pset.getParameter<edm::InputTag>("GenJetTag");
}

/// Destructor
SDDijetsAnalyzer::~SDDijetsAnalyzer(){
}

void SDDijetsAnalyzer::beginJob(const edm::EventSetup& eventSetup){
  edm::Service<TFileService> fs;
  //TH1::SetDefaultSumw2(true);

  hJet1Pt = fs->make<TH1F>("hJet1Pt","hJet1Pt",100,0.,200.);
  hJet1Eta = fs->make<TH1F>("hJet1Eta","hJet1Eta",100,-5.,5.);
  hJet1Phi = fs->make<TH1F>("hJet1Phi","hJet1Phi",100,-3.141592,3.141592);
  hJet2Pt = fs->make<TH1F>("hJet2Pt","hJet2Pt",100,0.,200.);
  hJet2Eta = fs->make<TH1F>("hJet2Eta","hJet2Eta",100,-5.,5.);  
  hJet2Phi = fs->make<TH1F>("hJet2Phi","hJet2Phi",100,-3.141592,3.141592);
  hEnergyvsEta = fs->make<TH1F>("hEnergyvsEta","hEnergyvsEta",100,-15.0,15.0); 		

  nevents = 0;
}

void SDDijetsAnalyzer::endJob(){
  hEnergyvsEta->Scale(1/(float)nevents);	
}

void SDDijetsAnalyzer::analyze(const edm::Event & ev, const edm::EventSetup&){
  nevents++; 

  // Generator Information
  edm::Handle<reco::GenParticleCollection> genParticles;
  ev.getByLabel(genParticlesTag_, genParticles);
  for(size_t i = 0; i < genParticles->size(); ++i) {
      		const reco::GenParticle& genpart = (*genParticles)[i];
      		//LogTrace("") << ">>>>>>> pid,status,px,py,px,e= "  << genpart.pdgId() << " , " << genpart.status() << " , " << genpart.px() << " , " << genpart.py() << " , " << genpart.pz() << " , " << genpart.energy();	
		if(genpart.status() != 1) continue;

		hEnergyvsEta->Fill(genpart.eta(),genpart.energy());	
  }

  edm::Handle<reco::GenJetCollection> genJets;
  ev.getByLabel(genJetsTag_,genJets);
  reco::GenJetCollection::const_iterator jet1 = genJets->end();
  reco::GenJetCollection::const_iterator jet2 = genJets->end();
  double firstpt = -1.;
  double secondpt = -1.;	
  for(reco::GenJetCollection::const_iterator genjet = genJets->begin(); genjet != genJets->end(); ++genjet){
	if(genjet->pt() > firstpt){
		firstpt = genjet->pt();
		jet2 = jet1;
		jet1 = genjet;
	} else if(genjet->pt() > secondpt){
		secondpt = genjet->pt();
		jet2 = genjet;
	}
  }
  if(jet1 != genJets->end()){
	hJet1Pt->Fill(jet1->pt());
	hJet1Eta->Fill(jet1->eta());
	hJet1Phi->Fill(jet1->phi());
  }
  if(jet2 != genJets->end()){
        hJet2Pt->Fill(jet2->pt());
        hJet2Eta->Fill(jet2->eta());
        hJet2Phi->Fill(jet2->phi());
  }	
}

DEFINE_FWK_MODULE(SDDijetsAnalyzer);
