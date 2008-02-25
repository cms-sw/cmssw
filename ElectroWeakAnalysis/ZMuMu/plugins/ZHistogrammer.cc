#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TH1.h"

class ZHistogrammer : public edm::EDAnalyzer {
public:
  ZHistogrammer(const edm::ParameterSet& pset); 
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  edm::InputTag  z_, gen_, match_;
  size_t nbinsPt_, nbinsAng_;
  double ptMax_, angMax_;
  TH1F *h_nZ_, *h_mZ_, *h_ptZ_, *h_phiZ_, *h_thetaZ_, *h_etaZ_, *h_rapidityZ_;
  TH1F *h_nZMC_, *h_mZMC_, *h_ptZMC_, *h_phiZMC_, *h_thetaZMC_, *h_etaZMC_, *h_rapidityZMC_;
  TH1F *h_mResZ_, *h_ptResZ_, *h_phiResZ_, *h_thetaResZ_, *h_etaResZ_, *h_rapidityResZ_;
};

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;

ZHistogrammer::ZHistogrammer(const ParameterSet& pset) :
  z_(pset.getParameter<InputTag>("z")),
  gen_(pset.getParameter<InputTag>("gen")), 
  match_(pset.getParameter<InputTag>("match")), 
  nbinsPt_(pset.getUntrackedParameter<size_t>("nbinsPt")),
  nbinsAng_(pset.getUntrackedParameter<size_t>("nbinsAng")),
  ptMax_(pset.getUntrackedParameter<double>("ptMax")),
  angMax_(pset.getUntrackedParameter<double>("angMax")) { 
  cout << ">>> Z Histogrammer constructor" << endl;
  Service<TFileService> fs;
  TFileDirectory ZHisto = fs->mkdir( "ZRecoHisto" );
  TFileDirectory ZMCHisto = fs->mkdir( "ZMCHisto" );
  TFileDirectory ZResHisto = fs->mkdir( "ZResHisto" );
  h_nZ_ = ZHisto.make<TH1F>("ZNumber", "number of Z particles", 11, -0.5, 10.5);
  h_mZ_ = ZHisto.make<TH1F>("ZMass", "Z mass (GeV/c^{2})", 100,  0, 200);
  h_ptZ_ = ZHisto.make<TH1F>("ZPt", "Z p_{t} (GeV/c)", nbinsPt_, 0, ptMax_);
  h_phiZ_ = ZHisto.make<TH1F>("ZPhi", "Z #phi", nbinsAng_,  -angMax_, angMax_);
  h_thetaZ_ = ZHisto.make<TH1F>("Ztheta", "Z #theta", nbinsAng_,  0, angMax_);
  h_etaZ_ = ZHisto.make<TH1F>("ZEta", "Z #eta", nbinsAng_,  -angMax_, angMax_);
  h_rapidityZ_ = ZHisto.make<TH1F>("ZRapidity", "Z rapidity", nbinsAng_,  -angMax_, angMax_);
  h_nZMC_ = ZMCHisto.make<TH1F>("ZMCNumber", "number of Z MC particles", 11, -0.5, 10.5);
  h_mZMC_ = ZMCHisto.make<TH1F>("ZMCMass", "Z MC mass (GeV/c^{2})", 100,  0, 200);
  h_ptZMC_ = ZMCHisto.make<TH1F>("ZMCPt", "Z MC p_{t} (GeV/c)", nbinsPt_, 0, ptMax_);
  h_phiZMC_ = ZMCHisto.make<TH1F>("ZMCPhi", "Z MC #phi", nbinsAng_,  -angMax_, angMax_);
  h_thetaZMC_ = ZMCHisto.make<TH1F>("ZMCTheta", "Z MC #theta", nbinsAng_,  0, angMax_);
  h_etaZMC_ = ZMCHisto.make<TH1F>("ZMCEta", "Z MC #eta", nbinsAng_,  -angMax_, angMax_);
  h_rapidityZMC_ = ZMCHisto.make<TH1F>("ZMCRapidity", "Z MC rapidity", 
				       nbinsAng_,  -angMax_, angMax_);
  h_mResZ_ = ZResHisto.make<TH1F>("ZMassRes", "Z mass Resolution (GeV/c^{2})", 240,  -60, 60);
  h_ptResZ_ = ZResHisto.make<TH1F>("ZPtRes", "Z p_{t} Resolution (GeV/c)", 
				   nbinsPt_, -ptMax_, ptMax_);
  h_phiResZ_ = ZResHisto.make<TH1F>("ZPhiRes", "Z #phi Resolution", 
				    nbinsAng_,  -angMax_, angMax_);
  h_thetaResZ_ = ZResHisto.make<TH1F>("ZThetaRes", "Z #theta Resolution", 
				      nbinsAng_, -angMax_, angMax_);
  h_etaResZ_ = ZResHisto.make<TH1F>("ZEtaRes", "Z #eta Resolution", 
				    nbinsAng_,  -angMax_, angMax_);
  h_rapidityResZ_ = ZResHisto.make<TH1F>("ZRapidityRes", "Z rapidity Resolution", 
					 nbinsAng_,  -angMax_, angMax_);
}

void ZHistogrammer::analyze(const edm::Event& event, const edm::EventSetup& setup) { 
  cout << ">>> Z Histogrammer analyze" << endl;
  Handle<CandidateCollection> z;
  Handle<CandidateCollection> gen;
  Handle<CandMatchMap> match;
  event.getByLabel(z_, z);
  event.getByLabel(gen_, gen);
  event.getByLabel(match_, match);
  h_nZ_->Fill(z->size());
  for(size_t i = 0; i < z->size(); ++i) {
    const Candidate &zCand = (*z)[i];
    h_mZ_->Fill(zCand.mass());
    h_ptZ_->Fill(zCand.pt());
    h_phiZ_->Fill(zCand.phi());
    h_thetaZ_->Fill(zCand.theta());
    h_etaZ_->Fill(zCand.eta());
    h_rapidityZ_->Fill(zCand.rapidity());
    if(zCand.hasMasterClone()) {
      cout << ">>> Z has masterClone!" << endl;
      CandidateRef zCandRef(z, i);
      CandidateRef zMCMatch = (*match)[zCandRef];
      h_mResZ_->Fill(zCandRef->mass() - zMCMatch->mass());
      h_ptResZ_->Fill(zCandRef->pt() - zMCMatch->pt());
      h_phiResZ_->Fill(zCandRef->phi() - zMCMatch->phi());
      h_thetaResZ_->Fill(zCandRef->theta() - zMCMatch->theta());
      h_etaResZ_->Fill(zCandRef->eta() - zMCMatch->eta());
      h_rapidityResZ_->Fill(zCandRef->rapidity() - zMCMatch->rapidity());
    } else cout << ">>> Sorry, no masterClone for Z!" << endl;
  }
  h_nZMC_->Fill(gen->size());
  for(size_t i = 0; i < gen->size(); ++i) {
    const Candidate &genCand = (*gen)[i];
    if((genCand.pdgId() == 23) && (genCand.status() == 2)) //this is an intermediate Z0
      cout << ">>> intermediate Z0 found, with " << genCand.numberOfDaughters() 
	   << " daughters" << endl;
    if((genCand.pdgId() == 23)&&(genCand.status() == 3)) { //this is a Z0
      cout << ">>> Z0 found, with " << genCand.numberOfDaughters() 
	   << " daughters" << endl;
      h_mZMC_->Fill(genCand.mass());
      h_ptZMC_->Fill(genCand.pt());
      h_phiZMC_->Fill(genCand.phi());
      h_thetaZMC_->Fill(genCand.theta());
      h_etaZMC_->Fill(genCand.eta());
      h_rapidityZMC_->Fill(genCand.rapidity());
      if(genCand.numberOfDaughters() == 3) {//Z0 decays in mu+ mu-, the 3rd daughter is the same Z0
	const Candidate * dauGen0 = genCand.daughter(0);
	const Candidate * dauGen1 = genCand.daughter(1);
	const Candidate * dauGen2 = genCand.daughter(2);
	cout << ">>> daughter MC 0 PDG Id " << dauGen0->pdgId() 
	     << ", status " << dauGen0->status() 
	     << ", charge " << dauGen0->charge() 
	     << endl;
	cout << ">>> daughter MC 1 PDG Id " << dauGen1->pdgId() 
	     << ", status " << dauGen1->status()
	     << ", charge " << dauGen1->charge() 
	     << endl;
	cout << ">>> daughter MC 2 PDG Id " << dauGen2->pdgId() 
	     << ", status " << dauGen2->status()
	     << ", charge " << dauGen2->charge() << endl;
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZHistogrammer);

