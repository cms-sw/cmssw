#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "TH1.h"

class ZHistogrammer : public edm::EDAnalyzer {
public:
  ZHistogrammer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  edm::EDGetTokenT<reco::CandidateCollection>  zToken_;
  edm::EDGetTokenT<reco::CandidateCollection>  genToken_;
  edm::EDGetTokenT<reco::CandMatchMap>  matchToken_;
  unsigned int nbinsMass_, nbinsPt_, nbinsAng_, nbinsMassRes_;
  double massMax_, ptMax_, angMax_, massResMax_;
  TH1F *h_nZ_, *h_mZ_, *h_ptZ_, *h_phiZ_, *h_thetaZ_, *h_etaZ_, *h_rapidityZ_;
  TH1F *h_invmMuMu_;
  TH1F *h_nZMC_, *h_mZMC_, *h_ptZMC_, *h_phiZMC_, *h_thetaZMC_, *h_etaZMC_, *h_rapidityZMC_;
  TH1F *h_invmMuMuMC_;
  //TH1F *h_mZ2vs3MC_, *h_ptZ2vs3MC_, *h_phiZ2vs3MC_, *h_thetaZ2vs3MC_, *h_etaZ2vs3MC_, *h_rapidityZ2vs3MC_;
  TH1F *h_mResZ_, *h_ptResZ_, *h_phiResZ_, *h_thetaResZ_, *h_etaResZ_, *h_rapidityResZ_;
  TH1F *h_mResZMuMu_, *h_mRatioZMuMu_;
  TH1F *h_mResZMuMuMC_, *h_mRatioZMuMuMC_;
};

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <cmath>
#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;

ZHistogrammer::ZHistogrammer(const ParameterSet& pset) :
  zToken_(consumes<CandidateCollection>(pset.getParameter<InputTag>("z"))),
  genToken_(consumes<CandidateCollection>(pset.getParameter<InputTag>("gen"))),
  matchToken_(consumes<CandMatchMap>(pset.getParameter<InputTag>("match"))),
  nbinsMass_(pset.getUntrackedParameter<unsigned int>("nbinsMass")),
  nbinsPt_(pset.getUntrackedParameter<unsigned int>("nbinsPt")),
  nbinsAng_(pset.getUntrackedParameter<unsigned int>("nbinsAng")),
  nbinsMassRes_(pset.getUntrackedParameter<unsigned int>("nbinsMassRes")),
  massMax_(pset.getUntrackedParameter<double>("massMax")),
  ptMax_(pset.getUntrackedParameter<double>("ptMax")),
  angMax_(pset.getUntrackedParameter<double>("angMax")),
  massResMax_(pset.getUntrackedParameter<double>("massResMax")) {
  cout << ">>> Z Histogrammer constructor" << endl;
  Service<TFileService> fs;
  TFileDirectory ZHisto = fs->mkdir( "ZRecoHisto" );
  TFileDirectory ZMCHisto = fs->mkdir( "ZMCHisto" );
  TFileDirectory ZResHisto = fs->mkdir( "ZResolutionHisto" );
  //TFileDirectory Z2vs3MCHisto = fs->mkdir( "Z2vs3MCHisto" );
  h_nZ_ = ZHisto.make<TH1F>("ZNumber", "number of Z particles", 11, -0.5, 10.5);
  h_mZ_ = ZHisto.make<TH1F>("ZMass", "Z mass (GeV/c^{2})", nbinsMass_,  0, massMax_);
  h_ptZ_ = ZHisto.make<TH1F>("ZPt", "Z p_{t} (GeV/c)", nbinsPt_, 0, ptMax_);
  h_phiZ_ = ZHisto.make<TH1F>("ZPhi", "Z #phi", nbinsAng_,  -angMax_, angMax_);
  h_thetaZ_ = ZHisto.make<TH1F>("Ztheta", "Z #theta", nbinsAng_,  0, angMax_);
  h_etaZ_ = ZHisto.make<TH1F>("ZEta", "Z #eta", nbinsAng_,  -angMax_, angMax_);
  h_rapidityZ_ = ZHisto.make<TH1F>("ZRapidity", "Z rapidity", nbinsAng_,  -angMax_, angMax_);
  h_invmMuMu_ = ZHisto.make<TH1F>("MuMuMass", "#mu #mu invariant mass",
				  nbinsMass_,  0, massMax_);
  h_nZMC_ = ZMCHisto.make<TH1F>("ZMCNumber", "number of Z MC particles", 11, -0.5, 10.5);
  h_mZMC_ = ZMCHisto.make<TH1F>("ZMCMass", "Z MC mass (GeV/c^{2})", nbinsMass_,  0, massMax_);
  h_ptZMC_ = ZMCHisto.make<TH1F>("ZMCPt", "Z MC p_{t} (GeV/c)", nbinsPt_, 0, ptMax_);
  h_phiZMC_ = ZMCHisto.make<TH1F>("ZMCPhi", "Z MC #phi", nbinsAng_,  -angMax_, angMax_);
  h_thetaZMC_ = ZMCHisto.make<TH1F>("ZMCTheta", "Z MC #theta", nbinsAng_,  0, angMax_);
  h_etaZMC_ = ZMCHisto.make<TH1F>("ZMCEta", "Z MC #eta", nbinsAng_,  -angMax_, angMax_);
  h_rapidityZMC_ = ZMCHisto.make<TH1F>("ZMCRapidity", "Z MC rapidity",
				       nbinsAng_,  -angMax_, angMax_);
  h_invmMuMuMC_ = ZMCHisto.make<TH1F>("MuMuMCMass", "#mu #mu MC invariant mass",
				      nbinsMass_,  0, massMax_);
  /*
  h_mZ2vs3MC_ = Z2vs3MCHisto.make<TH1F>("Z2vs3MCMass", "Z MC st 2 vs st 3 mass (GeV/c^{2})",
					nbinsMassRes_, -massResMax_, massResMax_);
  h_ptZ2vs3MC_ = Z2vs3MCHisto.make<TH1F>("Z2vs3MCPt", "Z MC st 2 vs st 3 p_{t} (GeV/c)",
					 nbinsPt_, -ptMax_, ptMax_);
  h_phiZ2vs3MC_ = Z2vs3MCHisto.make<TH1F>("Z2vs3MCPhi", "Z MC st 2 vs st 3 #phi",
					  nbinsAng_,  -angMax_, angMax_);
  h_thetaZ2vs3MC_ = Z2vs3MCHisto.make<TH1F>("Z2vs3MCTheta", "Z MC st 2 vs st 3 #theta",
					    nbinsAng_,  -angMax_, angMax_);
  h_etaZ2vs3MC_ = Z2vs3MCHisto.make<TH1F>("Z2vs3MCEta", "Z MC st 2 vs st 3 #eta",
					  nbinsAng_,  -angMax_, angMax_);
  h_rapidityZ2vs3MC_ = Z2vs3MCHisto.make<TH1F>("Z2vs3MCRapidity", "Z MC st 2 vs st 3 rapidity",
				       nbinsAng_,  -angMax_, angMax_);
  */
  h_mResZ_ = ZResHisto.make<TH1F>("ZMassResolution", "Z mass Resolution (GeV/c^{2})",
				  nbinsMassRes_, -massResMax_, massResMax_);
  h_ptResZ_ = ZResHisto.make<TH1F>("ZPtResolution", "Z p_{t} Resolution (GeV/c)",
				   nbinsPt_, -ptMax_, ptMax_);
  h_phiResZ_ = ZResHisto.make<TH1F>("ZPhiResolution", "Z #phi Resolution",
				    nbinsAng_,  -angMax_, angMax_);
  h_thetaResZ_ = ZResHisto.make<TH1F>("ZThetaResolution", "Z #theta Resolution",
				      nbinsAng_, -angMax_, angMax_);
  h_etaResZ_ = ZResHisto.make<TH1F>("ZEtaResolution", "Z #eta Resolution",
				    nbinsAng_,  -angMax_, angMax_);
  h_rapidityResZ_ = ZResHisto.make<TH1F>("ZRapidityResolution", "Z rapidity Resolution",
					 nbinsAng_,  -angMax_, angMax_);
  h_mResZMuMu_ = ZResHisto.make<TH1F>("ZToMuMuRecoMassResolution",
				      "Z Reco vs matched final state #mu #mu mass Difference (GeV/c^{2})",
				      nbinsMassRes_, -massResMax_, massResMax_);
  h_mRatioZMuMu_ = ZResHisto.make<TH1F>("ZToMuMuRecoMassRatio",
					"Z Reco vs matched final state #mu #mu mass Ratio",
					4000, 0, 2);
  h_mResZMuMuMC_ = ZResHisto.make<TH1F>("ZToMuMuMCMassResolution",
					"Z vs final state #mu #mu MC mass Difference (GeV/c^{2})",
					nbinsMassRes_/2 + 1, -2*massResMax_/nbinsMassRes_, massResMax_);
  h_mRatioZMuMuMC_ = ZResHisto.make<TH1F>("ZToMuMuMCMassRatio",
					  "Z vs final state #mu #mu MC mass Ratio",
					  2002, 0.999, 2);
}

void ZHistogrammer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  cout << ">>> Z Histogrammer analyze" << endl;
  Handle<CandidateCollection> z;
  Handle<CandidateCollection> gen;
  Handle<CandMatchMap> match;
  event.getByToken(zToken_, z);
  event.getByToken(genToken_, gen);
  event.getByToken(matchToken_, match);
  h_nZ_->Fill(z->size());
  for(unsigned int i = 0; i < z->size(); ++i) {
    const Candidate &zCand = (*z)[i];
    h_mZ_->Fill(zCand.mass());
    h_ptZ_->Fill(zCand.pt());
    h_phiZ_->Fill(zCand.phi());
    h_thetaZ_->Fill(zCand.theta());
    h_etaZ_->Fill(zCand.eta());
    h_rapidityZ_->Fill(zCand.rapidity());
    CandidateRef zCandRef(z, i);
    CandidateRef zMCMatch = (*match)[zCandRef];
    if(zMCMatch.isNonnull() && zMCMatch->pdgId()==23) {
      h_mResZ_->Fill(zCandRef->mass() - zMCMatch->mass());
      h_ptResZ_->Fill(zCandRef->pt() - zMCMatch->pt());
      h_phiResZ_->Fill(zCandRef->phi() - zMCMatch->phi());
      h_thetaResZ_->Fill(zCandRef->theta() - zMCMatch->theta());
      h_etaResZ_->Fill(zCandRef->eta() - zMCMatch->eta());
      h_rapidityResZ_->Fill(zCandRef->rapidity() - zMCMatch->rapidity());
      const Candidate * dau0 = zMCMatch->daughter(0);
      const Candidate * dau1 = zMCMatch->daughter(1);
      for(unsigned int i0 = 0; i0 < dau0->numberOfDaughters(); ++i0) {
	const Candidate * ddau0 = dau0->daughter(i0);
	if(abs(ddau0->pdgId())==13 && ddau0->status()==1) {
	  dau0 = ddau0; break;
	}
      }
      for(unsigned int i1 = 0; i1 < dau1->numberOfDaughters(); ++i1) {
	const Candidate * ddau1 = dau1->daughter(i1);
	if(abs(ddau1->pdgId())==13 && ddau1->status()==1) {
	  dau1 = ddau1; break;
	}
      }
      assert(abs(dau0->pdgId())==13 && dau0->status()==1);
      assert(abs(dau1->pdgId())==13 && dau1->status()==1);
      double invMass = (dau0->p4()+dau1->p4()).mass();
      h_invmMuMu_->Fill(invMass);
      h_mResZMuMu_->Fill(zCand.mass() - invMass);
      h_mRatioZMuMu_->Fill(zCand.mass()/invMass);
    }
  }
  h_nZMC_->Fill(gen->size());
  for(unsigned int i = 0; i < gen->size(); ++i) {
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
      Particle::LorentzVector pZ(0, 0, 0, 0);
      int nMu = 0;
      for(unsigned int j = 0; j < genCand.numberOfDaughters(); ++j) {
	const Candidate *dauGen = genCand.daughter(j);
	/*
	if((dauGen->pdgId() == 23) && (dauGen->status() == 2)) {
	  h_mZ2vs3MC_->Fill(genCand.mass() - dauGen->mass());
	  h_ptZ2vs3MC_->Fill(genCand.pt() - dauGen->pt());
	  h_phiZ2vs3MC_->Fill(genCand.phi() - dauGen->phi());
	  h_thetaZ2vs3MC_->Fill(genCand.theta() - dauGen->theta());
	  h_etaZ2vs3MC_->Fill(genCand.eta() - dauGen->eta());
	  h_rapidityZ2vs3MC_->Fill(genCand.rapidity() - dauGen->rapidity());
	}
	*/
	if((abs(dauGen->pdgId()) == 13) && (dauGen->numberOfDaughters() != 0)) {
	  //we are looking for photons of final state radiation
	  cout << ">>> The muon " << j
	       << " has " << dauGen->numberOfDaughters() << " daughters" <<endl;
	  for(unsigned int k = 0; k < dauGen->numberOfDaughters(); ++k) {
	    const Candidate * dauMuGen = dauGen->daughter(k);
	    cout << ">>> Mu " << j
		 << " daughter MC " << k
		 << " PDG Id " << dauMuGen->pdgId()
		 << ", status " << dauMuGen->status()
		 << ", charge " << dauMuGen->charge()
		 << endl;
	    if(abs(dauMuGen->pdgId()) == 13 && dauMuGen->status() ==1) {
	      pZ += dauMuGen->p4();
	      nMu ++;
	    }
	  }
	}
      }
      assert(nMu == 2);
      double mZ = pZ.mass();
      h_invmMuMuMC_->Fill(mZ);
      h_mResZMuMuMC_->Fill(genCand.mass() - mZ);
      h_mRatioZMuMuMC_->Fill(genCand.mass()/mZ);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZHistogrammer);

