/* \class gamma_radiative_analyzer
 *
 * author: Pasquale Noli
 *
 * Gamma Radiative analyzer
 *
 *
 */
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <iostream>
#include <iterator>
#include <cmath>
#include <vector>
#include "TH1.h"
#include "TH2.h"
//#include "TH3.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;

typedef edm::ValueMap<float> IsolationCollection;

class gamma_radiative_analyzer : public edm::EDAnalyzer {
public:
  gamma_radiative_analyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  virtual void endJob() override;

  EDGetTokenT<CandidateView> zMuMuToken_;
  EDGetTokenT<GenParticleMatch> zMuMuMatchMapToken_;
  EDGetTokenT<CandidateView> zMuTkToken_;
  EDGetTokenT<GenParticleMatch> zMuTkMatchMapToken_;
  EDGetTokenT<CandidateView> zMuSaToken_;
  EDGetTokenT<GenParticleMatch> zMuSaMatchMapToken_;
  double dRVeto_, dRTrk_, ptThreshold_;
  //histograms
  TH2D *h_gamma_pt_eta_, *h_mu_pt_eta_FSR_, *h_mu_pt_eta_no_FSR_;

  //boolean
  bool  FSR_mu, FSR_tk, FSR_mu0, FSR_mu1;

  //counter
  int  zmmcounter , zmscounter, zmtcounter,numOfEvent,numofGamma;
};

gamma_radiative_analyzer::gamma_radiative_analyzer(const ParameterSet& pset) :
  zMuMuToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuMu"))),
  zMuMuMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuMuMatchMap"))),
  zMuTkToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuTk"))),
  zMuTkMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuTkMatchMap"))),
  zMuSaToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuSa"))),
  zMuSaMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuSaMatchMap"))){
  zmmcounter=0;
  zmscounter=0;
  zmtcounter=0;
  numOfEvent=0;
  numofGamma=0;
  Service<TFileService> fs;

  // general histograms
  h_gamma_pt_eta_= fs->make<TH2D>("h_gamma_pt_eta","pt vs eta of gamma",100,20,100,100,-2.0,2.0);
  h_mu_pt_eta_FSR_= fs->make<TH2D>("h_mu_pt_eta_FSR","pt vs eta of muon with FSR",100,20,100,100,-2.0,2.0 );
  h_mu_pt_eta_no_FSR_= fs->make<TH2D>("h_mu_pt_eta_no_FSR","pt vs eta of of muon withot FSR",100,20,100,100,-2.0,2.0);
}

void gamma_radiative_analyzer::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateView> zMuMu;                //Collection of Z made by  Mu global + Mu global
  Handle<GenParticleMatch> zMuMuMatchMap;     //Map of Z made by Mu global + Mu global with MC
  event.getByToken(zMuMuToken_, zMuMu);
  Handle<CandidateView> zMuTk;                //Collection of Z made by  Mu global + Track
  Handle<GenParticleMatch> zMuTkMatchMap;
  event.getByToken(zMuTkToken_, zMuTk);
  Handle<CandidateView> zMuSa;                //Collection of Z made by  Mu global + Sa
  Handle<GenParticleMatch> zMuSaMatchMap;
  event.getByToken(zMuSaToken_, zMuSa);
  numOfEvent++;
  // ZMuMu
  if (zMuMu->size() > 0 ) {
    event.getByToken(zMuMuMatchMapToken_, zMuMuMatchMap);
     for(unsigned int i = 0; i < zMuMu->size(); ++i) { //loop on candidates

      const Candidate & zMuMuCand = (*zMuMu)[i]; //the candidate
      CandidateBaseRef zMuMuCandRef = zMuMu->refAt(i);


      CandidateBaseRef dau0 = zMuMuCand.daughter(0)->masterClone();
      CandidateBaseRef dau1 = zMuMuCand.daughter(1)->masterClone();
      const pat::Muon& mu0 = dynamic_cast<const pat::Muon&>(*dau0);//cast in patMuon
      const pat::Muon& mu1 = dynamic_cast<const pat::Muon&>(*dau1);

      double zmass= zMuMuCand.mass();
      double pt0 = mu0.pt();
      double pt1 = mu1.pt();
      double eta0 = mu0.eta();
      double eta1 = mu1.eta();
      if(pt0>20 && pt1 > 20 && abs(eta0)<2 && abs(eta1)<2 && zmass > 20 && zmass < 200){
	GenParticleRef zMuMuMatch = (*zMuMuMatchMap)[zMuMuCandRef];
	if(zMuMuMatch.isNonnull()) {  // ZMuMu matched
	  zmmcounter++;
	  FSR_mu0 = false;
	  FSR_mu1 = false;

	  //MonteCarlo Study
	  const reco::GenParticle * muMc0 = mu0.genLepton();
	  const reco::GenParticle * muMc1 = mu1.genLepton();
	  const Candidate * motherMu0 =  muMc0->mother();
	  const Candidate * motherMu1 =  muMc1->mother();
	  int num_dau_muon0 = motherMu0->numberOfDaughters();
	  int num_dau_muon1 = motherMu1->numberOfDaughters();
	  if( num_dau_muon0 > 1 ){
	    for(int j = 0; j <  num_dau_muon0; ++j){
	      int id =motherMu0 ->daughter(j)->pdgId();
	      if(id == 22){
		double etaG = motherMu0 ->daughter(j)->eta();
		double ptG = motherMu0 ->daughter(j)->pt();
		h_gamma_pt_eta_->Fill(ptG,etaG);
		h_mu_pt_eta_FSR_->Fill(pt0,eta0);
		FSR_mu0=true;
		numofGamma++;
	      }
	    }
	  }//end check of gamma
	  if(!FSR_mu0)	h_mu_pt_eta_no_FSR_->Fill(pt0,eta0);
	  if( num_dau_muon1 > 1 ){
	    for(int j = 0; j <  num_dau_muon1; ++j){
	      int id = motherMu1->daughter(j)->pdgId();
	      if(id == 22){
		double etaG = motherMu1 ->daughter(j)->eta();
		double ptG = motherMu1 ->daughter(j)->pt();
		h_gamma_pt_eta_->Fill(ptG,etaG);
		h_mu_pt_eta_FSR_->Fill(pt1,eta1);
		FSR_mu1=true;
		numofGamma++;
	      }
	    }
	  }//end check of gamma
	  if(!FSR_mu1)	h_mu_pt_eta_no_FSR_->Fill(pt1,eta1);
	}// end MC match
      }//end of cuts
     }// end loop on ZMuMu cand
  }// end if ZMuMu size > 0

  // ZMuSa
  if (zMuSa->size() > 0 ) {
    event.getByToken(zMuSaMatchMapToken_, zMuSaMatchMap);
     for(unsigned int i = 0; i < zMuSa->size(); ++i) { //loop on candidates

      const Candidate & zMuSaCand = (*zMuSa)[i]; //the candidate
      CandidateBaseRef zMuSaCandRef = zMuSa->refAt(i);


      CandidateBaseRef dau0 = zMuSaCand.daughter(0)->masterClone();
      CandidateBaseRef dau1 = zMuSaCand.daughter(1)->masterClone();
      const pat::Muon& mu0 = dynamic_cast<const pat::Muon&>(*dau0);//cast in patMuon
      const pat::Muon& mu1 = dynamic_cast<const pat::Muon&>(*dau1);

      double zmass= zMuSaCand.mass();
      double pt0 = mu0.pt();
      double pt1 = mu1.pt();
      double eta0 = mu0.eta();
      double eta1 = mu1.eta();
      if(pt0>20 && pt1 > 20 && abs(eta0)<2 && abs(eta1)<2 && zmass > 20 && zmass < 200){
	GenParticleRef zMuSaMatch = (*zMuSaMatchMap)[zMuSaCandRef];
	if(zMuSaMatch.isNonnull()) {  // ZMuSa matched
	  FSR_mu0 = false;
	  FSR_mu1 = false;
	  zmscounter++;
	  //MonteCarlo Study
	  const reco::GenParticle * muMc0 = mu0.genLepton();
	  const reco::GenParticle * muMc1 = mu1.genLepton();
	  const Candidate * motherMu0 =  muMc0->mother();
	  const Candidate * motherMu1 =  muMc1->mother();
	  int num_dau_muon0 = motherMu0->numberOfDaughters();
	  int num_dau_muon1 = motherMu1->numberOfDaughters();
	  if( num_dau_muon0 > 1 ){
	    for(int j = 0; j <  num_dau_muon0; ++j){
	      int id =motherMu0 ->daughter(j)->pdgId();
	      if(id == 22){
		double etaG = motherMu0 ->daughter(j)->eta();
		double ptG = motherMu0 ->daughter(j)->pt();
		h_gamma_pt_eta_->Fill(ptG,etaG);
		h_mu_pt_eta_FSR_->Fill(pt0,eta0);
		numofGamma++;
		FSR_mu0=true;
	      }
	    }
	  }//end check of gamma
	  if(!FSR_mu0)	h_mu_pt_eta_no_FSR_->Fill(pt0,eta0);
	  if( num_dau_muon1 > 1 ){
	    for(int j = 0; j <  num_dau_muon1; ++j){
	      int id = motherMu1->daughter(j)->pdgId();
	      if(id == 22){
		double etaG = motherMu1 ->daughter(j)->eta();
		double ptG = motherMu1 ->daughter(j)->pt();
		h_gamma_pt_eta_->Fill(ptG,etaG);
		h_mu_pt_eta_FSR_->Fill(pt1,eta1);
		numofGamma++;
		FSR_mu1=true;
	      }
	    }
	  }//end check of gamma
	  if(!FSR_mu1)	h_mu_pt_eta_no_FSR_->Fill(pt1,eta1);
	}// end MC match
      }//end of cuts
     }// end loop on ZMuSa cand
  }// end if ZMuSa size > 0



  //ZMuTk
  if (zMuTk->size() > 0 ) {
    event.getByToken(zMuTkMatchMapToken_, zMuTkMatchMap);
    for(unsigned int i = 0; i < zMuTk->size(); ++i) { //loop on candidates
      const Candidate & zMuTkCand = (*zMuTk)[i]; //the candidate
      CandidateBaseRef zMuTkCandRef = zMuTk->refAt(i);


      CandidateBaseRef dau0 = zMuTkCand.daughter(0)->masterClone();
      CandidateBaseRef dau1 = zMuTkCand.daughter(1)->masterClone();
      const pat::Muon& mu0 = dynamic_cast<const pat::Muon&>(*dau0);//cast in patMuon
      const pat::GenericParticle& mu1 = dynamic_cast<const pat::GenericParticle &>(*dau1);


      double zmass= zMuTkCand.mass();
      double pt0 = mu0.pt();
      double pt1 = mu1.pt();
      double eta0 = mu0.eta();
      double eta1 = mu1.eta();
      if(pt0>20 && pt1 > 20 && abs(eta0)<2 && abs(eta1)<2 && zmass > 20 && zmass < 200){//kinematical cuts
	GenParticleRef zMuTkMatch = (*zMuTkMatchMap)[zMuTkCandRef];
	if(zMuTkMatch.isNonnull()) {  // ZMuTk matched
	  FSR_mu = false;
	  FSR_tk = false;
	  zmtcounter++;
	  //MonteCarlo Study
	  const reco::GenParticle * muMc0 = mu0.genLepton();
	  const reco::GenParticle * muMc1 = mu1.genParticle() ;
	  const Candidate * motherMu0 =  muMc0->mother();
	  const Candidate * motherMu1 =  muMc1->mother();
	  int num_dau_muon0 = motherMu0->numberOfDaughters();
	  int num_dau_muon1 = motherMu1->numberOfDaughters();
	  if( num_dau_muon0 > 1 ){
	    for(int j = 0; j <  num_dau_muon0; ++j){
	      int id = motherMu0->daughter(j)->pdgId();
	      if(id == 22){
		double etaG = motherMu0 ->daughter(j)->eta();
		double ptG = motherMu0 ->daughter(j)->pt();
		h_gamma_pt_eta_->Fill(ptG,etaG);
		h_mu_pt_eta_FSR_->Fill(pt0,eta0);
		numofGamma++;
		FSR_mu0=true;
	      }
	    }
	  }//end check of gamma
	  if(!FSR_mu0)	h_mu_pt_eta_no_FSR_->Fill(pt0,eta0);
	  if( num_dau_muon1 > 1 ){
	    for(int j = 0; j <  num_dau_muon1; ++j){
	      int id = motherMu1->daughter(j)->pdgId();
	      if(id == 22){
		double etaG = motherMu1 ->daughter(j)->eta();
		double ptG = motherMu1 ->daughter(j)->pt();
		h_gamma_pt_eta_->Fill(ptG,etaG);
		h_mu_pt_eta_FSR_->Fill(pt1,eta1);
		numofGamma++;
		FSR_mu1=true;
	      }
	    }
	  }//end check of gamma
	  if(!FSR_mu1)	h_mu_pt_eta_no_FSR_->Fill(pt1,eta1);
	}// end MC match
      }//end Kine-cuts
    }// end loop on ZMuTk cand
  }// end if ZMuTk size > 0
}// end analyze



void gamma_radiative_analyzer::endJob() {
  cout <<" ============= Summary =========="<<endl;
  cout <<" Numero di eventi = "<< numOfEvent << endl;
  cout <<" 1)Numero di ZMuMu matched dopo i tagli cinematici = "<< zmmcounter << endl;
  cout <<" 2)Numero di ZMuSa matched dopo i tagli cinematici = "<< zmscounter << endl;
  cout <<" 3)Numero di ZMuTk matched dopo i tagli cinematici = "<< zmtcounter << endl;
  cout <<" 4)Number of gamma = "<< numofGamma << endl;
  }

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(gamma_radiative_analyzer);
