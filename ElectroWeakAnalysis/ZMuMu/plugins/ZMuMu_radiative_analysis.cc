/* \class ZMuMu_Radiative_analyzer
 * 
 * author: Pasquale Noli
 *
 * ZMuMu Radiative analyzer
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
#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include <iostream>
#include <iterator>
#include <cmath>
#include <vector>
#include "TH1.h"
//#include "TH2.h"
//#include "TH3.h"


using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;

class ZMuMu_Radiative_analyzer : public edm::EDAnalyzer {
public:
  ZMuMu_Radiative_analyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endJob();

  edm::InputTag zMuMu_, zMuMuMatchMap_, zMuTk_, zMuTkMatchMap_;
  double dRVeto_, dRTrk_, ptThreshold_;
  //histograms 
  TH1D *h_eta_, *h_pt_,*h_zmass_FSR,*h_zmass_no_FSR;
  TH1D *h_eta_rad_, *h_pt_rad_;
  TH1D *h_eta_photon_, *h_pt_photon_;
  TH1D *h_Iso_tk_, *h_Iso_mu_;
  TH1D *h_Iso_tk_FSR_, *h_Iso_mu_FSR_;

  //couter
  int irradiatecouter;
  int zmatchedcounter;
  int numZMuTk,numZMuTkCut,numZMuTkMatched,numTk_FRS;
   //boolean 
  bool FSR,FSR_mu,FSR_tk;
 };


typedef edm::ValueMap<float> IsolationCollection;

ZMuMu_Radiative_analyzer::ZMuMu_Radiative_analyzer(const ParameterSet& pset) : 
  zMuMu_(pset.getParameter<InputTag>("zMuMu")), 
  zMuMuMatchMap_(pset.getParameter<InputTag>("zMuMuMatchMap")),
  zMuTk_(pset.getParameter<InputTag>("zMuTk")), 
  zMuTkMatchMap_(pset.getParameter<InputTag>("zMuTkMatchMap")),
  dRVeto_(pset.getUntrackedParameter<double>("veto")),
  dRTrk_(pset.getUntrackedParameter<double>("deltaRTrk")),
  ptThreshold_(pset.getUntrackedParameter<double>("ptThreshold")){ 

  
  Service<TFileService> fs;
   
  // general histograms
  h_eta_ = fs->make<TH1D>("h_Eta","Eta distribution",70,-3.5,3.5);
  h_pt_ = fs->make<TH1D>("h_Pt","Pt distribution",180,20,200);
  h_eta_rad_ = fs->make<TH1D>("h_Eta_rad","Eta distribution",70,-3.5,3.5);
  h_pt_rad_ = fs->make<TH1D>("h_Pt_rad","Pt distribution",180,20,200);
  h_eta_photon_ = fs->make<TH1D>("h_Eta_photon","Eta distribution",70,-3.5,3.5);
  h_pt_photon_ = fs->make<TH1D>("h_Pt_photon","Pt distribution",180,20,200);
  h_zmass_FSR= fs->make<TH1D>("h_zmass_FRS","Invariant Z mass distribution",200,0,200);
  h_zmass_no_FSR= fs->make<TH1D>("h_zmass_no_FSR","Invariant Z mass distribution",200,0,200);
  h_Iso_tk_= fs->make<TH1D>("h_iso_tk","Isolation distribution",100,0,20);
  h_Iso_tk_FSR_= fs->make<TH1D>("h_iso_tk_FSR","Isolation distribution ",100,0,20);
  h_Iso_mu_= fs->make<TH1D>("h_iso_mu","Isolation distribution",100,0,20);
  h_Iso_mu_FSR_= fs->make<TH1D>("h_iso_mu_FSR","Isolation distribution",100,0,20);
  irradiatecouter = 0;
  zmatchedcounter=0;
  numZMuTk =0;
  numZMuTkCut =0;
  numZMuTkMatched =0;
  numTk_FRS =0;
}

void ZMuMu_Radiative_analyzer::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateView> zMuMu;                //Collection of Z made by  Mu global + Mu global 
  Handle<GenParticleMatch> zMuMuMatchMap;     //Map of Z made by Mu global + Mu global with MC
  event.getByLabel(zMuMu_, zMuMu); 
  Handle<CandidateView> zMuTk;                //Collection of Z made by  Mu global + Track 
  Handle<GenParticleMatch> zMuTkMatchMap;
  event.getByLabel(zMuTk_, zMuTk); 
  cout << "         New Event"<<endl; 
  // ZMuMu
  if (zMuMu->size() > 0 ) {
    event.getByLabel(zMuMuMatchMap_, zMuMuMatchMap); 
     for(size_t i = 0; i < zMuMu->size(); ++i) { //loop on candidates
     
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
      
      GenParticleRef zMuMuMatch = (*zMuMuMatchMap)[zMuMuCandRef];
      if(zMuMuMatch.isNonnull()) {  // ZMuMu matched
	h_eta_->Fill(eta0);	  
	h_eta_->Fill(eta1);	  
	h_pt_->Fill(pt0);	  
	h_pt_->Fill(pt1);
	zmatchedcounter++;
	FSR = false;
	  
	//MonteCarlo Study
	const reco::GenParticle * muMc0 = mu0.genLepton();
	const reco::GenParticle * muMc1 = mu1.genLepton();
	const Candidate * motherMu0 =  muMc0->mother();
	const Candidate * motherMu1 =  muMc1->mother();
	int num_dau_muon0 = motherMu0->numberOfDaughters();
	int num_dau_muon1 = motherMu1->numberOfDaughters();
	cout<<"numero di figli muone0 = " << num_dau_muon0 <<endl;
	cout<<"numero di figli muone1 = " << num_dau_muon1 <<endl;
	if( num_dau_muon0 > 1 ){
	  for(int j = 0; j <  num_dau_muon0; ++j){
	    int id =motherMu0 ->daughter(j)->pdgId();
	    cout<<"dauther["<<j<<"] pdgId = "<<id<<endl; 
	    if(id == 22) {
	      irradiatecouter++;
	      break;
	      FSR=true;
	    }
	  }
	}//end check of gamma
	if( num_dau_muon1 > 1 ){
	  for(int j = 0; j <  num_dau_muon1; ++j){
	    int id = motherMu1->daughter(j)->pdgId(); 
	    cout<<"dauther["<<j<<"] pdgId = "<<id<<endl; 
	    if(id == 22) {
	      if(FSR)break;
	      else FSR=true;
	      irradiatecouter++;
	      break;
	    }
	  }
	}//end check of gamma
	if(FSR)h_zmass_FSR->Fill(zmass);
	else h_zmass_no_FSR->Fill(zmass);
      }// end MC match
     }// end loop on ZMuMu cand
  }// end if ZMuMu size > 0
  
  //ZMuTk  
  if (zMuTk->size() > 0 ) {
    event.getByLabel(zMuTkMatchMap_, zMuTkMatchMap); 
    for(size_t i = 0; i < zMuTk->size(); ++i) { //loop on candidates
      numZMuTk++;
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
	numZMuTkCut++;
	GenParticleRef zMuTkMatch = (*zMuTkMatchMap)[zMuTkCandRef];
	if(zMuTkMatch.isNonnull()) {  // ZMuTk matched
	  numZMuTkMatched++;
	  FSR_mu = false;
	  FSR_tk = false;
	  cout<<"          ZmuTk cuts && matched"<<endl;
	  //Isodeposit
	  const pat::IsoDeposit * muTrackIso =mu0.trackerIsoDeposit();
	  const pat::IsoDeposit * tkTrackIso =mu1.trackerIsoDeposit();
	  Direction muDir = Direction(mu0.eta(),mu0.phi());
	  Direction tkDir = Direction(mu1.eta(),mu1.phi());
	  
	  IsoDeposit::AbsVetos vetos_mu;
	  vetos_mu.push_back(new ConeVeto( muDir, dRVeto_ ));
	  vetos_mu.push_back(new ThresholdVeto( ptThreshold_ ));
	  
	  reco::IsoDeposit::AbsVetos vetos_tk;
	  vetos_tk.push_back(new ConeVeto( tkDir, dRVeto_ ));
	  vetos_tk.push_back(new ThresholdVeto( ptThreshold_ ));
	  
	  double  Tracker_isovalue_mu = muTrackIso->sumWithin(dRTrk_,vetos_mu);
	  double  Tracker_isovalue_tk = tkTrackIso->sumWithin(dRTrk_,vetos_tk);
	  
	  //MonteCarlo Study
	
	  const reco::GenParticle * muMc0 = mu0.genLepton();
	  const reco::GenParticle * muMc1 = mu1.genParticle() ;
	  const Candidate * motherMu0 =  muMc0->mother();
	  const Candidate * motherMu1 =  muMc1->mother();
	  int num_dau_muon0 = motherMu0->numberOfDaughters();
	  int num_dau_muon1 = motherMu1->numberOfDaughters();
	  cout<<"numero di figli muone0 = " << num_dau_muon0 <<endl;
	  cout<<"numero di figli muone1 = " << num_dau_muon1 <<endl; 
	
	  cout<<"         muon"<<endl;
	  cout<<"         num di daughters = "<< num_dau_muon0 <<endl;
	  if( num_dau_muon0 > 1 ){
	    for(int j = 0; j <  num_dau_muon0; ++j){
	      int id = motherMu0->daughter(j)->pdgId(); 
	      cout<<"         dau["<<j<<"] pdg ID = "<<id<<endl;
	      if(id == 22) {
	 	FSR_mu=true;
	      }
	    }
	  }//end check of gamma
	  else cout<<"         dau[0] pdg ID = "<<motherMu0->daughter(0)->pdgId()<<endl;
	  cout<<"         traccia"<<endl;
	  cout<<"         num di daughters = "<< num_dau_muon1 <<endl;
	  if( num_dau_muon1 > 1 ){
	    for(int j = 0; j <  num_dau_muon1; ++j){
	      int id = motherMu1->daughter(j)->pdgId(); 
	      cout<<"         dau["<<j<<"] pdg ID = "<<id<<endl;
	      if(id == 22) {
		FSR_tk=true;
		numTk_FRS++;
	      }
	    }
	  }//end check of gamma
	  else cout<<"         dau[0] pdg ID = "<<motherMu1->daughter(0)->pdgId()<<endl;
	  if(FSR_mu)h_Iso_mu_FSR_->Fill(Tracker_isovalue_mu);
	  else h_Iso_mu_->Fill( Tracker_isovalue_mu);
	  if(FSR_tk)h_Iso_tk_FSR_->Fill(Tracker_isovalue_tk);
	  else h_Iso_tk_->Fill( Tracker_isovalue_tk);
	}// end MC match
      }//end Kine-cuts
    }// end loop on ZMuTk cand
  }// end if ZMuTk size > 0
}// end analyze



void ZMuMu_Radiative_analyzer::endJob() {
  cout <<" Numero di ZMuMu matched dopo i tagli cinematici = "<< zmatchedcounter << endl;
  cout <<" Numero di ZMuMu di cui almeno un muone ha irradiato un gamma = "<<  irradiatecouter << endl;
  cout <<" Numero di ZMuTk  = "<<  numZMuTk << endl;
  cout <<" Numero di ZMuTk (dopo i tagli) = "<< numZMuTkCut << endl;
  cout <<" Numero di ZMuTk MC matched = "<<numZMuTkMatched << endl;
  cout <<" Numero di gamma associati a Tracce = "<< numTk_FRS << endl;


 }
  
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMu_Radiative_analyzer);
  
