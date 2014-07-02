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
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
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
#include "TH3.h"


using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;

class ZMuMu_Radiative_analyzer : public edm::EDAnalyzer {
public:
  ZMuMu_Radiative_analyzer(const edm::ParameterSet& pset);
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
  TH1D *h_zmass_FSR,*h_zmass_no_FSR;
  TH1D *h_zMuSamass_FSR,*h_zMuSamass_no_FSR;
  TH1D *h_zMuTkmass_FSR,*h_zMuTkmass_no_FSR;
  TH1D *h_Iso_,*h_Iso_FSR_ ;
  TH3D *h_Iso_3D_, *h_Iso_FSR_3D_;
  TH2D *h_staProbe_pt_eta_no_FSR_, *h_staProbe_pt_eta_FSR_, *h_ProbeOk_pt_eta_no_FSR_, *h_ProbeOk_pt_eta_FSR_;
  TH1D *h_trackProbe_eta_no_FSR, *h_trackProbe_pt_no_FSR, *h_staProbe_eta_no_FSR, *h_staProbe_pt_no_FSR, *h_ProbeOk_eta_no_FSR, *h_ProbeOk_pt_no_FSR;
  TH1D *h_trackProbe_eta_FSR, *h_trackProbe_pt_FSR, *h_staProbe_eta_FSR, *h_staProbe_pt_FSR, *h_ProbeOk_eta_FSR, *h_ProbeOk_pt_FSR;
  //boolean
  bool FSR_mu, FSR_tk, FSR_mu0, FSR_mu1;
  bool trig0found, trig1found;
  //counter
  int  zmmcounter , zmscounter, zmtcounter, evntcounter;
 };


typedef edm::ValueMap<float> IsolationCollection;

ZMuMu_Radiative_analyzer::ZMuMu_Radiative_analyzer(const ParameterSet& pset) :
  zMuMuToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuMu"))),
  zMuMuMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuMuMatchMap"))),
  zMuTkToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuTk"))),
  zMuTkMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuTkMatchMap"))),
  zMuSaToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuSa"))),
  zMuSaMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuSaMatchMap"))),
  dRVeto_(pset.getUntrackedParameter<double>("veto")),
  dRTrk_(pset.getUntrackedParameter<double>("deltaRTrk")),
  ptThreshold_(pset.getUntrackedParameter<double>("ptThreshold")){
  zmmcounter=0;
  zmscounter=0;
  zmtcounter=0;
  evntcounter=0;
  Service<TFileService> fs;

  // general histograms
  h_zmass_FSR= fs->make<TH1D>("h_zmass_FRS","Invariant Z mass distribution",200,0,200);
  h_zmass_no_FSR= fs->make<TH1D>("h_zmass_no_FSR","Invariant Z mass distribution",200,0,200);
  h_zMuSamass_FSR= fs->make<TH1D>("h_zMuSamass_FRS","Invariant Z mass distribution",200,0,200);
  h_zMuSamass_no_FSR= fs->make<TH1D>("h_zMuSamass_no_FSR","Invariant Z mass distribution",200,0,200);
  h_zMuTkmass_FSR= fs->make<TH1D>("h_zMuTkmass_FRS","Invariant Z mass distribution",200,0,200);
  h_zMuTkmass_no_FSR= fs->make<TH1D>("h_zMuTkmass_no_FSR","Invariant Z mass distribution",200,0,200);
  h_Iso_= fs->make<TH1D>("h_iso","Isolation distribution of muons without FSR",100,0,20);
  h_Iso_FSR_= fs->make<TH1D>("h_iso_FSR","Isolation distribution of muons with FSR ",100,0,20);
  h_Iso_3D_= fs->make<TH3D>("h_iso_3D","Isolation distribution of muons without FSR",100,20,100,100,-2.0,2.0,100,0,20);
  h_Iso_FSR_3D_= fs->make<TH3D>("h_iso_FSR_3D","Isolation distribution of muons with FSR ",100,20,100,100,-2.0,2.0,100,0,20);
  h_staProbe_pt_eta_no_FSR_= fs->make<TH2D>("h_staProbe_pt_eta_no_FSR","Pt vs Eta StandAlone without FSR ",100,20,100,100,-2.0,2.0);
  h_staProbe_pt_eta_FSR_= fs->make<TH2D>("h_staProbe_pt_eta_FSR","Pt vs Eta StandAlone with FSR ",100,20,100,100,-2.0,2.0);
  h_ProbeOk_pt_eta_no_FSR_= fs->make<TH2D>("h_ProbeOk_pt_eta_no_FSR","Pt vs Eta probeOk without FSR ",100,20,100,100,-2.0,2.0);
  h_ProbeOk_pt_eta_FSR_= fs->make<TH2D>("h_ProbeOk_pt_eta_FSR","Pt vs Eta probeOk with FSR ",100,20,100,100,-2.0,2.0);

  h_trackProbe_eta_no_FSR = fs->make<TH1D>("trackProbeEta_no_FSR","Eta of tracks",100,-2.0,2.0);
  h_trackProbe_pt_no_FSR = fs->make<TH1D>("trackProbePt_no_FSR","Pt of tracks",100,0,100);
  h_staProbe_eta_no_FSR = fs->make<TH1D>("standAloneProbeEta_no_FSR","Eta of standAlone",100,-2.0,2.0);
  h_staProbe_pt_no_FSR = fs->make<TH1D>("standAloneProbePt_no_FSR","Pt of standAlone",100,0,100);
  h_ProbeOk_eta_no_FSR = fs->make<TH1D>("probeOkEta_no_FSR","Eta of probe Ok",100,-2.0,2.0);
  h_ProbeOk_pt_no_FSR = fs->make<TH1D>("probeOkPt_no_FSR","Pt of probe ok",100,0,100);

  h_trackProbe_eta_FSR = fs->make<TH1D>("trackProbeEta_FSR","Eta of tracks",100,-2.0,2.0);
  h_trackProbe_pt_FSR  = fs->make<TH1D>("trackProbePt_FSR","Pt of tracks",100,0,100);
  h_staProbe_eta_FSR  = fs->make<TH1D>("standAloneProbeEta_FSR","Eta of standAlone",100,-2.0,2.0);
  h_staProbe_pt_FSR  = fs->make<TH1D>("standAloneProbePt_FSR","Pt of standAlone",100,0,100);
  h_ProbeOk_eta_FSR  = fs->make<TH1D>("probeOkEta_FSR","Eta of probe Ok",100,-2.0,2.0);
  h_ProbeOk_pt_FSR  = fs->make<TH1D>("probeOkPt_FSR","Pt of probe ok",100,0,100);



}

void ZMuMu_Radiative_analyzer::analyze(const Event& event, const EventSetup& setup) {
  evntcounter++;
  Handle<CandidateView> zMuMu;                //Collection of Z made by  Mu global + Mu global
  Handle<GenParticleMatch> zMuMuMatchMap;     //Map of Z made by Mu global + Mu global with MC
  event.getByToken(zMuMuToken_, zMuMu);
  Handle<CandidateView> zMuTk;                //Collection of Z made by  Mu global + Track
  Handle<GenParticleMatch> zMuTkMatchMap;
  event.getByToken(zMuTkToken_, zMuTk);
  Handle<CandidateView> zMuSa;                //Collection of Z made by  Mu global + Sa
  Handle<GenParticleMatch> zMuSaMatchMap;
  event.getByToken(zMuSaToken_, zMuSa);
  cout << "**********  New Event  ***********"<<endl;
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
	  cout<<"         Zmumu cuts && matched" <<endl;
	  FSR_mu0 = false;
	  FSR_mu1 = false;

	  //Isodeposit
	  const pat::IsoDeposit * mu0TrackIso =mu0.isoDeposit(pat::TrackIso);
	  const pat::IsoDeposit * mu1TrackIso =mu1.isoDeposit(pat::TrackIso);
	  Direction mu0Dir = Direction(mu0.eta(),mu0.phi());
	  Direction mu1Dir = Direction(mu1.eta(),mu1.phi());

	  reco::IsoDeposit::AbsVetos vetos_mu0;
	  vetos_mu0.push_back(new ConeVeto( mu0Dir, dRVeto_ ));
	  vetos_mu0.push_back(new ThresholdVeto( ptThreshold_ ));

	  reco::IsoDeposit::AbsVetos vetos_mu1;
	  vetos_mu1.push_back(new ConeVeto( mu1Dir, dRVeto_ ));
	  vetos_mu1.push_back(new ThresholdVeto( ptThreshold_ ));

	  double  Tracker_isovalue_mu0 = mu0TrackIso->sumWithin(dRTrk_,vetos_mu0);
	  double  Tracker_isovalue_mu1 = mu1TrackIso->sumWithin(dRTrk_,vetos_mu1);

	  //trigger study
	  trig0found = false;
	  trig1found = false;
	  const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	    mu0.triggerObjectMatchesByPath( "HLT_Mu9" );
	  const pat::TriggerObjectStandAloneCollection mu1HLTMatches =
	    mu1.triggerObjectMatchesByPath( "HLT_Mu9" );
	  if( mu0HLTMatches.size()>0 )
	    trig0found = true;
	  if( mu1HLTMatches.size()>0 )
	    trig1found = true;

	  //MonteCarlo Study
	  const reco::GenParticle * muMc0 = mu0.genLepton();
	  const reco::GenParticle * muMc1 = mu1.genLepton();
	  const Candidate * motherMu0 =  muMc0->mother();
	  int num_dau_muon0 = motherMu0->numberOfDaughters();
	  const Candidate * motherMu1 =  muMc1->mother();
	  int num_dau_muon1 = motherMu1->numberOfDaughters();
	  cout<<"         muone0"<<endl;
	  cout<<"         num di daughters = "<< num_dau_muon0 <<endl;
	  if( num_dau_muon0 > 1 ){
	    for(int j = 0; j <  num_dau_muon0; ++j){
	      int id =motherMu0 ->daughter(j)->pdgId();
	      cout<<"         dauther["<<j<<"] pdgId = "<<id<<endl;
	      if(id == 22) FSR_mu0=true;
	    }
	  }//end check of gamma

	  cout<<"         muone1"<<endl;
	  cout<<"         num di daughters = "<< num_dau_muon1 <<endl;
	  if( num_dau_muon1 > 1 ){
	    for(int j = 0; j <  num_dau_muon1; ++j){
	      int id = motherMu1->daughter(j)->pdgId();
	      cout<<"         dauther["<<j<<"] pdgId = "<<id<<endl;
	      if(id == 22) FSR_mu1=true;
	    }
	  }//end check of gamma

	  if(FSR_mu0 || FSR_mu1 )h_zmass_FSR->Fill(zmass);
	  else h_zmass_no_FSR->Fill(zmass);

	  if (trig1found) {       // check efficiency of muon0 not imposing the trigger on it
	    cout<<"muon 1 is triggered "<<endl;
	    if(FSR_mu0){
	      cout<< "and muon 0 does FSR"<<endl;
	      h_trackProbe_eta_FSR->Fill(eta0);
	      h_trackProbe_pt_FSR->Fill(pt0);
	      h_staProbe_eta_FSR->Fill(eta0);
	      h_staProbe_pt_FSR->Fill(pt0);
	      h_staProbe_pt_eta_FSR_->Fill(pt0,eta0);
	      h_ProbeOk_eta_FSR->Fill(eta0);
	      h_ProbeOk_pt_FSR->Fill(pt0);
	      h_ProbeOk_pt_eta_FSR_->Fill(pt0,eta0);
	    }else{
	      cout<<"and muon 0 doesn't FSR"<<endl;
	      h_trackProbe_eta_no_FSR->Fill(eta0);
	      h_trackProbe_pt_no_FSR->Fill(pt0);
	      h_staProbe_eta_no_FSR->Fill(eta0);
	      h_staProbe_pt_no_FSR->Fill(pt0);
	      h_staProbe_pt_eta_no_FSR_->Fill(pt0,eta0);
	      h_ProbeOk_eta_no_FSR->Fill(eta0);
	      h_ProbeOk_pt_no_FSR->Fill(pt0);
	      h_ProbeOk_pt_eta_no_FSR_->Fill(pt0,eta0);
	    }
	    if(FSR_mu0){
	      h_Iso_FSR_->Fill(Tracker_isovalue_mu0);
	      h_Iso_FSR_3D_->Fill(pt0,eta0,Tracker_isovalue_mu0);
	    }
	    else{
	      h_Iso_->Fill(Tracker_isovalue_mu0);
	      h_Iso_3D_->Fill(pt0,eta0,Tracker_isovalue_mu0);
	    }
	  }
	  if (trig0found) {         // check efficiency of muon1 not imposing the trigger on it
	    cout<<"muon 0 is triggered"<<endl;
	    if(FSR_mu1){
	      cout<<"and muon 1  does FSR"<<endl;
	      h_trackProbe_eta_FSR->Fill(eta1);
	      h_staProbe_eta_FSR->Fill(eta1);
	      h_trackProbe_pt_FSR->Fill(pt1);
	      h_staProbe_pt_FSR->Fill(pt1);
	      h_ProbeOk_eta_FSR->Fill(eta1);
	      h_ProbeOk_pt_FSR->Fill(pt1);
	      h_staProbe_pt_eta_FSR_->Fill(pt1,eta1);
	      h_ProbeOk_pt_eta_FSR_->Fill(pt1,eta1);

	    }else{
	      cout<<"and muon 1 doesn't FSR"<<endl;
	      h_trackProbe_eta_no_FSR->Fill(eta1);
	      h_staProbe_eta_no_FSR->Fill(eta1);
	      h_trackProbe_pt_no_FSR->Fill(pt1);
	      h_staProbe_pt_no_FSR->Fill(pt1);
	      h_ProbeOk_eta_no_FSR->Fill(eta1);
	      h_ProbeOk_pt_no_FSR->Fill(pt1);
	      h_staProbe_pt_eta_no_FSR_->Fill(pt1,eta1);
	      h_ProbeOk_pt_eta_no_FSR_->Fill(pt1,eta1);


	    }
	    if(FSR_mu1){
	      h_Iso_FSR_->Fill(Tracker_isovalue_mu1);
	      h_Iso_FSR_3D_->Fill(pt1,eta1,Tracker_isovalue_mu1);
	    }else{
	      h_Iso_->Fill(Tracker_isovalue_mu1);
	      h_Iso_3D_->Fill(pt1,eta1,Tracker_isovalue_mu1);
	    }
	  }
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
      const Candidate *  lep0 =zMuSaCand.daughter(0);
      const Candidate *  lep1 =zMuSaCand.daughter(1);
      CandidateBaseRef dau0 = lep0->masterClone();
      CandidateBaseRef dau1 = lep1->masterClone();
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
	  cout<<"         Zmusa cuts && matched" <<endl;
	  FSR_mu0 = false;
	  FSR_mu1 = false;
	  zmscounter++;
	  //Isodeposit
	  const pat::IsoDeposit * mu0TrackIso =mu0.isoDeposit(pat::TrackIso);
	  const pat::IsoDeposit * mu1TrackIso =mu1.isoDeposit(pat::TrackIso);
	  Direction mu0Dir = Direction(mu0.eta(),mu0.phi());
	  Direction mu1Dir = Direction(mu1.eta(),mu1.phi());

	  reco::IsoDeposit::AbsVetos vetos_mu0;
	  vetos_mu0.push_back(new ConeVeto( mu0Dir, dRVeto_ ));
	  vetos_mu0.push_back(new ThresholdVeto( ptThreshold_ ));

	  reco::IsoDeposit::AbsVetos vetos_mu1;
	  vetos_mu1.push_back(new ConeVeto( mu1Dir, dRVeto_ ));
	  vetos_mu1.push_back(new ThresholdVeto( ptThreshold_ ));

	  double  Tracker_isovalue_mu0 = mu0TrackIso->sumWithin(dRTrk_,vetos_mu0);
	  double  Tracker_isovalue_mu1 = mu1TrackIso->sumWithin(dRTrk_,vetos_mu1);

	  // HLT match (check just dau0 the global)
	  const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	    mu0.triggerObjectMatchesByPath( "HLT_Mu9" );
	  const pat::TriggerObjectStandAloneCollection mu1HLTMatches =
	    mu1.triggerObjectMatchesByPath( "HLT_Mu9" );
	  trig0found = false;
	  trig1found = false;
	  if( mu0HLTMatches.size()>0 )
	    trig0found = true;
	  if( mu1HLTMatches.size()>0 )
	    trig1found = true;

	  //MonteCarlo Study
	  const reco::GenParticle * muMc0 = mu0.genLepton();
	  const reco::GenParticle * muMc1 = mu1.genLepton();
	  const Candidate * motherMu0 =  muMc0->mother();
	  const Candidate * motherMu1 =  muMc1->mother();
	  int num_dau_muon0 = motherMu0->numberOfDaughters();
	  int num_dau_muon1 = motherMu1->numberOfDaughters();
	  cout<<"         muone0"<<endl;
	  cout<<"         num di daughters = "<< num_dau_muon0 <<endl;
	  if( num_dau_muon0 > 1 ){
	    for(int j = 0; j <  num_dau_muon0; ++j){
	      int id =motherMu0 ->daughter(j)->pdgId();
	      cout<<"         dauther["<<j<<"] pdgId = "<<id<<endl;
	      if(id == 22) FSR_mu0=true;
	    }
	  }//end check of gamma

	  cout<<"         muone1"<<endl;
	  cout<<"         num di daughters = "<< num_dau_muon1 <<endl;
	  if( num_dau_muon1 > 1 ){
	    for(int j = 0; j <  num_dau_muon1; ++j){
	      int id = motherMu1->daughter(j)->pdgId();
	      cout<<"         dauther["<<j<<"] pdgId = "<<id<<endl;
	      if(id == 22) FSR_mu1=true;
	    }
	  }//end check of gamma
	  if(FSR_mu0 || FSR_mu1 )h_zMuSamass_FSR->Fill(zmass);
	  else h_zMuSamass_no_FSR->Fill(zmass);
	  if(lep0->isGlobalMuon() && trig0found){
	    if(FSR_mu1){
	      h_staProbe_eta_FSR->Fill(eta1);
	      h_staProbe_pt_FSR->Fill(pt1);
	      h_staProbe_pt_eta_FSR_->Fill(pt1,eta1);

	    }else{
	      h_staProbe_eta_no_FSR->Fill(eta1);
	      h_staProbe_pt_no_FSR->Fill(pt1);
	      h_staProbe_pt_eta_no_FSR_->Fill(pt1,eta1);

	    }
	    if(FSR_mu1){
	      h_Iso_FSR_->Fill(Tracker_isovalue_mu1);
	      h_Iso_FSR_3D_->Fill(pt1,eta1,Tracker_isovalue_mu1);
	    }
	    else{
	      h_Iso_->Fill(Tracker_isovalue_mu1);
	      h_Iso_3D_->Fill(pt1,eta1,Tracker_isovalue_mu1);
	    }
	  }
	  if(lep1->isGlobalMuon() && trig1found){
	    if(FSR_mu0){
	      h_staProbe_eta_FSR->Fill(eta0);
	      h_staProbe_pt_FSR->Fill(pt0);
	      h_staProbe_pt_eta_FSR_->Fill(pt0,eta0);

	    }else{
	      h_staProbe_eta_no_FSR->Fill(eta0);
	      h_staProbe_pt_no_FSR->Fill(pt0);
	      h_staProbe_pt_eta_FSR_->Fill(pt0,eta0);

	    }
	    if(FSR_mu0){
	      h_Iso_FSR_->Fill(Tracker_isovalue_mu0);
	      h_Iso_FSR_3D_->Fill(pt0,eta0,Tracker_isovalue_mu0);
	    }
	    else{
	      h_Iso_->Fill(Tracker_isovalue_mu0);
	      h_Iso_3D_->Fill(pt0,eta0,Tracker_isovalue_mu0);
	    }
	  }
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
	  cout<<"          ZmuTk cuts && matched"<<endl;
	  zmtcounter++;
	  //Isodeposit
	  const pat::IsoDeposit * muTrackIso =mu0.isoDeposit(pat::TrackIso);
	  const pat::IsoDeposit * tkTrackIso =mu1.isoDeposit(pat::TrackIso);
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
	      }
	    }
	  }//end check of gamma
	  else cout<<"         dau[0] pdg ID = "<<motherMu1->daughter(0)->pdgId()<<endl;
	  cout<<"Mu Isolation = "<< Tracker_isovalue_mu <<endl;
	  cout<<"Track Isolation = "<< Tracker_isovalue_tk <<endl;
	  if(FSR_mu){
	    h_Iso_FSR_->Fill(Tracker_isovalue_mu);
	    h_Iso_FSR_3D_->Fill(pt0,eta0,Tracker_isovalue_mu);
	  }
	  else{
	    h_Iso_->Fill( Tracker_isovalue_mu);
	    h_Iso_3D_->Fill(pt0,eta0,Tracker_isovalue_mu);

	  }
	  if(FSR_tk){
	    h_Iso_FSR_->Fill(Tracker_isovalue_tk);
	    h_Iso_FSR_3D_->Fill(pt1,eta1,Tracker_isovalue_tk);
	    h_trackProbe_eta_FSR->Fill(eta1);
	    h_trackProbe_pt_FSR->Fill(pt1);
 	  }
	  else{
	    h_Iso_->Fill( Tracker_isovalue_tk);
	    h_Iso_3D_->Fill(pt1,eta1,Tracker_isovalue_tk);
	    h_trackProbe_eta_no_FSR->Fill(eta1);
	    h_trackProbe_pt_no_FSR->Fill(pt1);
	  }
	}// end MC match
      }//end Kine-cuts
    }// end loop on ZMuTk cand
  }// end if ZMuTk size > 0
}// end analyze



void ZMuMu_Radiative_analyzer::endJob() {
  cout<<" ============= Summary =========="<<endl;
  cout <<" Numero di eventi "<< evntcounter << endl;
  cout <<" 1)Numero di ZMuMu matched dopo i tagli cinematici = "<< zmmcounter << endl;
  cout <<" 2)Numero di ZMuSa matched dopo i tagli cinematici = "<< zmscounter << endl;
  cout <<" 3)Numero di ZMuTk matched dopo i tagli cinematici = "<< zmtcounter << endl;
  double n1= h_Iso_FSR_->Integral();
  double icut1= h_Iso_FSR_->Integral(0,15);
  double eff_iso_FSR = (double)icut1/(double)n1;
  double err_iso_FSR = sqrt(eff_iso_FSR*(1-eff_iso_FSR)/n1);
  double n2= h_Iso_->Integral();
  double icut2= h_Iso_->Integral(0,15);
  double eff_iso= (double)icut2/(double)n2;
  double err_iso = sqrt(eff_iso*(1-eff_iso)/n2);
  cout<<" ============= Isolation Efficiecy =========="<<endl;
  cout<<"Isolation Efficiency  = "<< eff_iso <<" +/- "<< err_iso <<endl;
  cout<<"Isolation Efficiency with FSR = "<< eff_iso_FSR <<" +/- "<< err_iso_FSR <<endl;

 }

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMu_Radiative_analyzer);

