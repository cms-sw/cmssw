/* \class ZMuMu_MCanalyzer
 *
 * author: Davide Piccolo
 *
 * ZMuMu MC analyzer:
 * check muon reco efficiencies from MC truth,
 *
 */

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include <vector>

using namespace edm;
using namespace std;
using namespace reco;
using namespace reco;
using namespace isodeposit;

typedef ValueMap<float> IsolationCollection;

class ZMuMu_MCanalyzer : public edm::EDAnalyzer {
public:
  ZMuMu_MCanalyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  bool check_ifZmumu(const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2);
  float getParticlePt(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2);
  float getParticleEta(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2);
  float getParticlePhi(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2);
  Particle::LorentzVector getParticleP4(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2);
  virtual void endJob() override;

  EDGetTokenT<CandidateView> zMuMuToken_;
  EDGetTokenT<GenParticleMatch> zMuMuMatchMapToken_;
  EDGetTokenT<CandidateView> zMuStandAloneToken_;
  EDGetTokenT<GenParticleMatch> zMuStandAloneMatchMapToken_;
  EDGetTokenT<CandidateView> zMuTrackToken_;
  EDGetTokenT<GenParticleMatch> zMuTrackMatchMapToken_;
  EDGetTokenT<CandidateView> muonsToken_;
  EDGetTokenT<CandidateView> tracksToken_;
  EDGetTokenT<GenParticleCollection> genParticlesToken_;

  bool bothMuons_;

  double etamin_, etamax_, ptmin_, massMin_, massMax_, isoMax_;

  double ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_;

  bool relativeIsolation_;
  string hltPath_;
  reco::CandidateBaseRef globalMuonCandRef_, trackMuonCandRef_, standAloneMuonCandRef_;
  OverlapChecker overlap_;

  // general histograms
  TH1D *h_trackProbe_eta, *h_trackProbe_pt, *h_staProbe_eta, *h_staProbe_pt, *h_ProbeOk_eta, *h_ProbeOk_pt;

  // global counters
  int nGlobalMuonsMatched_passed;    // total number of global muons MC matched and passing cuts (and triggered)
  int nGlobalMuonsMatched_passedIso;    // total number of global muons MC matched and passing cuts including Iso
  int n2GlobalMuonsMatched_passedIso;    // total number of Z->2 global muons MC matched and passing cuts including Iso
  int nStaMuonsMatched_passedIso;       // total number of sta only muons MC matched and passing cuts including Iso
  int nTracksMuonsMatched_passedIso;    // total number of tracks only muons MC matched and passing cuts including Iso
  int n2GlobalMuonsMatched_passedIso2Trg;    // total number of Z->2 global muons MC matched and passing cuts including Iso and both triggered
  int nMu0onlyTriggered;               // n. of events zMuMu with mu0 only triggered
  int nMu1onlyTriggered;               // n. of events zMuMu with mu1 only triggered

  int nZMuMu_matched;               // n. of events zMuMu macthed
  int nZMuSta_matched;              // n  of events zMuSta macthed
  int nZMuTrk_matched;              // n. of events zMuTrk mathced
};


template<typename T>
double isolation(const T * t, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta, bool relativeIsolation) {
  // on 34X:
  const pat::IsoDeposit * trkIso = t->isoDeposit(pat::TrackIso);
  //  const pat::IsoDeposit * trkIso = t->trackerIsoDeposit();
  // on 34X
  const pat::IsoDeposit * ecalIso = t->isoDeposit(pat::EcalIso);
  //  const pat::IsoDeposit * ecalIso = t->ecalIsoDeposit();
  //    on 34X
  const pat::IsoDeposit * hcalIso = t->isoDeposit(pat::HcalIso);
  //    const pat::IsoDeposit * hcalIso = t->hcalIsoDeposit();

  Direction dir = Direction(t->eta(), t->phi());

  pat::IsoDeposit::AbsVetos vetosTrk;
  vetosTrk.push_back(new ConeVeto( dir, dRVetoTrk ));
  vetosTrk.push_back(new ThresholdVeto( ptThreshold ));

  pat::IsoDeposit::AbsVetos vetosEcal;
  vetosEcal.push_back(new ConeVeto( dir, 0.));
  vetosEcal.push_back(new ThresholdVeto( etEcalThreshold ));

  pat::IsoDeposit::AbsVetos vetosHcal;
  vetosHcal.push_back(new ConeVeto( dir, 0. ));
  vetosHcal.push_back(new ThresholdVeto( etHcalThreshold ));

  double isovalueTrk = (trkIso->sumWithin(dRTrk,vetosTrk));
  double isovalueEcal = (ecalIso->sumWithin(dREcal,vetosEcal));
  double isovalueHcal = (hcalIso->sumWithin(dRHcal,vetosHcal));


  double iso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk) ;
  if(relativeIsolation) iso /= t->pt();
  return iso;
}


double candidateIsolation( const reco::Candidate* c, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta, bool relativeIsolation) {
  const pat::Muon * mu = dynamic_cast<const pat::Muon *>(&*c->masterClone());
  if(mu != 0) return isolation(mu, ptThreshold, etEcalThreshold, etHcalThreshold ,dRVetoTrk, dRTrk, dREcal , dRHcal,  alpha, beta, relativeIsolation);
  const pat::GenericParticle * trk = dynamic_cast<const pat::GenericParticle*>(&*c->masterClone());
  if(trk != 0) return isolation(trk,  ptThreshold, etEcalThreshold, etHcalThreshold ,dRVetoTrk, dRTrk, dREcal ,
				dRHcal,  alpha, beta, relativeIsolation);
  throw edm::Exception(edm::errors::InvalidReference)
    << "Candidate daughter #0 is neither pat::Muons nor pat::GenericParticle\n";
  return -1;
}




#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <iostream>
#include <iterator>
#include <cmath>

ZMuMu_MCanalyzer::ZMuMu_MCanalyzer(const ParameterSet& pset) :
  zMuMuToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuMu"))),
  zMuMuMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuMuMatchMap"))),
  zMuStandAloneToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuStandAlone"))),
  zMuStandAloneMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuStandAloneMatchMap"))),
  zMuTrackToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuTrack"))),
  zMuTrackMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuTrackMatchMap"))),
  muonsToken_(consumes<CandidateView>(pset.getParameter<InputTag>("muons"))),
  tracksToken_(consumes<CandidateView>(pset.getParameter<InputTag>("tracks"))),
  genParticlesToken_(consumes<GenParticleCollection>(pset.getParameter<InputTag>("genParticles"))),

  bothMuons_(pset.getParameter<bool>("bothMuons")),
  etamin_(pset.getUntrackedParameter<double>("etamin")),
  etamax_(pset.getUntrackedParameter<double>("etamax")),
  ptmin_(pset.getUntrackedParameter<double>("ptmin")),
  massMin_(pset.getUntrackedParameter<double>("zMassMin")),
  massMax_(pset.getUntrackedParameter<double>("zMassMax")),
  isoMax_(pset.getUntrackedParameter<double>("isomax")),
  ptThreshold_(pset.getUntrackedParameter<double>("ptThreshold")),
  etEcalThreshold_(pset.getUntrackedParameter<double>("etEcalThreshold")),
  etHcalThreshold_(pset.getUntrackedParameter<double>("etHcalThreshold")),
  dRVetoTrk_(pset.getUntrackedParameter<double>("deltaRVetoTrk")),
  dRTrk_(pset.getUntrackedParameter<double>("deltaRTrk")),
  dREcal_(pset.getUntrackedParameter<double>("deltaREcal")),
  dRHcal_(pset.getUntrackedParameter<double>("deltaRHcal")),
  alpha_(pset.getUntrackedParameter<double>("alpha")),
  beta_(pset.getUntrackedParameter<double>("beta")),
  relativeIsolation_(pset.getUntrackedParameter<bool>("relativeIsolation")),
  hltPath_(pset.getUntrackedParameter<std::string >("hltPath")) {
  Service<TFileService> fs;

  // binning of entries array (at moment defined by hand and not in cfg file)
  double etaRange[8] = {-2.5,-2.,-1.2,-0.8,0.8,1.2,2.,2.5};
  double ptRange[4] = {20.,40.,60.,100.};

  // general histograms
  h_trackProbe_eta = fs->make<TH1D>("trackProbeEta","Eta of tracks",7,etaRange);
  h_trackProbe_pt = fs->make<TH1D>("trackProbePt","Pt of tracks",3,ptRange);
  h_staProbe_eta = fs->make<TH1D>("standAloneProbeEta","Eta of standAlone",7,etaRange);
  h_staProbe_pt = fs->make<TH1D>("standAloneProbePt","Pt of standAlone",3,ptRange);
  h_ProbeOk_eta = fs->make<TH1D>("probeOkEta","Eta of probe Ok",7,etaRange);
  h_ProbeOk_pt = fs->make<TH1D>("probeOkPt","Pt of probe ok",3,ptRange);

  // clear global counters
  nGlobalMuonsMatched_passed = 0;
  nGlobalMuonsMatched_passedIso = 0;
  n2GlobalMuonsMatched_passedIso = 0;
  nStaMuonsMatched_passedIso = 0;
  nTracksMuonsMatched_passedIso = 0;
  n2GlobalMuonsMatched_passedIso2Trg = 0;
  nMu0onlyTriggered = 0;
  nMu1onlyTriggered = 0;
  nZMuMu_matched = 0;
  nZMuSta_matched = 0;
  nZMuTrk_matched = 0;
}

void ZMuMu_MCanalyzer::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateView> zMuMu;
  Handle<GenParticleMatch> zMuMuMatchMap; //Map of Z made by Mu global + Mu global
  Handle<CandidateView> zMuStandAlone;
  Handle<GenParticleMatch> zMuStandAloneMatchMap; //Map of Z made by Mu + StandAlone
  Handle<CandidateView> zMuTrack;
  Handle<GenParticleMatch> zMuTrackMatchMap; //Map of Z made by Mu + Track
  Handle<CandidateView> muons; //Collection of Muons
  Handle<CandidateView> tracks; //Collection of Tracks

  Handle<GenParticleCollection> genParticles;  // Collection of Generatd Particles

  event.getByToken(zMuMuToken_, zMuMu);
  event.getByToken(zMuStandAloneToken_, zMuStandAlone);
  event.getByToken(zMuTrackToken_, zMuTrack);
  event.getByToken(genParticlesToken_, genParticles);
  event.getByToken(muonsToken_, muons);
  event.getByToken(tracksToken_, tracks);

  /*
  cout << "*********  zMuMu         size : " << zMuMu->size() << endl;
  cout << "*********  zMuStandAlone size : " << zMuStandAlone->size() << endl;
  cout << "*********  zMuTrack      size : " << zMuTrack->size() << endl;
  cout << "*********  muons         size : " << muons->size() << endl;
  cout << "*********  tracks        size : " << tracks->size() << endl;
  */
  //      std::cout<<"Run-> "<<event.id().run()<<std::endl;
  //      std::cout<<"Event-> "<<event.id().event()<<std::endl;


  bool zMuMu_found = false;

  // loop on ZMuMu
  if (zMuMu->size() > 0 ) {
    event.getByToken(zMuMuMatchMapToken_, zMuMuMatchMap);
    for(unsigned int i = 0; i < zMuMu->size(); ++i) { //loop on candidates
      const Candidate & zMuMuCand = (*zMuMu)[i]; //the candidate
      CandidateBaseRef zMuMuCandRef = zMuMu->refAt(i);

      const Candidate * lep0 = zMuMuCand.daughter( 0 );
      const Candidate * lep1 = zMuMuCand.daughter( 1 );
      const pat::Muon & muonDau0 = dynamic_cast<const pat::Muon &>(*lep0->masterClone());
      //      double trkiso0 = muonDau0.trackIso();
      const pat::Muon & muonDau1 = dynamic_cast<const pat::Muon &>(*lep1->masterClone());
      //double trkiso1 = muonDau1.trackIso();

      double iso0 = candidateIsolation(lep0,ptThreshold_, etEcalThreshold_, etHcalThreshold_,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_, relativeIsolation_);
      double iso1 = candidateIsolation(lep1,ptThreshold_, etEcalThreshold_, etHcalThreshold_,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_, relativeIsolation_);

      double pt0 = zMuMuCand.daughter(0)->pt();
      double pt1 = zMuMuCand.daughter(1)->pt();
      double eta0 = zMuMuCand.daughter(0)->eta();
      double eta1 = zMuMuCand.daughter(1)->eta();
      double mass = zMuMuCand.mass();

      // HLT match
      const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	muonDau0.triggerObjectMatchesByPath( hltPath_ );
      const pat::TriggerObjectStandAloneCollection mu1HLTMatches =
	muonDau1.triggerObjectMatchesByPath( hltPath_ );

      bool trig0found = false;
      bool trig1found = false;
      if( mu0HLTMatches.size()>0 )
	trig0found = true;
      if( mu1HLTMatches.size()>0 )
	trig1found = true;

      GenParticleRef zMuMuMatch = (*zMuMuMatchMap)[zMuMuCandRef];
      if(zMuMuMatch.isNonnull()) {  // ZMuMu matched
	zMuMu_found = true;
	nZMuMu_matched++;
	if (pt0>ptmin_ && pt1>ptmin_ && abs(eta0)>etamin_ && abs(eta1) >etamin_ && abs(eta0)<etamax_ && abs(eta1) <etamax_ && mass >massMin_ && mass < massMax_ && (trig0found || trig1found)) { // kinematic and trigger cuts passed
	  nGlobalMuonsMatched_passed++; // first global Muon passed kine cuts
	  nGlobalMuonsMatched_passed++; // second global muon passsed kine cuts
	  if (iso0<isoMax_) nGlobalMuonsMatched_passedIso++;       // first global muon passed the iso cut
	  if (iso1<isoMax_) nGlobalMuonsMatched_passedIso++;       // second global muon passed the iso cut
	  if (iso0<isoMax_ && iso1<isoMax_) {
	    n2GlobalMuonsMatched_passedIso++;  // both muons passed iso cut
	    if (trig0found && trig1found) n2GlobalMuonsMatched_passedIso2Trg++;  // both muons have HLT
	    if (trig0found && !trig1found) nMu0onlyTriggered++;
	    if (trig1found && !trig0found) nMu1onlyTriggered++;
	    // histograms vs eta and pt
	    if (trig1found) {         // check efficiency of muon0 not imposing the trigger on it
	      h_trackProbe_eta->Fill(eta0);
	      h_trackProbe_pt->Fill(pt0);
	      h_staProbe_eta->Fill(eta0);
	      h_staProbe_pt->Fill(pt0);
	      h_ProbeOk_eta->Fill(eta0);
	      h_ProbeOk_pt->Fill(pt0);
	    }
	    if (trig0found) {         // check efficiency of muon1 not imposing the trigger on it
	      h_trackProbe_eta->Fill(eta1);
	      h_staProbe_eta->Fill(eta1);
	      h_trackProbe_pt->Fill(pt1);
	      h_staProbe_pt->Fill(pt1);
	      h_ProbeOk_eta->Fill(eta1);
	      h_ProbeOk_pt->Fill(pt1);
	    }
	  }
	}
      } // end MC match

    }  // end loop on ZMuMu cand
  }    // end if ZMuMu size > 0

  // loop on ZMuSta
  bool zMuSta_found = false;
  if (!zMuMu_found && zMuStandAlone->size() > 0 ) {
    event.getByToken(zMuStandAloneMatchMapToken_, zMuStandAloneMatchMap);
    for(unsigned int i = 0; i < zMuStandAlone->size(); ++i) { //loop on candidates
      const Candidate & zMuStandAloneCand = (*zMuStandAlone)[i]; //the candidate
      CandidateBaseRef zMuStandAloneCandRef = zMuStandAlone->refAt(i);
      GenParticleRef zMuStandAloneMatch = (*zMuStandAloneMatchMap)[zMuStandAloneCandRef];

      const Candidate * lep0 = zMuStandAloneCand.daughter( 0 );
      const Candidate * lep1 = zMuStandAloneCand.daughter( 1 );
      const pat::Muon & muonDau0 = dynamic_cast<const pat::Muon &>(*lep0->masterClone());
      //double trkiso0 = muonDau0.trackIso();
      //      const pat::Muon & muonDau1 = dynamic_cast<const pat::Muon &>(*lep1->masterClone());
      //double trkiso1 = muonDau1.trackIso();

      double iso0 = candidateIsolation(lep0,ptThreshold_, etEcalThreshold_, etHcalThreshold_,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_, relativeIsolation_);
      double iso1 = candidateIsolation(lep1,ptThreshold_, etEcalThreshold_, etHcalThreshold_,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_, relativeIsolation_);

      double pt0 = zMuStandAloneCand.daughter(0)->pt();
      double pt1 = zMuStandAloneCand.daughter(1)->pt();
      double eta0 = zMuStandAloneCand.daughter(0)->eta();
      double eta1 = zMuStandAloneCand.daughter(1)->eta();
      double mass = zMuStandAloneCand.mass();

      // HLT match (check just dau0 the global)
      const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	muonDau0.triggerObjectMatchesByPath( hltPath_ );

      bool trig0found = false;
      if( mu0HLTMatches.size()>0 )
	trig0found = true;

      if(zMuStandAloneMatch.isNonnull()) {  // ZMuStandAlone matched
	zMuSta_found = true;
	nZMuSta_matched++;
	if (pt0>ptmin_ && pt1>ptmin_ && abs(eta0)>etamin_ && abs(eta1)>etamin_ && abs(eta0)<etamax_ && abs(eta1) <etamax_ && mass >massMin_ &&
	    mass < massMax_ && iso0<isoMax_ && iso1 < isoMax_ && trig0found) { // all cuts and trigger passed
	  nStaMuonsMatched_passedIso++;
	  // histograms vs eta and pt
	  h_staProbe_eta->Fill(eta1);
	  h_staProbe_pt->Fill(pt1);
	}
      } // end MC match
    }  // end loop on ZMuStandAlone cand
  }    // end if ZMuStandAlone size > 0


  // loop on ZMuTrack
  if (!zMuMu_found && !zMuSta_found && zMuTrack->size() > 0 ) {
    event.getByToken(zMuTrackMatchMapToken_, zMuTrackMatchMap);
    for(unsigned int i = 0; i < zMuTrack->size(); ++i) { //loop on candidates
      const Candidate & zMuTrackCand = (*zMuTrack)[i]; //the candidate
      CandidateBaseRef zMuTrackCandRef = zMuTrack->refAt(i);
      const Candidate * lep0 = zMuTrackCand.daughter( 0 );
      const Candidate * lep1 = zMuTrackCand.daughter( 1 );
      const pat::Muon & muonDau0 = dynamic_cast<const pat::Muon &>(*lep0->masterClone());
      //double trkiso0 = muonDau0.trackIso();
      //const pat::GenericParticle & trackDau1 = dynamic_cast<const pat::GenericParticle &>(*lep1->masterClone());
      //double trkiso1 = trackDau1.trackIso();
      double iso0 = candidateIsolation(lep0,ptThreshold_, etEcalThreshold_, etHcalThreshold_,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_, relativeIsolation_);
      double iso1 = candidateIsolation(lep1,ptThreshold_, etEcalThreshold_, etHcalThreshold_,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_, relativeIsolation_);


      double pt0 = zMuTrackCand.daughter(0)->pt();
      double pt1 = zMuTrackCand.daughter(1)->pt();
      double eta0 = zMuTrackCand.daughter(0)->eta();
      double eta1 = zMuTrackCand.daughter(1)->eta();
      double mass = zMuTrackCand.mass();

      // HLT match (check just dau0 the global)
      const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	muonDau0.triggerObjectMatchesByPath( hltPath_ );

      bool trig0found = false;
      if( mu0HLTMatches.size()>0 )
	trig0found = true;

      GenParticleRef zMuTrackMatch = (*zMuTrackMatchMap)[zMuTrackCandRef];
      if(zMuTrackMatch.isNonnull()) {  // ZMuTrack matched
	nZMuTrk_matched++;
	if (pt0>ptmin_ && pt1>ptmin_ && abs(eta0)>etamin_ && abs(eta1)>etamin_ && abs(eta0)<etamax_ && abs(eta1) <etamax_ && mass >massMin_ &&
	    mass < massMax_ && iso0<isoMax_ && iso1 < isoMax_ && trig0found) { // all cuts and trigger passed
	  nTracksMuonsMatched_passedIso++;
	  // histograms vs eta and pt
	  h_trackProbe_eta->Fill(eta1);
	  h_trackProbe_pt->Fill(pt1);
	}
      }  // end MC match
    }  // end loop on ZMuTrack cand
  }    // end if ZMuTrack size > 0

}       // end analyze

bool ZMuMu_MCanalyzer::check_ifZmumu(const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
{
  int partId0 = dauGen0->pdgId();
  int partId1 = dauGen1->pdgId();
  int partId2 = dauGen2->pdgId();
  bool muplusFound=false;
  bool muminusFound=false;
  bool ZFound=false;
  if (partId0==13 || partId1==13 || partId2==13) muminusFound=true;
  if (partId0==-13 || partId1==-13 || partId2==-13) muplusFound=true;
  if (partId0==23 || partId1==23 || partId2==23) ZFound=true;
  return muplusFound*muminusFound*ZFound;
}

float ZMuMu_MCanalyzer::getParticlePt(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
{
  int partId0 = dauGen0->pdgId();
  int partId1 = dauGen1->pdgId();
  int partId2 = dauGen2->pdgId();
  float ptpart=0.;
  if (partId0 == ipart) {
    for(unsigned int k = 0; k < dauGen0->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen0->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	ptpart = dauMuGen->pt();
      }
    }
  }
  if (partId1 == ipart) {
    for(unsigned int k = 0; k < dauGen1->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen1->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	ptpart = dauMuGen->pt();
      }
    }
  }
  if (partId2 == ipart) {
    for(unsigned int k = 0; k < dauGen2->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen2->daughter(k);
      if(abs(dauMuGen->pdgId()) == ipart && dauMuGen->status() ==1) {
	ptpart = dauMuGen->pt();
      }
    }
  }
  return ptpart;
}

float ZMuMu_MCanalyzer::getParticleEta(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
{
  int partId0 = dauGen0->pdgId();
  int partId1 = dauGen1->pdgId();
  int partId2 = dauGen2->pdgId();
  float etapart=0.;
  if (partId0 == ipart) {
    for(unsigned int k = 0; k < dauGen0->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen0->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	etapart = dauMuGen->eta();
      }
    }
  }
  if (partId1 == ipart) {
    for(unsigned int k = 0; k < dauGen1->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen1->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	etapart = dauMuGen->eta();
      }
    }
  }
  if (partId2 == ipart) {
    for(unsigned int k = 0; k < dauGen2->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen2->daughter(k);
      if(abs(dauMuGen->pdgId()) == ipart && dauMuGen->status() ==1) {
	etapart = dauMuGen->eta();
      }
    }
  }
  return etapart;
}

float ZMuMu_MCanalyzer::getParticlePhi(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
{
  int partId0 = dauGen0->pdgId();
  int partId1 = dauGen1->pdgId();
  int partId2 = dauGen2->pdgId();
  float phipart=0.;
  if (partId0 == ipart) {
    for(unsigned int k = 0; k < dauGen0->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen0->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	phipart = dauMuGen->phi();
      }
    }
  }
  if (partId1 == ipart) {
    for(unsigned int k = 0; k < dauGen1->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen1->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	phipart = dauMuGen->phi();
      }
    }
  }
  if (partId2 == ipart) {
    for(unsigned int k = 0; k < dauGen2->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen2->daughter(k);
      if(abs(dauMuGen->pdgId()) == ipart && dauMuGen->status() ==1) {
	phipart = dauMuGen->phi();
      }
    }
  }
  return phipart;
}

Particle::LorentzVector ZMuMu_MCanalyzer::getParticleP4(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
{
  int partId0 = dauGen0->pdgId();
  int partId1 = dauGen1->pdgId();
  int partId2 = dauGen2->pdgId();
  Particle::LorentzVector p4part(0.,0.,0.,0.);
  if (partId0 == ipart) {
    for(unsigned int k = 0; k < dauGen0->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen0->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	p4part = dauMuGen->p4();
      }
    }
  }
  if (partId1 == ipart) {
    for(unsigned int k = 0; k < dauGen1->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen1->daughter(k);
      if(dauMuGen->pdgId() == ipart && dauMuGen->status() ==1) {
	p4part = dauMuGen->p4();
      }
    }
  }
  if (partId2 == ipart) {
    for(unsigned int k = 0; k < dauGen2->numberOfDaughters(); ++k) {
      const Candidate * dauMuGen = dauGen2->daughter(k);
      if(abs(dauMuGen->pdgId()) == ipart && dauMuGen->status() ==1) {
	p4part = dauMuGen->p4();
      }
    }
  }
  return p4part;
}



void ZMuMu_MCanalyzer::endJob() {


  double eff_Iso = double(nGlobalMuonsMatched_passedIso)/nGlobalMuonsMatched_passed;
  double err_effIso = sqrt(eff_Iso*(1-eff_Iso)/nGlobalMuonsMatched_passed);

  double n1_afterIso = 2*n2GlobalMuonsMatched_passedIso2Trg+nMu0onlyTriggered+nMu1onlyTriggered+nTracksMuonsMatched_passedIso;
  double n2_afterIso = 2*n2GlobalMuonsMatched_passedIso2Trg+nMu0onlyTriggered+nMu1onlyTriggered+nStaMuonsMatched_passedIso;
  double nGLB_afterIso = 2*n2GlobalMuonsMatched_passedIso2Trg+nMu0onlyTriggered+nMu1onlyTriggered;
  double effSta_afterIso = (2*n2GlobalMuonsMatched_passedIso2Trg+nMu0onlyTriggered+nMu1onlyTriggered)/n1_afterIso;
  double effTrk_afterIso = (2*n2GlobalMuonsMatched_passedIso2Trg+nMu0onlyTriggered+nMu1onlyTriggered)/n2_afterIso;
  double effHLT_afterIso = (2.* n2GlobalMuonsMatched_passedIso2Trg)/(2.* n2GlobalMuonsMatched_passedIso2Trg + nMu0onlyTriggered + nMu1onlyTriggered);
  double err_effHLT_afterIso= sqrt( effHLT_afterIso * (1 - effHLT_afterIso)/nGLB_afterIso);
  double err_effsta_afterIso = sqrt(effSta_afterIso*(1-effSta_afterIso)/n1_afterIso);
  double err_efftrk_afterIso = sqrt(effTrk_afterIso*(1-effTrk_afterIso)/n2_afterIso);


  cout << "------------------------------------  Counters  --------------------------------" << endl;

  cout << "number of events zMuMu matched " << nZMuMu_matched << endl;
  cout << "number of events zMuSta matched " << nZMuSta_matched << endl;
  cout << "number of events zMuTk matched " << nZMuTrk_matched << endl;
  cout << "number of events zMuMu with mu0 only triggered " << nMu0onlyTriggered << endl;
  cout << "number of events zMuMu with mu1 only triggered " << nMu1onlyTriggered << endl;
  cout << "=========================================" << endl;
  cout << "n. of global muons MC matched and passing cuts:           " << nGlobalMuonsMatched_passed << endl;
  cout << "n. of global muons MC matched and passing also Iso cut:       " << nGlobalMuonsMatched_passedIso << endl;
  cout << "n. of Z -> 2 global muons MC matched and passing ALL cuts:    " << n2GlobalMuonsMatched_passedIso << endl;
  cout << "n. of ZMuSta MC matched and passing ALL cuts:    " << nStaMuonsMatched_passedIso << endl;
  cout << "n. of ZmuTrck MC matched and passing ALL cuts:   " << nTracksMuonsMatched_passedIso << endl;
  cout << "n. of Z -> 2 global muons MC matched and passing ALL cuts and both triggered: " << n2GlobalMuonsMatched_passedIso2Trg << endl;
  cout << "=================================================================================" << endl;
  cout << "Iso efficiency: " << eff_Iso << " +/- " << err_effIso << endl;
  cout << "HLT efficiency: " << effHLT_afterIso << " +/- " << err_effHLT_afterIso << endl;
  cout << "eff StandAlone (after Isocut) : " << effSta_afterIso << "+/-" << err_effsta_afterIso << endl;
  cout << "eff Tracker (after Isocut)    : " << effTrk_afterIso << "+/-" << err_efftrk_afterIso << endl;

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMu_MCanalyzer);

