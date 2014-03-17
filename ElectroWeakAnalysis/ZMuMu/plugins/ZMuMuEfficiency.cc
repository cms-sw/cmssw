/* \class ZMuMuEfficiency
 *
 * author: Pasquale Noli
 * revised by Salvatore di Guida
 * revised for CSA08 by Davide Piccolo
 *
 * Efficiency of reconstruction tracker and muon Chamber
 *
 */

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "TH1.h"
#include <vector>

using namespace edm;
using namespace std;
using namespace reco;

typedef ValueMap<float> IsolationCollection;

class ZMuMuEfficiency : public edm::EDAnalyzer {
public:
  ZMuMuEfficiency(const edm::ParameterSet& pset);
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
  EDGetTokenT<CandidateView> zMuTrackToken_;
  EDGetTokenT<GenParticleMatch> zMuTrackMatchMapToken_;
  EDGetTokenT<CandidateView> zMuStandAloneToken_;
  EDGetTokenT<GenParticleMatch> zMuStandAloneMatchMapToken_;
  EDGetTokenT<CandidateView> muonsToken_;
  EDGetTokenT<GenParticleMatch> muonMatchMapToken_;
  EDGetTokenT<IsolationCollection> muonIsoToken_;
  EDGetTokenT<CandidateView> tracksToken_;
  EDGetTokenT<IsolationCollection> trackIsoToken_;
  EDGetTokenT<CandidateView> standAloneToken_;
  EDGetTokenT<IsolationCollection> standAloneIsoToken_;
  EDGetTokenT<GenParticleCollection> genParticlesToken_;

  double zMassMin_, zMassMax_, ptmin_, etamax_, isomax_;
  unsigned int nbinsPt_, nbinsEta_;
  reco::CandidateBaseRef globalMuonCandRef_, trackMuonCandRef_, standAloneMuonCandRef_;
  OverlapChecker overlap_;

  //histograms for measuring tracker efficiency
  TH1D *h_etaStandAlone_, *h_etaMuonOverlappedToStandAlone_;
  TH1D *h_ptStandAlone_, *h_ptMuonOverlappedToStandAlone_;

  //histograms for measuring standalone efficiency
  TH1D *h_etaTrack_, *h_etaMuonOverlappedToTrack_;
  TH1D *h_ptTrack_, *h_ptMuonOverlappedToTrack_;

  //histograms for MC acceptance
  TH1D *h_nZMCfound_;
  TH1D *h_ZetaGen_, *h_ZptGen_, *h_ZmassGen_;
  TH1D *h_muetaGen_, *h_muptGen_, *h_muIsoGen_;
  TH1D *h_dimuonPtGen_, *h_dimuonMassGen_, *h_dimuonEtaGen_;
  TH1D *h_ZetaGenPassed_, *h_ZptGenPassed_, *h_ZmassGenPassed_;
  TH1D *h_muetaGenPassed_, *h_muptGenPassed_, *h_muIsoGenPassed_;
  TH1D *h_dimuonPtGenPassed_, *h_dimuonMassGenPassed_, *h_dimuonEtaGenPassed_;
  //histograms for invarian mass resolution
  TH1D *h_DELTA_ZMuMuMassReco_dimuonMassGen_, *h_DELTA_ZMuStaMassReco_dimuonMassGen_, *h_DELTA_ZMuTrackMassReco_dimuonMassGen_;

  int numberOfEventsWithZMuMufound, numberOfEventsWithZMuStafound;
  int numberOfMatchedZMuSta_,numberOfMatchedSelectedZMuSta_;
  int numberOfMatchedZMuMu_, numberOfMatchedSelectedZMuMu_;
  int numberOfOverlappedStandAlone_, numberOfOverlappedTracks_, numberOfMatchedZMuTrack_notOverlapped;
  int numberOfMatchedZMuTrack_exclusive ,numberOfMatchedSelectedZMuTrack_exclusive;
  int numberOfMatchedZMuTrack_matchedZMuMu, numberOfMatchedZMuTrack_matchedSelectedZMuMu ;
  int totalNumberOfevents, totalNumberOfZfound, totalNumberOfZPassed;
  int noMCmatching, ZMuTrack_exclusive_1match, ZMuTrack_exclusive_morematch;
  int ZMuTrackselected_exclusive_1match, ZMuTrackselected_exclusive_morematch;
  int ZMuTrack_ZMuMu_1match, ZMuTrack_ZMuMu_2match, ZMuTrack_ZMuMu_morematch;

  int n_zMuMufound_genZsele, n_zMuStafound_genZsele, n_zMuTrkfound_genZsele;
};

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include <iostream>
#include <iterator>
#include <cmath>


ZMuMuEfficiency::ZMuMuEfficiency(const ParameterSet& pset) :
  zMuMuToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuMu"))),
  zMuMuMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuMuMatchMap"))),
  zMuTrackToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuTrack"))),
  zMuTrackMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuTrackMatchMap"))),
  zMuStandAloneToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuStandAlone"))),
  zMuStandAloneMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuStandAloneMatchMap"))),
  muonsToken_(consumes<CandidateView>(pset.getParameter<InputTag>("muons"))),
  muonMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("muonMatchMap"))),
  muonIsoToken_(mayConsume<IsolationCollection>(pset.getParameter<InputTag>("muonIso"))),
  tracksToken_(consumes<CandidateView>(pset.getParameter<InputTag>("tracks"))),
  trackIsoToken_(mayConsume<IsolationCollection>(pset.getParameter<InputTag>("trackIso"))),
  standAloneToken_(consumes<CandidateView>(pset.getParameter<InputTag>("standAlone"))),
  standAloneIsoToken_(mayConsume<IsolationCollection>(pset.getParameter<InputTag>("standAloneIso"))),
  genParticlesToken_(consumes<GenParticleCollection>(pset.getParameter<InputTag>( "genParticles"))),

  zMassMin_(pset.getUntrackedParameter<double>("zMassMin")),
  zMassMax_(pset.getUntrackedParameter<double>("zMassMax")),
  ptmin_(pset.getUntrackedParameter<double>("ptmin")),
  etamax_(pset.getUntrackedParameter<double>("etamax")),
  isomax_(pset.getUntrackedParameter<double>("isomax")),
  nbinsPt_(pset.getUntrackedParameter<unsigned int>("nbinsPt")),
  nbinsEta_(pset.getUntrackedParameter<unsigned int>("nbinsEta")) {
  Service<TFileService> fs;
  TFileDirectory trackEffDir = fs->mkdir("TrackEfficiency");

  // tracker efficiency distributions
  h_etaStandAlone_ = trackEffDir.make<TH1D>("StandAloneMuonEta",
					    "StandAlone #eta for Z -> #mu + standalone",
					    nbinsEta_, -etamax_, etamax_);
  h_etaMuonOverlappedToStandAlone_ = trackEffDir.make<TH1D>("MuonOverlappedToStandAloneEta",
							    "Global muon overlapped to standAlone #eta for Z -> #mu + sa",
							    nbinsEta_, -etamax_, etamax_);
  h_ptStandAlone_ = trackEffDir.make<TH1D>("StandAloneMuonPt",
					   "StandAlone p_{t} for Z -> #mu + standalone",
					   nbinsPt_, ptmin_, 100);
  h_ptMuonOverlappedToStandAlone_ = trackEffDir.make<TH1D>("MuonOverlappedToStandAlonePt",
							   "Global muon overlapped to standAlone p_{t} for Z -> #mu + sa",
							   nbinsPt_, ptmin_, 100);


  // StandAlone efficiency distributions
  TFileDirectory standaloneEffDir = fs->mkdir("StandaloneEfficiency");
  h_etaTrack_ = standaloneEffDir.make<TH1D>("TrackMuonEta",
					    "Track #eta for Z -> #mu + track",
					    nbinsEta_, -etamax_, etamax_);
  h_etaMuonOverlappedToTrack_ = standaloneEffDir.make<TH1D>("MuonOverlappedToTrackEta",
							    "Global muon overlapped to track #eta for Z -> #mu + tk",
							    nbinsEta_, -etamax_, etamax_);
  h_ptTrack_ = standaloneEffDir.make<TH1D>("TrackMuonPt",
					   "Track p_{t} for Z -> #mu + track",
					   nbinsPt_, ptmin_, 100);
  h_ptMuonOverlappedToTrack_ = standaloneEffDir.make<TH1D>("MuonOverlappedToTrackPt",
							   "Global muon overlapped to track p_{t} for Z -> #mu + tk",
							   nbinsPt_, ptmin_, 100);


  // inv. mass resolution studies
  TFileDirectory invMassResolutionDir = fs->mkdir("invriantMassResolution");
  h_DELTA_ZMuMuMassReco_dimuonMassGen_ = invMassResolutionDir.make<TH1D>("zMuMu_invMassResolution","zMuMu invariant Mass Resolution",50,-25,25);
  h_DELTA_ZMuStaMassReco_dimuonMassGen_ = invMassResolutionDir.make<TH1D>("zMuSta_invMassResolution","zMuSta invariant Mass Resolution",50,-25,25);
  h_DELTA_ZMuTrackMassReco_dimuonMassGen_ = invMassResolutionDir.make<TH1D>("zMuTrack_invMassResolution","zMuTrack invariant Mass Resolution",50,-25,25);


  // generator level histograms
  TFileDirectory genParticleDir = fs->mkdir("genParticle");
  h_nZMCfound_ = genParticleDir.make<TH1D>("NumberOfgeneratedZeta","n. of generated Z per event",4,-.5,3.5);
  h_ZetaGen_ = genParticleDir.make<TH1D>("generatedZeta","#eta of generated Z",100,-5.,5.);
  h_ZptGen_ = genParticleDir.make<TH1D>("generatedZpt","pt of generated Z",100,0.,200.);
  h_ZmassGen_ = genParticleDir.make<TH1D>("generatedZmass","mass of generated Z",100,0.,200.);
  h_muetaGen_ = genParticleDir.make<TH1D>("generatedMuonEta","#eta of generated muons from Z decay",100,-5.,5.);
  h_muptGen_ = genParticleDir.make<TH1D>("generatedMuonpt","pt of generated muons from Z decay",100,0.,200.);
  h_dimuonEtaGen_ = genParticleDir.make<TH1D>("generatedDimuonEta","#eta of generated dimuon",100,-5.,5.);
  h_dimuonPtGen_ = genParticleDir.make<TH1D>("generatedDimuonPt","pt of generated dimuon",100,0.,200.);
  h_dimuonMassGen_ = genParticleDir.make<TH1D>("generatedDimuonMass","mass of generated dimuon",100,0.,200.);
  h_ZetaGenPassed_ = genParticleDir.make<TH1D>("generatedZeta_passed","#eta of generated Z after cuts",100,-5.,5.);
  h_ZptGenPassed_ = genParticleDir.make<TH1D>("generatedZpt_passed","pt of generated Z after cuts",100,0.,200.);
  h_ZmassGenPassed_ = genParticleDir.make<TH1D>("generatedZmass_passed","mass of generated Z after cuts",100,0.,200.);
  h_muetaGenPassed_ = genParticleDir.make<TH1D>("generatedMuonEta_passed","#eta of generated muons from Z decay after cuts",100,-5.,5.);
  h_muptGenPassed_ = genParticleDir.make<TH1D>("generatedMuonpt_passed","pt of generated muons from Z decay after cuts",100,0.,200.);
  h_dimuonEtaGenPassed_ = genParticleDir.make<TH1D>("generatedDimuonEta_passed","#eta of generated dimuon after cuts",100,-5.,5.);
  h_dimuonPtGenPassed_ = genParticleDir.make<TH1D>("generatedDimuonPt_passed","pt of generated dimuon after cuts",100,0.,200.);
  h_dimuonMassGenPassed_ = genParticleDir.make<TH1D>("generatedDimuonMass_passed","mass of generated dimuon after cuts",100,0.,200.);
  // to insert isolation histograms  ..............

  numberOfEventsWithZMuMufound = 0;
  numberOfEventsWithZMuStafound = 0;
  numberOfMatchedZMuMu_ = 0;
  numberOfMatchedSelectedZMuMu_ = 0;
  numberOfMatchedZMuSta_ = 0;
  numberOfMatchedSelectedZMuSta_ = 0;
  numberOfMatchedZMuTrack_matchedZMuMu = 0;
  numberOfMatchedZMuTrack_matchedSelectedZMuMu = 0;
  numberOfMatchedZMuTrack_exclusive = 0;
  numberOfMatchedSelectedZMuTrack_exclusive = 0;
  numberOfOverlappedStandAlone_ = 0;
  numberOfOverlappedTracks_ = 0;
  numberOfMatchedZMuTrack_notOverlapped = 0;
  noMCmatching = 0;
  ZMuTrack_exclusive_1match = 0;
  ZMuTrack_exclusive_morematch = 0;
  ZMuTrackselected_exclusive_1match = 0;
  ZMuTrackselected_exclusive_morematch = 0;
  ZMuTrack_ZMuMu_1match = 0;
  ZMuTrack_ZMuMu_2match = 0;
  ZMuTrack_ZMuMu_morematch = 0;

  n_zMuMufound_genZsele = 0;
  n_zMuStafound_genZsele = 0;
  n_zMuTrkfound_genZsele = 0;

  // generator counters
  totalNumberOfevents = 0;
  totalNumberOfZfound = 0;
  totalNumberOfZPassed = 0;

}

void ZMuMuEfficiency::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateView> zMuMu;
  Handle<GenParticleMatch> zMuMuMatchMap; //Map of Z made by Mu global + Mu global
  Handle<CandidateView> zMuTrack;
  Handle<GenParticleMatch> zMuTrackMatchMap; //Map of Z made by Mu + Track
  Handle<CandidateView> zMuStandAlone;
  Handle<GenParticleMatch> zMuStandAloneMatchMap; //Map of Z made by Mu + StandAlone
  Handle<CandidateView> muons; //Collection of Muons
  Handle<GenParticleMatch> muonMatchMap;
  Handle<IsolationCollection> muonIso;
  Handle<CandidateView> tracks; //Collection of Tracks
  Handle<IsolationCollection> trackIso;
  Handle<CandidateView> standAlone; //Collection of StandAlone
  Handle<IsolationCollection> standAloneIso;
  Handle<GenParticleCollection> genParticles;  // Collection of Generatd Particles

  event.getByToken(zMuMuToken_, zMuMu);
  event.getByToken(zMuTrackToken_, zMuTrack);
  event.getByToken(zMuStandAloneToken_, zMuStandAlone);
  event.getByToken(muonsToken_, muons);
  event.getByToken(tracksToken_, tracks);
  event.getByToken(standAloneToken_, standAlone);
  event.getByToken(genParticlesToken_, genParticles);

  cout << "*********  zMuMu         size : " << zMuMu->size() << endl;
  cout << "*********  zMuStandAlone size : " << zMuStandAlone->size() << endl;
  cout << "*********  zMuTrack      size : " << zMuTrack->size() << endl;
  cout << "*********  muons         size : " << muons->size()<< endl;
  cout << "*********  standAlone    size : " << standAlone->size()<< endl;
  cout << "*********  tracks        size : " << tracks->size()<< endl;
  cout << "*********  generated     size : " << genParticles->size()<< endl;


  // generator level distributions

  int nZMCfound = 0;
  totalNumberOfevents++;
  int ngen = genParticles->size();
  bool ZMuMuMatchedfound = false;
  bool ZMuMuMatchedSelectedfound = false;
  bool ZMuStaMatchedfound = false;
  //bool ZMuStaMatchedSelectedfound = false;
  int ZMuTrackMatchedfound = 0;
  int ZMuTrackMatchedSelected_exclusivefound = 0;

  double dimuonMassGen = 0;

  for (int i=0; i<ngen; i++) {
    const Candidate &genCand = (*genParticles)[i];

    //   if((genCand.pdgId() == 23) && (genCand.status() == 2)) //this is an intermediate Z0
      //      cout << ">>> intermediate Z0 found, with " << genCand.numberOfDaughters() << " daughters" << endl;
    if((genCand.pdgId() == 23)&&(genCand.status() == 3)) { //this is a Z0
      if(genCand.numberOfDaughters() == 3) {                    // possible Z0 decays in mu+ mu-, the 3rd daughter is the same Z0
	const Candidate * dauGen0 = genCand.daughter(0);
	const Candidate * dauGen1 = genCand.daughter(1);
	const Candidate * dauGen2 = genCand.daughter(2);
	if (check_ifZmumu(dauGen0, dauGen1, dauGen2)) {         // Z0 in mu+ mu-
	  totalNumberOfZfound++;
	  nZMCfound++;
	  bool checkpt = false;
	  bool checketa = false;
	  bool checkmass = false;
	  float mupluspt, muminuspt, mupluseta, muminuseta;
	  mupluspt = getParticlePt(-13,dauGen0,dauGen1,dauGen2);
	  muminuspt = getParticlePt(13,dauGen0,dauGen1,dauGen2);
	  mupluseta = getParticleEta(-13,dauGen0,dauGen1,dauGen2);
	  muminuseta = getParticleEta(13,dauGen0,dauGen1,dauGen2);
	  //float muplusphi = getParticlePhi(-13,dauGen0,dauGen1,dauGen2);
	  //float muminusphi = getParticlePhi(13,dauGen0,dauGen1,dauGen2);
	  Particle::LorentzVector pZ(0, 0, 0, 0);
	  Particle::LorentzVector muplusp4 = getParticleP4(-13,dauGen0,dauGen1,dauGen2);
	  Particle::LorentzVector muminusp4 = getParticleP4(13,dauGen0,dauGen1,dauGen2);
	  pZ = muplusp4 + muminusp4;
	  double dimuon_pt = sqrt(pZ.x()*pZ.x()+pZ.y()*pZ.y());
	  double tan_theta_half = tan(atan(dimuon_pt/pZ.z())/2.);
	  double dimuon_eta = 0.;
	  if (tan_theta_half>0) dimuon_eta = -log(tan(tan_theta_half));
	  if (tan_theta_half<=0) dimuon_eta = log(tan(-tan_theta_half));

	  dimuonMassGen = pZ.mass();  // dimuon invariant Mass at Generator Level

	  h_ZmassGen_->Fill(genCand.mass());
	  h_ZetaGen_->Fill(genCand.eta());
	  h_ZptGen_->Fill(genCand.pt());
	  h_dimuonMassGen_->Fill(pZ.mass());
	  h_dimuonEtaGen_->Fill(dimuon_eta);
	  h_dimuonPtGen_->Fill(dimuon_pt);
	  h_muetaGen_->Fill(mupluseta);
	  h_muetaGen_->Fill(muminuseta);
	  h_muptGen_->Fill(mupluspt);
	  h_muptGen_->Fill(muminuspt);
                             // dimuon 4-momentum
	  //	  h_mDimuonMC->Fill(pZ.mass());
	  //	  h_ZminusDimuonMassMC->Fill(genCand.mass()-pZ.mass());
	  //	  h_DeltaPhiMC->Fill(deltaPhi(muplusphi,muminusphi));
	  //	  if (dauGen2==23) float z_eta = dauGen2->eta();
	  //	  if (dauGen2==23) float Zpt = dauGen2->pt();
	  //	  h_DeltaPhivsZPtMC->Fill(DeltaPhi(muplusphi,muminusphi),ZPt);

	  if (mupluspt > ptmin_ && muminuspt > ptmin_) checkpt = true;
	  if (mupluseta < etamax_ && muminuseta < etamax_) checketa = true;
	  if (genCand.mass()>zMassMin_ && genCand.mass()<zMassMax_) checkmass = true;
	  if (checkpt && checketa && checkmass) {
	    totalNumberOfZPassed++;
	    h_ZmassGenPassed_->Fill(genCand.mass());
	    h_ZetaGenPassed_->Fill(genCand.eta());
	    h_ZptGenPassed_->Fill(genCand.pt());
	    h_dimuonMassGenPassed_->Fill(pZ.mass());
	    h_dimuonEtaGenPassed_->Fill(dimuon_eta);
	    h_dimuonPtGenPassed_->Fill(dimuon_pt);
	    h_muetaGenPassed_->Fill(mupluseta);
	    h_muetaGenPassed_->Fill(muminuseta);
	    h_muptGenPassed_->Fill(mupluspt);
	    h_muptGenPassed_->Fill(muminuspt);

	    if (zMuMu->size() > 0 ) {
	      n_zMuMufound_genZsele++;
	    }
	    else if (zMuStandAlone->size() > 0 ) {
		n_zMuStafound_genZsele++;
	    }
	    else {
	      n_zMuTrkfound_genZsele++;
	    }

	  }
	}

      }
    }
  }
  h_nZMCfound_->Fill(nZMCfound);                  // number of Z found in the event at generator level

  //TRACK efficiency (conto numero di eventi Zmumu global e ZmuSta (ricorda che sono due campioni esclusivi)

  if (zMuMu->size() > 0 ) {
    numberOfEventsWithZMuMufound++;
    event.getByToken(zMuMuMatchMapToken_, zMuMuMatchMap);
    event.getByToken(muonIsoToken_, muonIso);
    event.getByToken(standAloneIsoToken_, standAloneIso);
    event.getByToken(muonMatchMapToken_, muonMatchMap);
    for(unsigned int i = 0; i < zMuMu->size(); ++i) { //loop on candidates
      const Candidate & zMuMuCand = (*zMuMu)[i]; //the candidate
      CandidateBaseRef zMuMuCandRef = zMuMu->refAt(i);
      bool isMatched = false;
      GenParticleRef zMuMuMatch = (*zMuMuMatchMap)[zMuMuCandRef];

      if(zMuMuMatch.isNonnull()) {  // ZMuMu matched
	isMatched = true;
	numberOfMatchedZMuMu_++;
      }
      CandidateBaseRef dau0 = zMuMuCand.daughter(0)->masterClone();
      CandidateBaseRef dau1 = zMuMuCand.daughter(1)->masterClone();
      if (isMatched) ZMuMuMatchedfound = true;

      // Cuts
      if((dau0->pt() > ptmin_) && (dau1->pt() > ptmin_) &&
	 (fabs(dau0->eta()) < etamax_) && (fabs(dau1->eta()) < etamax_) &&
	 (zMuMuCand.mass() > zMassMin_) && (zMuMuCand.mass() < zMassMax_) &&
	 (isMatched)) {
	//The Z daughters are already matched!
	const double globalMuonIsolation0 = (*muonIso)[dau0];
	const double globalMuonIsolation1 = (*muonIso)[dau1];
	if((globalMuonIsolation0 < isomax_) && (globalMuonIsolation1 < isomax_)) {      // ZMuMu matched and selected by cuts
	  ZMuMuMatchedSelectedfound = true;
	  numberOfMatchedSelectedZMuMu_++;
	  h_etaStandAlone_->Fill(dau0->eta());            // StandAlone found dau0, eta
	  h_etaStandAlone_->Fill(dau1->eta());            // StandAlone found dau1, eta
	  h_etaMuonOverlappedToStandAlone_->Fill(dau0->eta());  // is global muon so dau0 is also found as a track, eta
	  h_etaMuonOverlappedToStandAlone_->Fill(dau1->eta());  // is global muon so dau1 is also found as a track, eta
	  h_ptStandAlone_->Fill(dau0->pt());            // StandAlone found dau0, pt
	  h_ptStandAlone_->Fill(dau1->pt());            // StandAlone found dau1, pt
	  h_ptMuonOverlappedToStandAlone_->Fill(dau0->pt());  // is global muon so dau0 is also found as a track, pt
	  h_ptMuonOverlappedToStandAlone_->Fill(dau1->pt());  // is global muon so dau1 is also found as a track, pt

	  h_etaTrack_->Fill(dau0->eta());            // Track found dau0, eta
	  h_etaTrack_->Fill(dau1->eta());            // Track found dau1, eta
	  h_etaMuonOverlappedToTrack_->Fill(dau0->eta());  // is global muon so dau0 is also found as a StandAlone, eta
	  h_etaMuonOverlappedToTrack_->Fill(dau1->eta());  // is global muon so dau1 is also found as a StandAlone, eta
	  h_ptTrack_->Fill(dau0->pt());            // Track found dau0, pt
	  h_ptTrack_->Fill(dau1->pt());            // Track found dau1, pt
	  h_ptMuonOverlappedToTrack_->Fill(dau0->pt());  // is global muon so dau0 is also found as a StandAlone, pt
	  h_ptMuonOverlappedToTrack_->Fill(dau1->pt());  // is global muon so dau1 is also found as a StandAlone, pt

	  h_DELTA_ZMuMuMassReco_dimuonMassGen_->Fill(zMuMuCand.mass()-dimuonMassGen);
	  // check that the two muons are matched . .per ora è solo un mio controllo
	  for(unsigned int j = 0; j < muons->size() ; ++j) {
	    CandidateBaseRef muCandRef = muons->refAt(j);
	    GenParticleRef muonMatch = (*muonMatchMap)[muCandRef];
	    //	    if (muonMatch.isNonnull()) cout << "mu match n. " << j << endl;
	  }
	}
      }
    }
  }

  if (zMuStandAlone->size() > 0) {
    numberOfEventsWithZMuStafound++;
    event.getByToken(zMuStandAloneMatchMapToken_, zMuStandAloneMatchMap);
    event.getByToken(muonIsoToken_, muonIso);
    event.getByToken(standAloneIsoToken_, standAloneIso);
    event.getByToken(muonMatchMapToken_, muonMatchMap);
    for(unsigned int i = 0; i < zMuStandAlone->size(); ++i) { //loop on candidates
      const Candidate & zMuStaCand = (*zMuStandAlone)[i]; //the candidate
      CandidateBaseRef zMuStaCandRef = zMuStandAlone->refAt(i);
      bool isMatched = false;
      GenParticleRef zMuStaMatch = (*zMuStandAloneMatchMap)[zMuStaCandRef];
      if(zMuStaMatch.isNonnull()) {        // ZMuSta Macthed
	isMatched = true;
	ZMuStaMatchedfound = true;
	numberOfMatchedZMuSta_++;
      }
      CandidateBaseRef dau0 = zMuStaCand.daughter(0)->masterClone();
      CandidateBaseRef dau1 = zMuStaCand.daughter(1)->masterClone();

      // Cuts
      if((dau0->pt() > ptmin_) && (dau1->pt() > ptmin_) &&
	 (fabs(dau0->eta()) < etamax_) && (fabs(dau1->eta()) < etamax_) &&
	 (zMuStaCand.mass() > zMassMin_) && (zMuStaCand.mass() < zMassMax_) &&
	 (isMatched)) {
	CandidateBaseRef standAloneMuonCandRef_, globalMuonCandRef_;
	if(dau0->isGlobalMuon()) {
	  standAloneMuonCandRef_ = dau1;
	  globalMuonCandRef_ = dau0;
	}
	if(dau1->isGlobalMuon()) {
	  standAloneMuonCandRef_ = dau0;
	  globalMuonCandRef_ = dau1;
	}

	const double globalMuonIsolation = (*muonIso)[globalMuonCandRef_];
	const double standAloneMuonIsolation = (*standAloneIso)[standAloneMuonCandRef_];

	if((globalMuonIsolation < isomax_) && (standAloneMuonIsolation < isomax_)) {   // ZMuSta matched and selected
	  //ZMuStaMatchedSelectedfound = true;
	  numberOfMatchedSelectedZMuSta_++;
	  h_etaStandAlone_->Fill(standAloneMuonCandRef_->eta()); //Denominator eta for measuring track efficiency
	  h_ptStandAlone_->Fill(standAloneMuonCandRef_->pt());   //Denominator pt for measuring track eff
	  h_DELTA_ZMuStaMassReco_dimuonMassGen_->Fill(zMuStaCand.mass()-dimuonMassGen); // differnce between ZMuSta reco and dimuon mass gen

	}
      }
    }
  } //end loop on Candidate

  //STANDALONE efficiency

  if (zMuTrack->size() > 0) {
    event.getByToken(zMuTrackMatchMapToken_, zMuTrackMatchMap);
    event.getByToken(muonIsoToken_, muonIso);
    event.getByToken(trackIsoToken_, trackIso);
    event.getByToken(muonMatchMapToken_, muonMatchMap);
    for(unsigned int i = 0; i < zMuTrack->size(); ++i) { //loop on candidates
      const Candidate & zMuTrkCand = (*zMuTrack)[i]; //the candidate
      CandidateBaseRef zMuTrkCandRef = zMuTrack->refAt(i);
      bool isMatched = false;
      GenParticleRef zMuTrkMatch = (*zMuTrackMatchMap)[zMuTrkCandRef];
      if(zMuTrkMatch.isNonnull()) {
	isMatched = true;
      }
      CandidateBaseRef dau0 = zMuTrkCand.daughter(0)->masterClone();
      CandidateBaseRef dau1 = zMuTrkCand.daughter(1)->masterClone();

      if (isMatched) {
	ZMuTrackMatchedfound++;
	if (ZMuMuMatchedfound) numberOfMatchedZMuTrack_matchedZMuMu++;
	if (ZMuMuMatchedSelectedfound) numberOfMatchedZMuTrack_matchedSelectedZMuMu++;
	if (!ZMuMuMatchedfound) numberOfMatchedZMuTrack_exclusive++;
      }
      // Cuts
      if ((dau0->pt() > ptmin_) && (dau1->pt() > ptmin_) &&
	  (fabs(dau0->eta()) < etamax_) && (fabs(dau1->eta())< etamax_) &&
	  (zMuTrkCand.mass() > zMassMin_) && (zMuTrkCand.mass() < zMassMax_) &&
	  (isMatched) && !ZMuMuMatchedfound && !ZMuStaMatchedfound ) {

	// dau0 is always the global muon, dau1 is the track for ZMuTrack collection
	const double globalMuonIsolation = (*muonIso)[dau0];
	const double trackMuonIsolation = (*trackIso)[dau1];
	if((globalMuonIsolation < isomax_) && (trackMuonIsolation < isomax_)) { // ZMuTRack matched - selected without ZMuMu found (exclusive)
	  numberOfMatchedSelectedZMuTrack_exclusive++;
	  ZMuTrackMatchedSelected_exclusivefound++;
	  h_etaTrack_->Fill(dau1->eta()); //Denominator eta Sta
	  h_ptTrack_->Fill(dau1->pt());   //Denominator pt Sta
	  h_DELTA_ZMuTrackMassReco_dimuonMassGen_->Fill(zMuTrkCand.mass()-dimuonMassGen);
	}

      }
    }
  } //end loop on Candidate

  if (!ZMuMuMatchedfound && !ZMuStaMatchedfound && ZMuTrackMatchedfound == 0) noMCmatching++;
  if (!ZMuMuMatchedfound && ZMuTrackMatchedfound == 1) ZMuTrack_exclusive_1match++;
  if (!ZMuMuMatchedfound && ZMuTrackMatchedfound > 1) ZMuTrack_exclusive_morematch++;
  if (!ZMuMuMatchedfound && ZMuTrackMatchedSelected_exclusivefound == 1) ZMuTrackselected_exclusive_1match++;
  if (!ZMuMuMatchedfound && ZMuTrackMatchedSelected_exclusivefound > 1) ZMuTrackselected_exclusive_morematch++;
  if (ZMuMuMatchedfound && ZMuTrackMatchedfound == 1) ZMuTrack_ZMuMu_1match++;
  if (ZMuMuMatchedfound && ZMuTrackMatchedfound == 2) ZMuTrack_ZMuMu_2match++;
  if (ZMuMuMatchedfound && ZMuTrackMatchedfound > 2) ZMuTrack_ZMuMu_morematch++;

}

bool ZMuMuEfficiency::check_ifZmumu(const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMuEfficiency::getParticlePt(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMuEfficiency::getParticleEta(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMuEfficiency::getParticlePhi(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

Particle::LorentzVector ZMuMuEfficiency::getParticleP4(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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



void ZMuMuEfficiency::endJob() {
  //  double efficiencySTA =(double)numberOfOverlappedStandAlone_/(double)numberOfMatchedZMuTrack_;
  //  double errorEff_STA = sqrt( efficiencySTA*(1 - efficiencySTA)/(double)numberOfMatchedZMuTrack_);

  double myTrackEff = 2.*numberOfMatchedSelectedZMuMu_/(2.*numberOfMatchedSelectedZMuMu_+(double)numberOfMatchedSelectedZMuSta_);
  double myErrTrackEff = sqrt(myTrackEff*(1-myTrackEff)/(2.*numberOfMatchedSelectedZMuMu_+(double)numberOfMatchedSelectedZMuSta_));

  double myStaEff = 2.*numberOfMatchedSelectedZMuMu_/(2.*numberOfMatchedSelectedZMuMu_+(double)numberOfMatchedSelectedZMuTrack_exclusive);
  double myErrStaEff = sqrt(myTrackEff*(1-myTrackEff)/(2.*numberOfMatchedSelectedZMuMu_+(double)numberOfMatchedSelectedZMuTrack_exclusive));

  //  double efficiencyTRACK =(double)numberOfOverlappedTracks_/(double)numberOfMatchedZMuSta_;
  //  double errorEff_TRACK = sqrt( efficiencyTRACK*(1 - efficiencyTRACK)/(double)numberOfMatchedZMuSta_);

  cout << "------------------------------------  Counters for MC acceptance --------------------------------" << endl;
  cout << "totalNumberOfevents = " << totalNumberOfevents << endl;
  cout << "totalNumberOfZfound = " << totalNumberOfZfound << endl;
  cout << "totalNumberOfZpassed = " << totalNumberOfZPassed << endl;
  cout << "n. of events zMuMu found (gen level selected)" <<   n_zMuMufound_genZsele << endl;
  cout << "n. of events zMuSta found (gen level selected)" <<   n_zMuStafound_genZsele << endl;
  cout << "n. of events zMuTrk found (gen level selected)" <<   n_zMuTrkfound_genZsele << endl;

  cout << "----------------------------   Counter for MC truth efficiency calculation--------------------- " << endl;

  cout << "number of events with ZMuMu found = " << numberOfEventsWithZMuMufound << endl;
  cout << "number of events with ZMuSta found = " << numberOfEventsWithZMuStafound << endl;
  cout << "-------------------------------------------------------------------------------------- " << endl;

  cout << "number of events without MC maching = " << noMCmatching << endl;
  cout << "number of ZMuTrack exclsive 1 match = " << ZMuTrack_exclusive_1match << endl;
  cout << "number of ZMuTrack exclsive more match = " << ZMuTrack_exclusive_morematch << endl;
  cout << "number of ZMuTrack selected exclusive 1 match = " << ZMuTrackselected_exclusive_1match << endl;
  cout << "number of ZMuTrack selected exclusive more match = " << ZMuTrackselected_exclusive_morematch << endl;
  cout << "number of ZMuTrack ZMuMu 1 match = " << ZMuTrack_ZMuMu_1match << endl;
  cout << "number of ZMuTrack ZMuMu 2 match = " << ZMuTrack_ZMuMu_2match << endl;
  cout << "number of ZMuTrack ZMuMu more match = " << ZMuTrack_ZMuMu_morematch << endl;
  cout << "numberOfMatchedZMuMu = " << numberOfMatchedZMuMu_ << endl;
  cout << "numberOfMatchedSelectdZMuMu = " << numberOfMatchedSelectedZMuMu_ << endl;
  cout << "numberOfMatchedZMuSta = " << numberOfMatchedZMuSta_ << endl;
  cout << "numberOfMatchedSelectedZMuSta = " << numberOfMatchedSelectedZMuSta_ << endl;
  cout << "numberOfMatchedZMuTrack_matchedZMuMu = " << numberOfMatchedZMuTrack_matchedZMuMu << endl;
  cout << "numberOfMatchedZMuTrack_matchedSelectedZMuMu = " << numberOfMatchedZMuTrack_matchedSelectedZMuMu << endl;
  cout << "numberOfMatchedZMuTrack exclusive = " << numberOfMatchedZMuTrack_exclusive << endl;
  cout << "numberOfMatchedSelectedZMuTrack exclusive = " << numberOfMatchedSelectedZMuTrack_exclusive << endl;
  cout << " ----------------------------- Efficiency --------------------------------- " << endl;
  cout << "Efficiency StandAlone = " << myStaEff << " +/- " << myErrStaEff << endl;
  cout << "Efficiency Track      = " << myTrackEff << " +/- " << myErrTrackEff << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuEfficiency);

