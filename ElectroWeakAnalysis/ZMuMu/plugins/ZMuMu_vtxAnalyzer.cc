/* \class ZMuMu_vtxAnalyzer
 *
 * author: Davide Piccolo
 *
 * ZMuMu Vtx analyzer:
 * check muon vtx distributions,
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
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include <vector>

using namespace std;
using namespace reco;
using namespace edm;

typedef edm::ValueMap<float> IsolationCollection;

class ZMuMu_vtxAnalyzer : public edm::EDAnalyzer {
public:
  ZMuMu_vtxAnalyzer(const edm::ParameterSet& pset);
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
  EDGetTokenT<VertexCollection> primaryVerticesToken_;

  bool bothMuons_;

  double etamax_, ptmin_, massMin_, massMax_, isoMax_;

  reco::CandidateBaseRef globalMuonCandRef_, trackMuonCandRef_, standAloneMuonCandRef_;
  OverlapChecker overlap_;

  // general histograms

  // vertex studies
  // ... zmumu No cuts
  TH1D *h_muon_vz, *h_dimuon_vz, *h_muon_d0signed;
  TH1D *h_muon_vz_respectToPV, *h_muon_d0signed_respectToPV;
  // ... cynematic cuts zmumu
  TH1D *h_zmumuSele_muon_vz, *h_zmumuSele_dimuon_vz, *h_zmumuSele_muon_d0signed;
  TH1D *h_zmumuSele_muon_vz_respectToPV, *h_zmumuSele_muon_d0signed_respectToPV;
  // ... cynematic cuts zmumuNotIso
  TH1D *h_zmumuNotIsoSele_dimuon_vz;
  TH1D *h_zmumuNotIsoSele_muonIso_vz, *h_zmumuNotIsoSele_muonIso_d0signed;
  TH1D *h_zmumuNotIsoSele_muonIso_vz_respectToPV, *h_zmumuNotIsoSele_muonIso_d0signed_respectToPV;
  TH1D *h_zmumuNotIsoSele_muonNotIso_vz, *h_zmumuNotIsoSele_muonNotIso_d0signed;
  TH1D *h_zmumuNotIsoSele_muonNotIso_vz_respectToPV, *h_zmumuNotIsoSele_muonNotIso_d0signed_respectToPV;
  // ... cynematic cuts zmutrack
  TH1D *h_zmutrackSele_muon_vz, *h_zmutrackSele_muon_d0signed;
  TH1D *h_zmutrackSele_muon_vz_respectToPV, *h_zmutrackSele_muon_d0signed_respectToPV;
  TH1D *h_zmutrackSele_track_vz, *h_zmutrackSele_track_d0signed;
  TH1D *h_zmutrackSele_track_vz_respectToPV, *h_zmutrackSele_track_d0signed_respectToPV;
  // ... cynematic cuts zmusta
  TH1D *h_zmustaSele_muon_vz, *h_zmustaSele_muon_d0signed;
  TH1D *h_zmustaSele_muon_vz_respectToPV, *h_zmustaSele_muon_d0signed_respectToPV;
  TH1D *h_zmustaSele_sta_vz, *h_zmustaSele_sta_d0signed;
  TH1D *h_zmustaSele_sta_vz_respectToPV, *h_zmustaSele_sta_d0signed_respectToPV;


  // global counters
};

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <iostream>
#include <iterator>
#include <cmath>

ZMuMu_vtxAnalyzer::ZMuMu_vtxAnalyzer(const ParameterSet& pset) :
  zMuMuToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuMu"))),
  zMuMuMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuMuMatchMap"))),
  zMuStandAloneToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuStandAlone"))),
  zMuStandAloneMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuStandAloneMatchMap"))),
  zMuTrackToken_(consumes<CandidateView>(pset.getParameter<InputTag>("zMuTrack"))),
  zMuTrackMatchMapToken_(mayConsume<GenParticleMatch>(pset.getParameter<InputTag>("zMuTrackMatchMap"))),
  muonsToken_(consumes<CandidateView>(pset.getParameter<InputTag>("muons"))),
  tracksToken_(consumes<CandidateView>(pset.getParameter<InputTag>("tracks"))),
  genParticlesToken_(consumes<GenParticleCollection>(pset.getParameter<InputTag>("genParticles"))),
  primaryVerticesToken_(consumes<VertexCollection>(pset.getParameter<InputTag>("primaryVertices"))),

  bothMuons_(pset.getParameter<bool>("bothMuons")),

  etamax_(pset.getUntrackedParameter<double>("etamax")),
  ptmin_(pset.getUntrackedParameter<double>("ptmin")),
  massMin_(pset.getUntrackedParameter<double>("zMassMin")),
  massMax_(pset.getUntrackedParameter<double>("zMassMax")),
  isoMax_(pset.getUntrackedParameter<double>("isomax")) {
  Service<TFileService> fs;

  // general histograms

  // vertex histograms
  // ... zmumu no Cuts
  h_muon_vz = fs->make<TH1D>("muonVz","z vertex of muons",50,-20.,20.);
  h_muon_d0signed = fs->make<TH1D>("muonD0signed","d0 vertex of muons",50,-.1,.1);
  h_dimuon_vz = fs->make<TH1D>("dimuonVz","z vertex of dimuon",50,-20.,20.);
  h_muon_vz_respectToPV = fs->make<TH1D>("muonVz_respectToPV","z vertex of muons respect to PrimaryVertex",50,-.05,.05);
  h_muon_d0signed_respectToPV = fs->make<TH1D>("muonD0signed_respectToPV","d0 vertex of muons respect to PrimaryVertex",50,-.05,.05);
  // ... zmumu cynematic Cuts
  h_zmumuSele_muon_vz = fs->make<TH1D>("zmumuSele_muonVz","z vertex of muons (zmumu sele)",50,-20.,20.);
  h_zmumuSele_muon_d0signed = fs->make<TH1D>("zmumuSele_muonD0signed","d0 vertex of muons (zmumu sele)",50,-.1,.1);
  h_zmumuSele_dimuon_vz = fs->make<TH1D>("zmumuSele_dimuonVz","z vertex of dimuon (zmumu sele)",50,-20.,20.);
  h_zmumuSele_muon_vz_respectToPV = fs->make<TH1D>("zmumuSele_muonVz_respectToPV","z vertex of muons respect to PrimaryVertex (zmumu sele)",50,-.05,.05);
  h_zmumuSele_muon_d0signed_respectToPV = fs->make<TH1D>("zmumuSele_muonD0signed_respectToPV","d0 vertex of muons respect to PrimaryVertex (zmumu sele)",50,-.05,.05);
  // ... zmumuNotIso cynematic Cuts
  h_zmumuNotIsoSele_dimuon_vz = fs->make<TH1D>("zmumuNotIsoSele_dimuonVz","z vertex of dimuon (zmumuNotIso sele)",50,-20.,20.);
  h_zmumuNotIsoSele_muonIso_vz = fs->make<TH1D>("zmumuNotIsoSele_muonIsoVz","z vertex of muons (zmumuNotIso sele muon Iso)",50,-20.,20.);
  h_zmumuNotIsoSele_muonIso_d0signed = fs->make<TH1D>("zmumuNotIsoSele_muonIsoD0signed","d0 vertex of muons (zmumuNotIso sele muon Iso)",50,-.1,.1);
  h_zmumuNotIsoSele_muonIso_vz_respectToPV = fs->make<TH1D>("zmumuNotIsoSele_muonIsoVz_respectToPV","z vertex of muons respect to PrimaryVertex (zmumuNotIso sele muon Iso)",50,-.05,.05);
  h_zmumuNotIsoSele_muonIso_d0signed_respectToPV = fs->make<TH1D>("zmumuNotIsoSele_muonIsoD0signed_respectToPV","d0 vertex of muons respect to PrimaryVertex (zmumuNotIso sele muon Iso)",50,-.05,.05);
  h_zmumuNotIsoSele_muonNotIso_vz = fs->make<TH1D>("zmumuNotIsoSele_muonNotIsoVz","z vertex of muons (zmumuNotIso sele muon Not Iso)",50,-20.,20.);
  h_zmumuNotIsoSele_muonNotIso_d0signed = fs->make<TH1D>("zmumuNotIsoSele_muonNotIsoD0signed","d0 vertex of muons (zmumuNotIso sele muon Not Iso)",50,-.1,.1);
  h_zmumuNotIsoSele_muonNotIso_vz_respectToPV = fs->make<TH1D>("zmumuNotIsoSele_muonNotIsoVz_respectToPV","z vertex of muons respect to PrimaryVertex (zmumuNotIso sele muon Not Iso)",50,-.05,.05);
  h_zmumuNotIsoSele_muonNotIso_d0signed_respectToPV = fs->make<TH1D>("zmumuNotIsoSele_muonNotIsoD0signed_respectToPV","d0 vertex of muons respect to PrimaryVertex (zmumuNotIso sele muon Not Iso)",50,-.05,.05);
  // ... zmutrack cynematic Cuts
  h_zmutrackSele_muon_vz = fs->make<TH1D>("zmutrackSele_muonVz","z vertex of muon (zmutrack sele)",50,-20.,20.);
  h_zmutrackSele_muon_d0signed = fs->make<TH1D>("zmutrackSele_muonD0signed","d0 vertex of muon (zmutrack sele)",50,-.1,.1);
  h_zmutrackSele_muon_vz_respectToPV = fs->make<TH1D>("zmutrackSele_muonVz_respectToPV","z vertex of muon respect to PV (zmutrack sele)",50,-.05,.05);
  h_zmutrackSele_muon_d0signed_respectToPV = fs->make<TH1D>("zmutrackSele_muonD0signed_respectToPV","d0 vertex of muon respect to PV (zmutrack sele)",50,-.1,.1);
  h_zmutrackSele_track_vz = fs->make<TH1D>("zmutrackSele_trackVz","z vertex of track (zmutrack sele)",50,-20.,20.);
  h_zmutrackSele_track_d0signed = fs->make<TH1D>("zmutrackSele_trackD0signed","d0 vertex of track (zmutrack sele)",50,-.1,.1);
  h_zmutrackSele_track_vz_respectToPV = fs->make<TH1D>("zmutrackSele_trackVz_respectToPV","z vertex of track respect to PV (zmutrack sele)",50,-.05,.05);
  h_zmutrackSele_track_d0signed_respectToPV = fs->make<TH1D>("zmutrackSele_trackD0signed_respectToPV","d0 vertex of track respect to PV (zmutrack sele)",50,-.1,.1);
  // ... zmusta cynematic Cuts
  h_zmustaSele_muon_vz = fs->make<TH1D>("zmustaSele_muonVz","z vertex of muon (zmusta sele)",50,-20.,20.);
  h_zmustaSele_muon_d0signed = fs->make<TH1D>("zmustaSele_muonD0signed","d0 vertex of muon (zmusta sele)",50,-.1,.1);
  h_zmustaSele_muon_vz_respectToPV = fs->make<TH1D>("zmustaSele_muonVz_respectToPV","z vertex of muon respect to PV (zmusta sele)",50,-.05,.05);
  h_zmustaSele_muon_d0signed_respectToPV = fs->make<TH1D>("zmustaSele_muonD0signed_respectToPV","d0 vertex of muon respect to PV (zmusta sele)",50,-.1,.1);
  h_zmustaSele_sta_vz = fs->make<TH1D>("zmustaSele_staVz","z vertex of sta (zmusta sele)",50,-20.,20.);
  h_zmustaSele_sta_d0signed = fs->make<TH1D>("zmustaSele_staD0signed","d0 vertex of sta (zmusta sele)",50,-.1,.1);
  h_zmustaSele_sta_vz_respectToPV = fs->make<TH1D>("zmustaSele_staVz_respectToPV","z vertex of sta respect to PV (zmusta sele)",50,-.05,.05);
  h_zmustaSele_sta_d0signed_respectToPV = fs->make<TH1D>("zmustaSele_staD0signed_respectToPV","d0 vertex of sta respect to PV (zmusta sele)",50,-.1,.1);

}

void ZMuMu_vtxAnalyzer::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateView> zMuMu;
  Handle<GenParticleMatch> zMuMuMatchMap; //Map of Z made by Mu global + Mu global
  Handle<CandidateView> zMuStandAlone;
  Handle<GenParticleMatch> zMuStandAloneMatchMap; //Map of Z made by Mu + StandAlone
  Handle<CandidateView> zMuTrack;
  Handle<GenParticleMatch> zMuTrackMatchMap; //Map of Z made by Mu + Track
  Handle<CandidateView> muons; //Collection of Muons
  Handle<CandidateView> tracks; //Collection of Tracks

  Handle<GenParticleCollection> genParticles;  // Collection of Generatd Particles
  Handle<reco::VertexCollection> primaryVertices;  // Collection of primary Vertices

  event.getByToken(zMuMuToken_, zMuMu);
  event.getByToken(zMuStandAloneToken_, zMuStandAlone);
  event.getByToken(zMuTrackToken_, zMuTrack);
  event.getByToken(genParticlesToken_, genParticles);
  event.getByToken(primaryVerticesToken_, primaryVertices);
  event.getByToken(muonsToken_, muons);
  event.getByToken(tracksToken_, tracks);

  /*
  cout << "*********  zMuMu         size : " << zMuMu->size() << endl;
  cout << "*********  zMuStandAlone size : " << zMuStandAlone->size() << endl;
  cout << "*********  zMuTrack      size : " << zMuTrack->size() << endl;
  cout << "*********  muons         size : " << muons->size() << endl;
  cout << "*********  tracks        size : " << tracks->size() << endl;
  cout << "*********  vertices      size : " << primaryVertices->size() << endl;
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
      double trkiso0 = muonDau0.trackIso();
      const pat::Muon & muonDau1 = dynamic_cast<const pat::Muon &>(*lep1->masterClone());
      double trkiso1 = muonDau1.trackIso();

      // vertex
      h_muon_vz->Fill(muonDau0.vz());
      h_muon_vz->Fill(muonDau1.vz());
      h_dimuon_vz->Fill((muonDau0.vz()+muonDau1.vz())/2.);

      TrackRef mu0TrkRef = muonDau0.track();
      float d0signed_mu0 = (*mu0TrkRef).dxy();
      float d0signed_mu0_respectToPV= (*mu0TrkRef).dxy( primaryVertices->begin()->position() );
      float vz_mu0_respectToPV= (*mu0TrkRef).dz( primaryVertices->begin()->position() );

      TrackRef mu1TrkRef = muonDau1.track();
      float d0signed_mu1 = (*mu1TrkRef).dxy();
      float d0signed_mu1_respectToPV= (*mu1TrkRef).dxy( primaryVertices->begin()->position() );
      float vz_mu1_respectToPV= (*mu1TrkRef).dz( primaryVertices->begin()->position() );
      h_muon_d0signed->Fill(d0signed_mu0);
      h_muon_d0signed->Fill(d0signed_mu1);
      h_muon_d0signed_respectToPV->Fill(d0signed_mu0_respectToPV);
      h_muon_d0signed_respectToPV->Fill(d0signed_mu1_respectToPV);
      h_muon_vz_respectToPV->Fill(vz_mu0_respectToPV);
      h_muon_vz_respectToPV->Fill(vz_mu1_respectToPV);

      // eta , pt distributions
      double pt0 = zMuMuCand.daughter(0)->pt();
      double pt1 = zMuMuCand.daughter(1)->pt();
      double eta0 = zMuMuCand.daughter(0)->eta();
      double eta1 = zMuMuCand.daughter(1)->eta();
      double mass = zMuMuCand.mass();

      // HLT match
      const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	muonDau0.triggerObjectMatchesByPath( "HLT_Mu9" );
      const pat::TriggerObjectStandAloneCollection mu1HLTMatches =
	muonDau1.triggerObjectMatchesByPath( "HLT_Mu9" );

      bool trig0found = false;
      bool trig1found = false;
      if( mu0HLTMatches.size()>0 )
	trig0found = true;
      if( mu1HLTMatches.size()>0 )
	trig1found = true;

      // cynematical selection
      if ((trig0found || trig1found) && pt0>ptmin_ && pt1>ptmin_ && abs(eta0)<etamax_ && abs(eta1)<etamax_ && mass > massMin_) {
	if (trkiso0<isoMax_ && trkiso1<isoMax_) {  // zmumu both isolated
	  h_zmumuSele_muon_vz->Fill(muonDau0.vz());
	  h_zmumuSele_muon_vz->Fill(muonDau1.vz());
	  h_zmumuSele_dimuon_vz->Fill((muonDau0.vz()+muonDau1.vz())/2.);
	  h_zmumuSele_muon_d0signed->Fill(d0signed_mu0);
	  h_zmumuSele_muon_d0signed->Fill(d0signed_mu1);
	  h_zmumuSele_muon_d0signed_respectToPV->Fill(d0signed_mu0_respectToPV);
	  h_zmumuSele_muon_d0signed_respectToPV->Fill(d0signed_mu1_respectToPV);
	  h_zmumuSele_muon_vz_respectToPV->Fill(vz_mu0_respectToPV);
	  h_zmumuSele_muon_vz_respectToPV->Fill(vz_mu1_respectToPV);
	}
	if (trkiso0>=isoMax_ && trkiso1<isoMax_) {  // zmumu just muon1 isolated
	  h_zmumuNotIsoSele_muonNotIso_vz->Fill(muonDau0.vz());
	  h_zmumuNotIsoSele_muonIso_vz->Fill(muonDau1.vz());
	  h_zmumuNotIsoSele_dimuon_vz->Fill((muonDau0.vz()+muonDau1.vz())/2.);
	  h_zmumuNotIsoSele_muonNotIso_d0signed->Fill(d0signed_mu0);
	  h_zmumuNotIsoSele_muonIso_d0signed->Fill(d0signed_mu1);
	  h_zmumuNotIsoSele_muonNotIso_d0signed_respectToPV->Fill(d0signed_mu0_respectToPV);
	  h_zmumuNotIsoSele_muonIso_d0signed_respectToPV->Fill(d0signed_mu1_respectToPV);
	  h_zmumuNotIsoSele_muonNotIso_vz_respectToPV->Fill(vz_mu0_respectToPV);
	  h_zmumuNotIsoSele_muonIso_vz_respectToPV->Fill(vz_mu1_respectToPV);
	}
	if (trkiso0<isoMax_ && trkiso1>=isoMax_) {  // zmumu just muon0 isolated
	  h_zmumuNotIsoSele_muonNotIso_vz->Fill(muonDau1.vz());
	  h_zmumuNotIsoSele_muonIso_vz->Fill(muonDau0.vz());
	  h_zmumuNotIsoSele_dimuon_vz->Fill((muonDau1.vz()+muonDau1.vz())/2.);
	  h_zmumuNotIsoSele_muonNotIso_d0signed->Fill(d0signed_mu1);
	  h_zmumuNotIsoSele_muonIso_d0signed->Fill(d0signed_mu0);
	  h_zmumuNotIsoSele_muonNotIso_d0signed_respectToPV->Fill(d0signed_mu1_respectToPV);
	  h_zmumuNotIsoSele_muonIso_d0signed_respectToPV->Fill(d0signed_mu0_respectToPV);
	  h_zmumuNotIsoSele_muonNotIso_vz_respectToPV->Fill(vz_mu1_respectToPV);
	  h_zmumuNotIsoSele_muonIso_vz_respectToPV->Fill(vz_mu0_respectToPV);
	}
      }
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
      double trkiso0 = muonDau0.trackIso();
      const pat::Muon & muonDau1 = dynamic_cast<const pat::Muon &>(*lep1->masterClone());
      double trkiso1 = muonDau1.trackIso();

      // vertex

      TrackRef mu0TrkRef = muonDau0.track();
      float d0signed_mu0 = (*mu0TrkRef).dxy();
      float d0signed_mu0_respectToPV= (*mu0TrkRef).dxy( primaryVertices->begin()->position() );
      float vz_mu0_respectToPV= (*mu0TrkRef).dz( primaryVertices->begin()->position() );

      TrackRef mu1TrkRef = muonDau1.track();
      float d0signed_mu1 = (*mu1TrkRef).dxy();
      float d0signed_mu1_respectToPV= (*mu1TrkRef).dxy( primaryVertices->begin()->position() );
      float vz_mu1_respectToPV= (*mu1TrkRef).dz( primaryVertices->begin()->position() );

      double pt0 = zMuStandAloneCand.daughter(0)->pt();
      double pt1 = zMuStandAloneCand.daughter(1)->pt();
      double eta0 = zMuStandAloneCand.daughter(0)->eta();
      double eta1 = zMuStandAloneCand.daughter(1)->eta();
      double mass = zMuStandAloneCand.mass();

      // HLT match
      const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	muonDau0.triggerObjectMatchesByPath( "HLT_Mu9" );
      const pat::TriggerObjectStandAloneCollection mu1HLTMatches =
	muonDau1.triggerObjectMatchesByPath( "HLT_Mu9" );

      bool trig0found = false;
      bool trig1found = false;
      if( mu0HLTMatches.size()>0 )
	trig0found = true;
      if( mu1HLTMatches.size()>0 )
	trig1found = true;

      // check the global muon ... trigger is required just on global muon
      bool trigfound = false;
      if (muonDau0.isGlobalMuon()) trigfound = trig0found;
      if (muonDau1.isGlobalMuon()) trigfound = trig1found;

      // cynematical selection
      if (trigfound && pt0>ptmin_ && pt1>ptmin_ && abs(eta0)<etamax_ && abs(eta1)<etamax_ && mass>massMin_ && trkiso0<isoMax_ && trkiso1<isoMax_) {
	zMuSta_found = true;
	h_zmustaSele_muon_vz->Fill(muonDau0.vz(),1.);     // muon vz
	h_zmustaSele_sta_vz->Fill(muonDau1.vz(),1.);    // sta vz
	h_zmustaSele_muon_d0signed->Fill(d0signed_mu0,1.);   // muon d0
	h_zmustaSele_sta_d0signed->Fill(d0signed_mu1,1.);  // sta d0
	h_zmustaSele_muon_d0signed_respectToPV->Fill(d0signed_mu0_respectToPV,1.); // muon d0 respect PV
	h_zmustaSele_sta_d0signed_respectToPV->Fill(d0signed_mu1_respectToPV,1.); // sta d0 respect PV
	h_zmustaSele_muon_vz_respectToPV->Fill(vz_mu0_respectToPV,1.);             // muon vz respect PV
	h_zmustaSele_sta_vz_respectToPV->Fill(vz_mu1_respectToPV,1.);	          // sta vz respect PV
      }

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
      double trkiso0 = muonDau0.trackIso();
      const pat::GenericParticle & trackDau1 = dynamic_cast<const pat::GenericParticle &>(*lep1->masterClone());
      double trkiso1 = trackDau1.trackIso();

      // vertex

      TrackRef mu0TrkRef = muonDau0.track();
      float d0signed_mu0 = (*mu0TrkRef).dxy();
      float d0signed_mu0_respectToPV= (*mu0TrkRef).dxy( primaryVertices->begin()->position() );
      float vz_mu0_respectToPV= (*mu0TrkRef).dz( primaryVertices->begin()->position() );

      TrackRef mu1TrkRef = trackDau1.track();
      float d0signed_mu1 = (*mu1TrkRef).dxy();
      float d0signed_mu1_respectToPV= (*mu1TrkRef).dxy( primaryVertices->begin()->position() );
      float vz_mu1_respectToPV= (*mu1TrkRef).dz( primaryVertices->begin()->position() );

      // cynematical parameters

      double pt0 = zMuTrackCand.daughter(0)->pt();
      double pt1 = zMuTrackCand.daughter(1)->pt();
      double eta0 = zMuTrackCand.daughter(0)->eta();
      double eta1 = zMuTrackCand.daughter(1)->eta();
      double mass = zMuTrackCand.mass();

      // HLT match (check just dau0 the global)
       const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	muonDau0.triggerObjectMatchesByPath( "HLT_Mu9" );

      bool trig0found = false;
      if( mu0HLTMatches.size()>0 )
	trig0found = true;

      // cynematical selection
      if (trig0found && pt0>ptmin_ && pt1>ptmin_ && abs(eta0)<etamax_ && abs(eta1)<etamax_ && mass>massMin_ && trkiso0<isoMax_ && trkiso1<isoMax_) {
	h_zmutrackSele_muon_vz->Fill(muonDau0.vz(),1.);     // muon vz
	h_zmutrackSele_track_vz->Fill(trackDau1.vz(),1.);    // track vz
	h_zmutrackSele_muon_d0signed->Fill(d0signed_mu0,1.);   // muon d0
	h_zmutrackSele_track_d0signed->Fill(d0signed_mu1,1.);  // track d0
	h_zmutrackSele_muon_d0signed_respectToPV->Fill(d0signed_mu0_respectToPV,1.); // muon d0 respect PV
	h_zmutrackSele_track_d0signed_respectToPV->Fill(d0signed_mu1_respectToPV,1.); // track d0 respect PV
	h_zmutrackSele_muon_vz_respectToPV->Fill(vz_mu0_respectToPV,1.);             // muon vz respect PV
	h_zmutrackSele_track_vz_respectToPV->Fill(vz_mu1_respectToPV,1.);	          // track vz respect PV
      }

    }  // end loop on ZMuTrack cand
  }    // end if ZMuTrack size > 0

}       // end analyze

bool ZMuMu_vtxAnalyzer::check_ifZmumu(const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMu_vtxAnalyzer::getParticlePt(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMu_vtxAnalyzer::getParticleEta(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMu_vtxAnalyzer::getParticlePhi(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

Particle::LorentzVector ZMuMu_vtxAnalyzer::getParticleP4(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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



void ZMuMu_vtxAnalyzer::endJob() {

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMu_vtxAnalyzer);

