/* \class ZMuMu_efficieyAnalyzer
 *
 * author: Davide Piccolo
 *
 * ZMuMu efficiency analyzer:
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
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include <vector>

using namespace edm;
using namespace std;
using namespace reco;

typedef edm::ValueMap<float> IsolationCollection;

class ZMuMu_efficiencyAnalyzer : public edm::EDAnalyzer {
public:
  ZMuMu_efficiencyAnalyzer(const edm::ParameterSet& pset);
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

  // binning of entries array (at moment defined by hand and not in cfg file)
  unsigned int etaBins;
  unsigned int ptBins;
  double  etaRange[7];
  double  ptRange[5];

  reco::CandidateBaseRef globalMuonCandRef_, trackMuonCandRef_, standAloneMuonCandRef_;
  OverlapChecker overlap_;

  // general histograms
  TH1D *h_zmm_mass, *h_zmm2HLT_mass;
  TH1D *h_zmm1HLTplus_mass, *h_zmmNotIsoplus_mass, *h_zmsplus_mass, *h_zmtplus_mass;
  TH1D *h_zmm1HLTminus_mass, *h_zmmNotIsominus_mass, *h_zmsminus_mass, *h_zmtminus_mass;

  // global counters
  int nGlobalMuonsMatched_passed;    // total number of global muons MC matched and passing cuts (and triggered)

  vector<TH1D *>  hmumu2HLTplus_eta, hmumu1HLTplus_eta, hmustaplus_eta, hmutrackplus_eta, hmumuNotIsoplus_eta;
  vector<TH1D *>  hmumu2HLTplus_pt, hmumu1HLTplus_pt, hmustaplus_pt, hmutrackplus_pt, hmumuNotIsoplus_pt;
  vector<TH1D *>  hmumu2HLTminus_eta, hmumu1HLTminus_eta, hmustaminus_eta, hmutrackminus_eta, hmumuNotIsominus_eta;
  vector<TH1D *>  hmumu2HLTminus_pt, hmumu1HLTminus_pt, hmustaminus_pt, hmutrackminus_pt, hmumuNotIsominus_pt;
};

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <iostream>
#include <iterator>
#include <cmath>

ZMuMu_efficiencyAnalyzer::ZMuMu_efficiencyAnalyzer(const ParameterSet& pset) :
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
  h_zmm_mass  = fs->make<TH1D>("zmm_mass","zmumu mass",100,0.,200.);
  h_zmm2HLT_mass  = fs->make<TH1D>("zmm2HLT_mass","zmumu 2HLT mass",100,0.,200.);
  h_zmm1HLTplus_mass  = fs->make<TH1D>("zmm1HLTplus_mass","zmumu 1HLT plus mass",100,0.,200.);
  h_zmmNotIsoplus_mass  = fs->make<TH1D>("zmmNotIsoplus_mass","zmumu a least One Not Iso plus mass",100,0.,200.);
  h_zmsplus_mass  = fs->make<TH1D>("zmsplus_mass","zmusta plus mass",100,0.,200.);
  h_zmtplus_mass  = fs->make<TH1D>("zmtplus_mass","zmutrack plus mass",100,0.,200.);
  h_zmm1HLTminus_mass  = fs->make<TH1D>("zmm1HLTminus_mass","zmumu 1HLT minus mass",100,0.,200.);
  h_zmmNotIsominus_mass  = fs->make<TH1D>("zmmNotIsominus_mass","zmumu a least One Not Iso minus mass",100,0.,200.);
  h_zmsminus_mass  = fs->make<TH1D>("zmsminus_mass","zmusta minus mass",100,0.,200.);
  h_zmtminus_mass  = fs->make<TH1D>("zmtminus_mass","zmutrack minus mass",100,0.,200.);

  cout << "primo" << endl;
  // creating histograms for each Pt, eta interval

  TFileDirectory etaDirectory = fs->mkdir("etaIntervals");   // in this directory will be saved all the histos of different eta intervals
  TFileDirectory ptDirectory = fs->mkdir("ptIntervals");   // in this directory will be saved all the histos of different pt intervals

  // binning of entries array (at moment defined by hand and not in cfg file)
  etaBins = 6;
  ptBins = 4;
  double  etaRangeTmp[7] = {-2.,-1.2,-0.8,0.,0.8,1.2,2.};
  double  ptRangeTmp[5] = {20.,40.,60.,80.,100.};
  for (unsigned int i=0;i<=etaBins;i++) etaRange[i] = etaRangeTmp[i];
  for (unsigned int i=0;i<=ptBins;i++) ptRange[i] = ptRangeTmp[i];

  // eta histograms creation
  cout << "eta istograms creation " << endl;

  for (unsigned int i=0;i<etaBins;i++) {
    cout << " bin eta plus  " << i << endl;
    // muon plus
    double range0 = etaRange[i];
    double range1= etaRange[i+1];
    char ap[30], bp[50];
    sprintf(ap,"zmumu2HLTplus_etaRange%d",i);
    sprintf(bp,"zmumu2HLT plus mass eta Range %f to %f",range0,range1);
    cout << ap << "   " << bp << endl;
    hmumu2HLTplus_eta.push_back(etaDirectory.make<TH1D>(ap,bp,200,0.,200.));
    sprintf(ap,"zmumu1HLTplus_etaRange%d",i);
    sprintf(bp,"zmumu1HLT plus mass eta Range %f to %f",range0,range1);
    cout << ap << "   " << bp << endl;
    hmumu1HLTplus_eta.push_back(etaDirectory.make<TH1D>(ap,bp,200,0.,200.));
    sprintf(ap,"zmustaplus_etaRange%d",i);
    sprintf(bp,"zmusta plus mass eta Range %f to %f",range0,range1);
    cout << ap << "   " << bp << endl;
    hmustaplus_eta.push_back(etaDirectory.make<TH1D>(ap,bp,50,0.,200.));
    sprintf(ap,"zmutrackplus_etaRange%d",i);
    sprintf(bp,"zmutrack plus mass eta Range %f to %f",range0,range1);
    cout << ap << "   " << bp << endl;
    hmutrackplus_eta.push_back(etaDirectory.make<TH1D>(ap,bp,100,0.,200.));
    sprintf(ap,"zmumuNotIsoplus_etaRange%d",i);
    sprintf(bp,"zmumuNotIso plus mass eta Range %f to %f",range0,range1);
    cout << ap << "   " << bp << endl;
    hmumuNotIsoplus_eta.push_back(etaDirectory.make<TH1D>(ap,bp,100,0.,200.));
    // muon minus
    cout << " bin eta minus  " << i << endl;
    char am[30], bm[50];
    sprintf(am,"zmumu2HLTminus_etaRange%d",i);
    sprintf(bm,"zmumu2HLT minus mass eta Range %f to %f",range0,range1);
    cout << am << "   " << bm << endl;
    hmumu2HLTminus_eta.push_back(etaDirectory.make<TH1D>(am,bm,200,0.,200.));
    sprintf(am,"zmumu1HLTminus_etaRange%d",i);
    sprintf(bm,"zmumu1HLT minus mass eta Range %f to %f",range0,range1);
    cout << am << "   " << bm << endl;
    hmumu1HLTminus_eta.push_back(etaDirectory.make<TH1D>(am,bm,200,0.,200.));
    sprintf(am,"zmustaminus_etaRange%d",i);
    sprintf(bm,"zmusta minus mass eta Range %f to %f",range0,range1);
    cout << am << "   " << bm << endl;
    hmustaminus_eta.push_back(etaDirectory.make<TH1D>(am,bm,50,0.,200.));
    sprintf(am,"zmutrackminus_etaRange%d",i);
    sprintf(bm,"zmutrack minus mass eta Range %f to %f",range0,range1);
    cout << am << "   " << bm << endl;
    hmutrackminus_eta.push_back(etaDirectory.make<TH1D>(am,bm,100,0.,200.));
    sprintf(am,"zmumuNotIsominus_etaRange%d",i);
    sprintf(bm,"zmumuNotIso minus mass eta Range %f to %f",range0,range1);
    cout << am << "   " << bm << endl;
    hmumuNotIsominus_eta.push_back(etaDirectory.make<TH1D>(am,bm,100,0.,200.));
  }

  // pt histograms creation
  cout << "pt istograms creation " << endl;

  for (unsigned int i=0;i<ptBins;i++) {
    double range0 = ptRange[i];
    double range1= ptRange[i+1];
    // muon plus
    cout << " bin pt plus  " << i << endl;
    char ap1[30], bp1[50];
    sprintf(ap1,"zmumu2HLTplus_ptRange%d",i);
    sprintf(bp1,"zmumu2HLT plus mass pt Range %f to %f",range0,range1);
    cout << ap1 << "   " << bp1 << endl;
    hmumu2HLTplus_pt.push_back(ptDirectory.make<TH1D>(ap1,bp1,200,0.,200.));
    sprintf(ap1,"zmumu1HLTplus_ptRange%d",i);
    sprintf(bp1,"zmumu1HLT plus mass pt Range %f to %f",range0,range1);
    cout << ap1 << "   " << bp1 << endl;
    hmumu1HLTplus_pt.push_back(ptDirectory.make<TH1D>(ap1,bp1,200,0.,200.));
    sprintf(ap1,"zmustaplus_ptRange%d",i);
    sprintf(bp1,"zmusta plus mass pt Range %f to %f",range0,range1);
    cout << ap1 << "   " << bp1 << endl;
    hmustaplus_pt.push_back(ptDirectory.make<TH1D>(ap1,bp1,50,0.,200.));
    sprintf(ap1,"zmutrackplus_ptRange%d",i);
    sprintf(bp1,"zmutrack plus mass pt Range %f to %f",range0,range1);
    cout << ap1 << "   " << bp1 << endl;
    hmutrackplus_pt.push_back(ptDirectory.make<TH1D>(ap1,bp1,100,0.,200.));
    sprintf(ap1,"zmumuNotIsoplus_ptRange%d",i);
    sprintf(bp1,"zmumuNotIso plus mass pt Range %f to %f",range0,range1);
    cout << ap1 << "   " << bp1 << endl;
    hmumuNotIsoplus_pt.push_back(ptDirectory.make<TH1D>(ap1,bp1,100,0.,200.));
    // muon minus
    cout << " bin pt minus  " << i << endl;
    char am1[30], bm1[50];
    sprintf(am1,"zmumu2HLTminus_ptRange%d",i);
    sprintf(bm1,"zmumu2HLT minus mass pt Range %f to %f",range0,range1);
    cout << am1 << "   " << bm1 << endl;
    hmumu2HLTminus_pt.push_back(ptDirectory.make<TH1D>(am1,bm1,200,0.,200.));
    sprintf(am1,"zmumu1HLTminus_ptRange%d",i);
    sprintf(bm1,"zmumu1HLT minus mass pt Range %f to %f",range0,range1);
    cout << am1 << "   " << bm1 << endl;
    hmumu1HLTminus_pt.push_back(ptDirectory.make<TH1D>(am1,bm1,200,0.,200.));
    sprintf(am1,"zmustaminus_ptRange%d",i);
    sprintf(bm1,"zmusta minus mass pt Range %f to %f",range0,range1);
    cout << am1 << "   " << bm1 << endl;
    hmustaminus_pt.push_back(ptDirectory.make<TH1D>(am1,bm1,50,0.,200.));
    sprintf(am1,"zmutrackminus_ptRange%d",i);
    sprintf(bm1,"zmutrack minus mass pt Range %f to %f",range0,range1);
    cout << am1 << "   " << bm1 << endl;
    hmutrackminus_pt.push_back(ptDirectory.make<TH1D>(am1,bm1,100,0.,200.));
    sprintf(am1,"zmumuNotIsominus_ptRange%d",i);
    sprintf(bm1,"zmumuNotIso minus mass pt Range %f to %f",range0,range1);
    cout << am1 << "   " << bm1 << endl;
    hmumuNotIsominus_pt.push_back(ptDirectory.make<TH1D>(am1,bm1,100,0.,200.));
  }

  // clear global counters
  nGlobalMuonsMatched_passed = 0;
}

void ZMuMu_efficiencyAnalyzer::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateView> zMuMu;
  Handle<GenParticleMatch> zMuMuMatchMap; //Map of Z made by Mu global + Mu global
  Handle<CandidateView> zMuStandAlone;
  Handle<GenParticleMatch> zMuStandAloneMatchMap; //Map of Z made by Mu + StandAlone
  Handle<CandidateView> zMuTrack;
  Handle<GenParticleMatch> zMuTrackMatchMap; //Map of Z made by Mu + Track
  Handle<CandidateView> muons; //Collection of Muons
  Handle<CandidateView> tracks; //Collection of Tracks

  Handle<GenParticleCollection> genParticles;  // Collection of Generatd Particles
  Handle<VertexCollection> primaryVertices;  // Collection of primary Vertices

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
    for(unsigned int i = 0; i < zMuMu->size(); ++i) { //loop on candidates
      const Candidate & zMuMuCand = (*zMuMu)[i]; //the candidate
      CandidateBaseRef zMuMuCandRef = zMuMu->refAt(i);

      const Candidate * lep0 = zMuMuCand.daughter( 0 );
      const Candidate * lep1 = zMuMuCand.daughter( 1 );
      const pat::Muon & muonDau0 = dynamic_cast<const pat::Muon &>(*lep0->masterClone());
      double trkiso0 = muonDau0.trackIso();
      const pat::Muon & muonDau1 = dynamic_cast<const pat::Muon &>(*lep1->masterClone());
      double trkiso1 = muonDau1.trackIso();

      // kinemtic variables
      double pt0 = zMuMuCand.daughter(0)->pt();
      double pt1 = zMuMuCand.daughter(1)->pt();
      double eta0 = zMuMuCand.daughter(0)->eta();
      double eta1 = zMuMuCand.daughter(1)->eta();
      double charge0 = zMuMuCand.daughter(0)->charge();
      double charge1 = zMuMuCand.daughter(1)->charge();
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

      // kinematic selection

      bool checkOppositeCharge = false;
      if (charge0 != charge1) checkOppositeCharge = true;
      if (pt0>ptmin_ && pt1>ptmin_ && abs(eta0)<etamax_ && abs(eta1)<etamax_ && mass>massMin_ && mass<massMax_ && checkOppositeCharge) {
	if (trig0found || trig1found) { // at least one muon match HLT
	  zMuMu_found = true;           // Z found as global-global (so don't check Zms and Zmt)
	  if (trkiso0 < isoMax_ && trkiso1 < isoMax_) { // both muons are isolated
	    if (trig0found && trig1found) {

	      // ******************** category zmm 2 HLT ****************

	      h_zmm2HLT_mass->Fill(mass);
	      h_zmm_mass->Fill(mass);

	      // check the cynematics to fill correct histograms

	      for (unsigned int j=0;j<etaBins;j++) {  // eta Bins loop
		double range0 = etaRange[j];
		double range1= etaRange[j+1];

		// eta histograms

		if (eta0>=range0 && eta0<range1)
		  {
		    if (charge0<0) hmumu2HLTminus_eta[j]->Fill(mass);  // mu- in bin eta
		    if (charge0>0) hmumu2HLTplus_eta[j]->Fill(mass);  // mu+ in bin eta
		  }
		if (eta1>=range0 && eta1<range1)
		  {
		    if (charge1<0) hmumu2HLTminus_eta[j]->Fill(mass);  // mu- in bin eta
		    if (charge1>0) hmumu2HLTplus_eta[j]->Fill(mass);  // mu+ in bin eta
		  }
	      } // end loop etaBins

	      for (unsigned int j=0;j<ptBins;j++) {  // pt Bins loop
		double range0pt = ptRange[j];
		double range1pt = ptRange[j+1];
		// pt histograms
		if (pt0>=range0pt && pt0<range1pt)
		  {
		    if (charge0<0) hmumu2HLTminus_pt[j]->Fill(mass);  // mu- in bin eta
		    if (charge0>0) hmumu2HLTplus_pt[j]->Fill(mass);  // mu+ in bin eta
		  }
		if (pt1>=range0pt && pt1<range1pt)
		  {
		    if (charge1<0) hmumu2HLTminus_pt[j]->Fill(mass);  // mu- in bin eta
		    if (charge1>0) hmumu2HLTplus_pt[j]->Fill(mass);  // mu+ in bin eta
		  }
	      } // end loop  ptBins

	    }  // ******************* end category zmm 2 HLT ****************

	    if (!trig0found || !trig1found) {
	      // ****************** category zmm 1 HLT ******************
	      h_zmm_mass->Fill(mass);
	      double eta = 9999;
	      double pt = 9999;
	      double charge = 0;
	      if (trig0found) {
		eta = eta1;       // check  muon not HLT matched
		pt = pt1;
		charge = charge1;
	      } else {
		eta = eta0;
		pt =pt0;
		charge = charge0;
	      }
	      if (charge<0) h_zmm1HLTminus_mass->Fill(mass);
	      if (charge>0) h_zmm1HLTplus_mass->Fill(mass);

	      for (unsigned int j=0;j<etaBins;j++) {  // eta Bins loop
		double range0 = etaRange[j];
		double range1= etaRange[j+1];
		// eta histograms fill the bin of the muon not HLT matched
		if (eta>=range0 && eta<range1)
		  {
		    if (charge<0) hmumu1HLTminus_eta[j]->Fill(mass);
		    if (charge>0) hmumu1HLTplus_eta[j]->Fill(mass);
		  }
	      } // end loop etaBins
	      for (unsigned int j=0;j<ptBins;j++) {  // pt Bins loop
		double range0 = ptRange[j];
		double range1= ptRange[j+1];
		// pt histograms
		if (pt>=range0 && pt<range1)
		  {
		    if (charge<0) hmumu1HLTminus_pt[j]->Fill(mass);
		    if (charge>0) hmumu1HLTplus_pt[j]->Fill(mass);
		  }
	      } // end loop ptBins

	    } // ****************** end category zmm 1 HLT ***************

	  } else {  // one or both muons are not isolated
	    // *****************  category zmumuNotIso **************** (per ora non studio iso vs eta e pt da capire meglio)

	  } // end if both muons isolated

	} // end if at least 1 HLT trigger found
      }  // end if kinematic selection


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
      double pt0 = zMuStandAloneCand.daughter(0)->pt();
      double pt1 = zMuStandAloneCand.daughter(1)->pt();
      double eta0 = zMuStandAloneCand.daughter(0)->eta();
      double eta1 = zMuStandAloneCand.daughter(1)->eta();
      double charge0 = zMuStandAloneCand.daughter(0)->charge();
      double charge1 = zMuStandAloneCand.daughter(1)->charge();
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

      // check HLT match of Global muon and save eta, pt of second muon (standAlone)
      bool trigGlbfound = false;
      double pt =999.;
      double eta = 999.;
      double charge = 0;
      if (muonDau0.isGlobalMuon()) {
	trigGlbfound = trig0found;
	pt = pt1;
	eta = eta1;
	charge = charge1;
      }
      if (muonDau1.isGlobalMuon()) {
	trigGlbfound = trig1found;
	pt = pt0;
	eta = eta0;
	charge = charge0;
      }

      bool checkOppositeCharge = false;
      if (charge0 != charge1) checkOppositeCharge = true;

      if (checkOppositeCharge && trigGlbfound && pt0>ptmin_ && pt1>ptmin_ && abs(eta0)<etamax_ && abs(eta1)<etamax_ && mass>massMin_ && mass<massMax_ && trkiso0<isoMax_ && trkiso1<isoMax_ ) {  // global mu match HLT + kinematic cuts + opposite charge

	if (charge<0) h_zmsminus_mass->Fill(mass);
	if (charge>0) h_zmsplus_mass->Fill(mass);

	for (unsigned int j=0;j<etaBins;j++) {  // eta Bins loop
	  double range0 = etaRange[j];
	  double range1= etaRange[j+1];
	  // eta histograms
	  if (eta>=range0 && eta<range1) {
	    if (charge<0)  hmustaminus_eta[j]->Fill(mass);
	    if (charge>0)  hmustaplus_eta[j]->Fill(mass);
	  }
	} // end loop etaBins
	for (unsigned int j=0;j<ptBins;j++) {  // pt Bins loop
	  double range0 = ptRange[j];
	  double range1= ptRange[j+1];
	  // pt histograms
	  if (pt>=range0 && pt<range1) {
	    if (charge<0)  hmustaminus_pt[j]->Fill(mass);
	    if (charge>0)  hmustaplus_pt[j]->Fill(mass);
	  }
	} // end loop ptBins

      } // end if trigGlbfound + kinecuts + OppostieCharge
    }  // end loop on ZMuStandAlone cand
  }    // end if ZMuStandAlone size > 0


  // loop on ZMuTrack
  //  bool zMuTrack_found = false;
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
      double pt0 = zMuTrackCand.daughter(0)->pt();
      double pt1 = zMuTrackCand.daughter(1)->pt();
      double eta0 = zMuTrackCand.daughter(0)->eta();
      double eta1 = zMuTrackCand.daughter(1)->eta();
      double charge0 = zMuTrackCand.daughter(0)->charge();
      double charge1 = zMuTrackCand.daughter(1)->charge();
      double mass = zMuTrackCand.mass();

      // HLT match (check just dau0 the global)
      const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	muonDau0.triggerObjectMatchesByPath( "HLT_Mu9" );

      bool trig0found = false;
      if( mu0HLTMatches.size()>0 )
	trig0found = true;

      bool checkOppositeCharge = false;
      if (charge0 != charge1) checkOppositeCharge = true;

      if (checkOppositeCharge && trig0found && pt0>ptmin_ && pt1>ptmin_ && abs(eta0)<etamax_ && abs(eta1)<etamax_ && mass>massMin_ && mass<massMax_ && trkiso0<isoMax_ && trkiso1<isoMax_ ) {  // global mu match HLT + kinematic cuts + opposite charge

	if (charge1<0) h_zmtminus_mass->Fill(mass);
	if (charge1>0) h_zmtplus_mass->Fill(mass);

	for (unsigned int j=0;j<etaBins;j++) {  // eta Bins loop
	  double range0 = etaRange[j];
	  double range1= etaRange[j+1];
	  // eta histograms
	  if (eta1>=range0 && eta1<range1) {
	    if (charge1<0)  hmutrackminus_eta[j]->Fill(mass);  // just check muon1 (mu0 is global by definition)
	    if (charge1>0)  hmutrackplus_eta[j]->Fill(mass);  // just check muon1 (mu0 is global by definition)
	  }
	} // end loop etaBins
	for (unsigned int j=0;j<ptBins;j++) {  // pt Bins loop
	  double range0 = ptRange[j];
	  double range1= ptRange[j+1];
	  // pt histograms
	  if (pt1>=range0 && pt1<range1) {
	    if (charge1<0)  hmutrackminus_pt[j]->Fill(mass);  // just check muon1 (mu0 is global by definition)
	    if (charge1>0)  hmutrackplus_pt[j]->Fill(mass);  // just check muon1 (mu0 is global by definition)
	  }
	} // end loop ptBins

      } // end if trig0found


    }  // end loop on ZMuTrack cand
  }    // end if ZMuTrack size > 0

}       // end analyze

bool ZMuMu_efficiencyAnalyzer::check_ifZmumu(const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMu_efficiencyAnalyzer::getParticlePt(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMu_efficiencyAnalyzer::getParticleEta(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMu_efficiencyAnalyzer::getParticlePhi(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

Particle::LorentzVector ZMuMu_efficiencyAnalyzer::getParticleP4(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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



void ZMuMu_efficiencyAnalyzer::endJob() {



}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMu_efficiencyAnalyzer);

