/* \class ZMuMuPerformances
 *
 * author: Davide Piccolo
 *
 * ZMuMu Performances:
 * check charge mis-id for standAlone and global muons,
 * check standAlne resolution vs track resolution
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
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
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

typedef ValueMap<float> IsolationCollection;

class ZMuMuPerformances : public edm::EDAnalyzer {
public:
  ZMuMuPerformances(const edm::ParameterSet& pset);
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
  EDGetTokenT<GenParticleCollection> genParticlesToken_;

  bool noCut_;
  double zMassMin_, zMassMax_;
  double ptminPlus_, ptmaxPlus_, etaminPlus_, etamaxPlus_;
  double ptminMinus_, ptmaxMinus_, etaminMinus_, etamaxMinus_, isomax_;

  double etamax_, ptmin_, massMin_, massMax_, isoMax_;

  reco::CandidateBaseRef globalMuonCandRef_, trackMuonCandRef_, standAloneMuonCandRef_;
  OverlapChecker overlap_;

  // general histograms
  TH1D *h_n_globalMuon_perEvent, *h_n_staOnlyMuon_perEvent, *h_n_trackerOnlyMuon_perEvent, *h_n_trackerStaOnlyMuon_perEvent;
  TH1D *h_n_globalMuon_perEvent_MCmatch, *h_n_staOnlyMuon_perEvent_MCmatch, *h_n_trackerOnlyMuon_perEvent_MCmatch;
  TH1D *h_n_trackerStaOnlyMuon_perEvent_MCmatch, *h_n_tracks_perEvent;
  TH1D *h_n_zMuMu_perEvent, *h_n_zMuSta_perEvent, *h_n_zMuTrack_perEvent;

  // zMuMu inv mass
  TH1D *h_zMuMuMassSameSign, *h_zMuMuMassSameSign_MCmatch,*h_zMuMuMassOppositeSign;
  // histograms with MC truth
  // charge truth
  TH1D *h_GlobalMuonChargeTimeGenCharge,*h_TrackerMuonChargeTimeGenCharge;
  // resolution respect to gen particles
  TH1D *h_GlobalMuonEtaMinusGenEta,*h_TrackerMuonEtaMinusGenEta,*h_GlobalMuonPtMinusGenPt,*h_TrackerMuonPtMinusGenPt;
  TH1D *h_GlobalMuonStaComponentEtaMinusGenEta, *h_GlobalMuonStaComponentPtMinusGenPt;
  TH2D *h_DEtaGlobalGenvsEtaGen, *h_DPtGlobalGenvsPtGen, *h_DEtaGlobalStaComponentGenvsEtaGen,*h_DPtGlobalStaComponentGenvsPtGen;
  TH2D *h_DPtGlobalGenvsEtaGen, *h_DPtGlobalStaComponentGenvsEtaGen;
  // resolution respect to gen particles for ZMuMuTagged events
  TH1D *h_GlobalMuonEtaMinusGenEta_ZMuMuTagged;
  TH1D *h_GlobalMuonPtMinusGenPt_ZMuMuTagged;
  TH1D *h_GlobalMuonStaComponentEtaMinusGenEta_ZMuMuTagged, *h_GlobalMuonStaComponentPtMinusGenPt_ZMuMuTagged;
  TH2D *h_DEtaGlobalGenvsEtaGen_ZMuMuTagged, *h_DPtGlobalGenvsPtGen_ZMuMuTagged;
  TH2D *h_DEtaGlobalStaComponentGenvsEtaGen_ZMuMuTagged,*h_DPtGlobalStaComponentGenvsPtGen_ZMuMuTagged;
  TH2D *h_DPtGlobalGenvsEtaGen_ZMuMuTagged, *h_DPtGlobalStaComponentGenvsEtaGen_ZMuMuTagged;
  TH2D *h_DPtTrackGenvsPtGen_ZMuMuTagged, *h_DPtTrackGenvsEtaGen_ZMuMuTagged;

  // histograms for cynematic of ZMuMutagged muons for STA performances studies
  TH1D *h_zMuTrackMass_ZMuMuTagged, *h_etaTrack_ZMuMuTagged, *h_phiTrack_ZMuMuTagged, *h_ptTrack_ZMuMuTagged, *h_DRTrack_ZMuMuTagged;
  // histograms for cynematic of ZMuMutagged muons when StandAlone has wrong charge
  TH1D *h_zMuTrackMass_wrongStaCharge_ZMuMuTagged, *h_etaTrack_wrongStaCharge_ZMuMuTagged;
  TH1D *h_phiTrack_wrongStaCharge_ZMuMuTagged, *h_ptTrack_wrongStaCharge_ZMuMuTagged, *h_DRTrack_wrongStaCharge_ZMuMuTagged;

  // hisograms for performances of Standlone when Sta has correct charge
  TH1D *h_zMuStaMass_correctStaCharge_ZMuMuTagged, *h_ptStaMinusptTrack_correctStaCharge_ZMuMuTagged;
  TH2D *h_ptStaMinusptTrack_vsEtaTracker_correctStaCharge_ZMuMuTagged;
  TH2D *h_ptStaMinusptTrack_vsPtTracker_correctStaCharge_ZMuMuTagged;

  // histograms for cynematic of ZMuMutagged muons for TRK performances studies
  TH1D *h_zMuStaMass_ZMuMuTagged, *h_etaSta_ZMuMuTagged, *h_phiSta_ZMuMuTagged, *h_ptSta_ZMuMuTagged, *h_DRSta_ZMuMuTagged;
  // histograms for cynematic of ZMuMutagged muons when TRK has wrong charge
  TH1D *h_zMuStaMass_wrongTrkCharge_ZMuMuTagged, *h_etaSta_wrongTrkCharge_ZMuMuTagged;
  TH1D *h_phiSta_wrongTrkCharge_ZMuMuTagged, *h_ptSta_wrongTrkCharge_ZMuMuTagged, *h_DRSta_wrongTrkCharge_ZMuMuTagged;

  // histograms for cynematic of ZMuTracktagged muons with unMatchd StandAlone for STA performances studies
  TH1D *h_zMuTrackMass_ZMuTrackTagged, *h_etaTrack_ZMuTrackTagged, *h_phiTrack_ZMuTrackTagged, *h_ptTrack_ZMuTrackTagged, *h_DRTrack_ZMuTrackTagged;
  // histograms for cynematic of ZMuTracktagged muons when unMatched StandAlone has wrong charge
  TH1D *h_zMuTrackMass_wrongStaCharge_ZMuTrackTagged, *h_etaTrack_wrongStaCharge_ZMuTrackTagged;
  TH1D *h_phiTrack_wrongStaCharge_ZMuTrackTagged, *h_ptTrack_wrongStaCharge_ZMuTrackTagged, *h_DRTrack_wrongStaCharge_ZMuTrackTagged;

  // histograms for cynematic of ZMuStatagged muons with unMatchd Track for Track performances studies
  TH1D *h_zMuStaMass_ZMuStaTagged, *h_etaSta_ZMuStaTagged, *h_phiSta_ZMuStaTagged, *h_ptSta_ZMuStaTagged;
  // histograms for cynematic of ZMuStatagged muons when unMatched Track has wrong charge
  TH1D *h_zMuStaMass_wrongTrkCharge_ZMuStaTagged, *h_etaSta_wrongTrkCharge_ZMuStaTagged;
  TH1D *h_phiSta_wrongTrkCharge_ZMuStaTagged, *h_ptSta_wrongTrkCharge_ZMuStaTagged;


  // global counters
  int totalNumberOfZfound;          // total number of events with Z found
  int totalNumberOfZpassed;         // total number of Z that pass cynematical cuts at generator level

  int nZMuMuSameSign;               // number of ZMuMu SameSIgn (no Cuts)
  int nZMuMuSameSign_mcMatched;     // number of ZMuMu Same Sign (no cuts) MCmatch

  int n_goodTrack_ZMuMutagged;      // total number of tracks selected and tagged to study Sta charge
  int n_correctStaCharge_ZMuMutagged;  // total number of tracks selected and tagged with correct charge of Sta
  int n_wrongStaCharge_ZMuMutagged;  // total number of tracks selected and tagged with wrong charge of Sta

  int n_goodSta_ZMuMutagged;          // total number of standAlone selected and tagged to study Trk charge
  int n_correctTrkCharge_ZMuMutagged;  // total number of standAlone selected and tagged with correct charge of Trk
  int n_wrongTrkCharge_ZMuMutagged;  // total number of standAlone selected and tagged with wrong charge of Trk

  int n_goodTrack_ZMuTracktagged;   // number of traks selected and tagged to study Sta charge (for ZMuTrack colllection no ZMuMu found)
  int n_correctStaCharge_ZMuTracktagged;  // total number of tracks selected and tagged with correct charge of unMatched Sta
  int n_wrongStaCharge_ZMuTracktagged;  // total number of tracks selected and tagged with wrong charge of unMatched Sta
  int n_StaNotFound_ZMuTracktagged;  // total number of tracks selected and tagged with no STA found

  int n_goodSta_ZMuStatagged;   // number of sta selected and tagged to study Trk charge (for ZMuSta collection no ZMuMu found)
  int n_correctTrkCharge_ZMuStatagged;  // total number of sta selected and tagged with correct charge of unMatched track
  int n_wrongTrkCharge_ZMuStatagged;  // total number of sta selected and tagged with wrong charge of unMatched track
  int n_TrkNotFound_ZMuStatagged;  // total number of selected selected and tagged with no Trk found

  int n_OneGoodZMuTrack;            // total number with just 1 good ZMuTrack found
  int n_MultipleGoodZMuTrack;       // total number with more than 1 good ZMuTrack found
  int numberOfMatchedZMuSta_;
  int n_ZMuStaTaggedMatched;
};

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <iostream>
#include <iterator>
#include <cmath>

ZMuMuPerformances::ZMuMuPerformances(const ParameterSet& pset) :
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
  genParticlesToken_(consumes<GenParticleCollection>(pset.getParameter<InputTag>( "genParticles"))),

  noCut_(pset.getParameter<bool>("noCut")),

  zMassMin_(pset.getUntrackedParameter<double>("zMassMin")),
  zMassMax_(pset.getUntrackedParameter<double>("zMassMax")),
  ptminPlus_(pset.getUntrackedParameter<double>("ptminPlus")),
  ptmaxPlus_(pset.getUntrackedParameter<double>("ptmaxPlus")),
  etaminPlus_(pset.getUntrackedParameter<double>("etaminPlus")),
  etamaxPlus_(pset.getUntrackedParameter<double>("etamaxPlus")),
  ptminMinus_(pset.getUntrackedParameter<double>("ptminMinus")),
  ptmaxMinus_(pset.getUntrackedParameter<double>("ptmaxMinus")),
  etaminMinus_(pset.getUntrackedParameter<double>("etaminMinus")),
  etamaxMinus_(pset.getUntrackedParameter<double>("etamaxMinus")),
  isomax_(pset.getUntrackedParameter<double>("isomax")) {
  Service<TFileService> fs;

  // cut setting
  etamax_ = etamaxPlus_;
  ptmin_ = ptminPlus_;
  massMin_ = zMassMin_;
  massMax_ = zMassMax_;
  isoMax_ = isomax_;

  // general histograms
  h_n_globalMuon_perEvent = fs->make<TH1D>("n_globalMuon_perEvent","n.of globalMuons per Event",6,-.5,5.5);
  h_n_staOnlyMuon_perEvent = fs->make<TH1D>("n_staOnlyMuon_perEvent","n.of standAlone Only Muons per Event",6,-.5,5.5);
  h_n_trackerOnlyMuon_perEvent = fs->make<TH1D>("n_trackerOnlyMuon_perEvent","n.of tracker Only Muons per Event",6,-.5,5.5);
  h_n_trackerStaOnlyMuon_perEvent = fs->make<TH1D>("n_trackerStaOnlyMuon_perEvent","n.of tracker & StandAlone Only Muons per Event",6,-.5,5.5);
  h_n_globalMuon_perEvent_MCmatch = fs->make<TH1D>("n_globalMuon_perEvent_MCmatch","n.of globalMuons per Event (MCmatch)",6,-.5,5.5);
  h_n_staOnlyMuon_perEvent_MCmatch = fs->make<TH1D>("n_staOnlyMuon_perEvent_MCmatch","n.of standAlone Only Muons per Event (MCmatch)",6,-.5,5.5);
  h_n_trackerOnlyMuon_perEvent_MCmatch = fs->make<TH1D>("n_trackerOnlyMuon_perEvent_MCmatch","n.of tracker Only Muons per Event (MCmatch)",6,-.5,5.5);
  h_n_trackerStaOnlyMuon_perEvent_MCmatch = fs->make<TH1D>("n_trackerStaOnlyMuon_perEvent_MCmatch","n.of tracker & StandAlone Only Muons per Event (MCmatch)",6,-.5,5.5);
  h_n_tracks_perEvent = fs->make<TH1D>("n_tracks_perEvent","n.of tracks per Event",100,-.5,99.5);
  h_n_zMuMu_perEvent = fs->make<TH1D>("n_zMuMu_perEvent","n.of global-global muons per Event",6,-.5,5.5);
  h_n_zMuSta_perEvent = fs->make<TH1D>("n_zMuSta_perEvent","n.of global-sta muons per Event",6,-.5,5.5);
  h_n_zMuTrack_perEvent = fs->make<TH1D>("n_zMuTrack_perEvent","n.of global-track muons per Event",100,-.5,99.5);

  // zMuMu inv mass
  h_zMuMuMassSameSign = fs->make<TH1D>("zMuMuMassSameSign","inv Mass ZMuMu cand SameSign",100, 0., 200.);
  h_zMuMuMassOppositeSign = fs->make<TH1D>("zMuMuMassOppositeSign","inv Mass ZMuMu cand OppositeSign",100, 0., 200.);
  h_zMuMuMassSameSign_MCmatch = fs->make<TH1D>("zMuMuMassSameSign_MCmatch","inv Mass ZMuMu cand SameSign (MC match)",100, 0., 200.);

  // histograms for MC truth
  // charge truth
  h_GlobalMuonChargeTimeGenCharge = fs->make<TH1D>("GlobalMuonChargeTimeGenCharge","charge global mu times charge generated mu",3, -1.5, 1.5);
  h_TrackerMuonChargeTimeGenCharge = fs->make<TH1D>("TrackerMuonChargeTimeGenCharge","charge Tracker mu times charge generated mu",3, -1.5, 1.5);
  // resolution respect to gen particles
  h_GlobalMuonEtaMinusGenEta = fs->make<TH1D>("GlobalMuonEtaMinusGenEta","global mu Eta minus generated mu Eta",100, -.005, .005);
  h_GlobalMuonPtMinusGenPt = fs->make<TH1D>("GlobalMuonPtMinusGenPtoverPt","global mu Pt minus generated mu Pt over Pt",100, -.5, .5);
  h_GlobalMuonStaComponentEtaMinusGenEta = fs->make<TH1D>("GlobalMuonStaComponentEtaMinusGenEta","global mu Sta cmponent Eta minus generated mu Eta",100, -.5, .5);
  h_GlobalMuonStaComponentPtMinusGenPt = fs->make<TH1D>("GlobalMuonStaComponentPtMinusGenPtoerPt","global mu Sta component Pt minus generated mu Pt over Pt",100, -1., 1.);
  h_TrackerMuonEtaMinusGenEta = fs->make<TH1D>("TrackerMuonEtaMinusGenEta","Tracker mu Eta minus Eta generated mu",100, -.005, .005);
  h_TrackerMuonPtMinusGenPt = fs->make<TH1D>("TrackerMuonPtMinusenPtoverPt","Tracker mu Pt minus Pt generated mu over Pt",100, -.5, .5);

  h_DEtaGlobalGenvsEtaGen = fs->make<TH2D>("h_DEtaGlobalGenvsEtaGen","Eta global - Eta Gen vs Eta gen",50,-2.5,2.5,100,-.005,.005);
  h_DEtaGlobalStaComponentGenvsEtaGen = fs->make<TH2D>("h_DEtaGlobalStaComponentGenvsEtaGen","Eta Sta component of a Global - Eta Gen vs Eta gen",50,-2.5,2.5,100,-.5,.5);
  h_DPtGlobalGenvsPtGen = fs->make<TH2D>("h_DPtGlobalGenovePtvsPtGen","Pt global - Pt Gen over Pt vs Pt gen",50,0.,100.,100,-.5,.5);
  h_DPtGlobalStaComponentGenvsPtGen = fs->make<TH2D>("h_DPtGlobalStaComponentGenoverPtvsPtGen","Pt Sta component of a Global - Pt Gen over Pt vs Pt gen",50,0.,100.,100,-1.,1.);

  // resolution respect to gen particles for ZMuMuTagged events
  h_GlobalMuonEtaMinusGenEta_ZMuMuTagged = fs->make<TH1D>("GlobalMuonEtaMinusGenEta_ZMuMuTagged","global mu Eta minus generated mu Eta",100, -.005, .005);
  h_GlobalMuonPtMinusGenPt_ZMuMuTagged = fs->make<TH1D>("GlobalMuonPtMinusGenPtoverPt_ZMuMuTagged","global mu Pt minus generated mu Pt over Pt",100, -.5, .5);
  h_GlobalMuonStaComponentEtaMinusGenEta_ZMuMuTagged = fs->make<TH1D>("GlobalMuonStaComponentEtaMinusGenEta_ZMuMuTagged","global mu Sta cmponent Eta minus generated mu Eta",100, -.5, .5);
  h_GlobalMuonStaComponentPtMinusGenPt_ZMuMuTagged = fs->make<TH1D>("GlobalMuonStaComponentPtMinusGenPtoverPt_ZMuMuTagged","global mu Sta component Pt minus generated mu Pt over Pt",100, -1., 1.);
  h_DEtaGlobalGenvsEtaGen_ZMuMuTagged = fs->make<TH2D>("h_DEtaGlobalGenvsEtaGen_ZMuMuTagged","Eta global - Eta Gen vs Eta gen",50,-2.5,2.5,100,-.005,.005);
  h_DEtaGlobalStaComponentGenvsEtaGen_ZMuMuTagged = fs->make<TH2D>("h_DEtaGlobalStaComponentGenvsEtaGen_ZMuMuTagged","Eta Sta component of a Global - Eta Gen vs Eta gen",50,-2.5,2.5,100,-.5,.5);
  h_DPtGlobalGenvsPtGen_ZMuMuTagged = fs->make<TH2D>("h_DPtGlobalGenOverPtvsPtGen_ZMuMuTagged","Pt global - Pt Gen vs Pt gen over Pt",50,0.,100.,100,-.5,.5);
  h_DPtGlobalStaComponentGenvsPtGen_ZMuMuTagged = fs->make<TH2D>("h_DPtGlobalStaComponentGenoverPtvsPtGen_ZMuMuTagged","Pt Sta component of a Global - Pt Gen over Pt vs Pt gen",50,0.,100.,100,-1.,1.);
  h_DPtGlobalGenvsEtaGen_ZMuMuTagged = fs->make<TH2D>("h_DPtGlobalGenOverPtvsEtaGen_ZMuMuTagged","Pt global - Pt Gen over Pt vs Eta gen",50,-2.5,2.5,100,-.5,.5);
  h_DPtGlobalStaComponentGenvsEtaGen_ZMuMuTagged = fs->make<TH2D>("h_DPtGlobalStaComponentGenoverPtvsEtaGen_ZMuMuTagged","Pt Sta component of a Global - Pt Gen over Pt vs Eta gen",50,-2.5,2.5,100,-1.,1.);
  h_DPtTrackGenvsPtGen_ZMuMuTagged = fs->make<TH2D>("h_DPtTrackGenOverPtvsPtGen_ZMuMuTagged","Pt track - Pt Gen vs Pt gen over Pt",50,0.,100.,100,-.5,.5);
  h_DPtTrackGenvsEtaGen_ZMuMuTagged = fs->make<TH2D>("h_DPtTrackGenOverPtvsEtaGen_ZMuMuTagged","Pt track - Pt Gen over Pt vs Eta gen",50,-2.5,2.5,100,-.5,.5);

  // histograms for cynematic of ZMuMutagged muons for Sta performances studies
  h_zMuTrackMass_ZMuMuTagged = fs->make<TH1D>("zMuTrackMass_ZMuMuTagged","inv Mass ZMuTrack cand (global-global)",100, 0., 200.);
  h_etaTrack_ZMuMuTagged = fs->make<TH1D>("etaTrack_ZMuMuTagged","eta of Track (global-global)",50, -2.5, 2.5);
  h_phiTrack_ZMuMuTagged = fs->make<TH1D>("phiTrack_ZMuMuTagged","phi of Track (global-global)",50, -3.1415, 3.1415);
  h_ptTrack_ZMuMuTagged = fs->make<TH1D>("ptTrack_ZMuMuTagged","pt of Track (global-global)",100, 0., 100.);
  h_DRTrack_ZMuMuTagged = fs->make<TH1D>("DRTrackSta_ZMuMuTagged","DR track-sta (global-global)",100, 0., 5.);

  // histograms for cynematic of ZMuMutagged muons when StandAlone has wrong charge
  h_zMuTrackMass_wrongStaCharge_ZMuMuTagged = fs->make<TH1D>("zMuTrackMass_wrongStaCharge_ZMuMuTagged","inv Mass ZMuTrack cand (global-global wrongStaCharge)",100, 0., 200.);
  h_etaTrack_wrongStaCharge_ZMuMuTagged = fs->make<TH1D>("etaTrack_wrongStaCharge_ZMuMuTagged","eta of Track (global-global wrongStaCharge)",50, -2.5, 2.5);
  h_phiTrack_wrongStaCharge_ZMuMuTagged = fs->make<TH1D>("phiTrack_wrongStaCharge_ZMuMuTagged","phi of Track (global-global wrongStaCharge)",50, -3.1415, 3.1415);
  h_ptTrack_wrongStaCharge_ZMuMuTagged = fs->make<TH1D>("ptTrack_wrongStaCharge_ZMuMuTagged","pt of Track (global-global wrongStaCharge)",100, 0., 100.);
  h_DRTrack_wrongStaCharge_ZMuMuTagged = fs->make<TH1D>("DRTrackSta_wrongStaCharge_ZMuMuTagged","DR track-sta (global-global wrongStaCharge)",100, 0., 5.);

  // hisograms for performances of StandAlone when StandAlone has correct charge
  h_zMuStaMass_correctStaCharge_ZMuMuTagged = fs->make<TH1D>("zMuStaMass_correctStaCharge_ZMuMuTagged","inv Mass ZMuSta cand (global-global correctStaCharge)",100, 0., 200.);
  h_ptStaMinusptTrack_correctStaCharge_ZMuMuTagged = fs->make<TH1D>("ptStaMinusptTrackoverPT_correctStaCharge_ZMuMuTagged","ptSta - ptTrack over Pt (global-global correctStaCharge)",100, -1., 1.);
  h_ptStaMinusptTrack_vsPtTracker_correctStaCharge_ZMuMuTagged = fs->make<TH2D>("ptStaMinusptTrackoverPt_vsPtTracker_correctStaCharge_ZMuMuTagged","ptSta - ptTrack over Pt vs ptTrack (global-global correctStaCharge)",100,0.,100.,100, -1., 1.);
  h_ptStaMinusptTrack_vsEtaTracker_correctStaCharge_ZMuMuTagged = fs->make<TH2D>("ptStaMinusptTrackoverPt_vsEtaTracker_correctStaCharge_ZMuMuTagged","ptSta - ptTrack over Pt vs etaTrack (global-global correctStaCharge)",100,-2.5, 2.5, 100, -1., 1.);

  // histograms for cynematic of ZMuMutagged muons for TRK performances studies
  h_zMuStaMass_ZMuMuTagged = fs->make<TH1D>("zMuStaMass_ZMuMuTagged","inv Mass ZMuSta cand (global-global)",100, 0., 200.);
  h_etaSta_ZMuMuTagged = fs->make<TH1D>("etaSta_ZMuMuTagged","eta of Sta (global-global)",50, -2.5, 2.5);
  h_phiSta_ZMuMuTagged = fs->make<TH1D>("phiSta_ZMuMuTagged","phi of Sta (global-global)",50, -3.1415, 3.1415);
  h_ptSta_ZMuMuTagged = fs->make<TH1D>("ptSta_ZMuMuTagged","pt of Sta (global-global)",100, 0., 100.);
  h_DRSta_ZMuMuTagged = fs->make<TH1D>("DRTrackSta_ZMuMuTagged_staSelected","DR track-sta sta selected (global-global)",100, 0., 5.);

  // histograms for cynematic of ZMuMutagged muons when Track has wrong charge
  h_zMuStaMass_wrongTrkCharge_ZMuMuTagged = fs->make<TH1D>("zMuStaMass_wrongTrkCharge_ZMuMuTagged","inv Mass ZMuSta cand (global-global wrongTrkCharge)",100, 0., 200.);
  h_etaSta_wrongTrkCharge_ZMuMuTagged = fs->make<TH1D>("etaSta_wrongTrkCharge_ZMuMuTagged","eta of Sta (global-global wrongTrkCharge)",50, -2.5, 2.5);
  h_phiSta_wrongTrkCharge_ZMuMuTagged = fs->make<TH1D>("phiSta_wrongTrkCharge_ZMuMuTagged","phi of Sta (global-global wrongTrkCharge)",50, -3.1415, 3.1415);
  h_ptSta_wrongTrkCharge_ZMuMuTagged = fs->make<TH1D>("ptSta_wrongTrkCharge_ZMuMuTagged","pt of Sta (global-global wrongTrkCharge)",100, 0., 100.);
  h_DRSta_wrongTrkCharge_ZMuMuTagged = fs->make<TH1D>("DRTrackSta_wrongTrkCharge_ZMuMuTagged","DR track-sta (global-global wrongTrkCharge)",100, 0., 5.);

  //
  // ****************************************************************************************************
  // histograms for cynematic of ZMuTracktagged muons with unMatched StandAlone
  h_zMuTrackMass_ZMuTrackTagged = fs->make<TH1D>("zMuTrackMass_ZMuTrackTagged","inv Mass ZMuTrack cand (global-track)",100, 0., 200.);
  h_etaTrack_ZMuTrackTagged = fs->make<TH1D>("etaTrack_ZMuTrackTagged","eta of Track (global-track)",50, -2.5, 2.5);
  h_phiTrack_ZMuTrackTagged = fs->make<TH1D>("phiTrack_ZMuTrackTagged","phi of Track (global-track)",50, -3.1415, 3.1415);
  h_ptTrack_ZMuTrackTagged = fs->make<TH1D>("ptTrack_ZMuTrackTagged","pt of Track (global-track)",100, 0., 100.);
  h_DRTrack_ZMuTrackTagged = fs->make<TH1D>("DRTrackSta_ZMuTrackTagged","DR track-sta (global-track)",100, 0., 5.);

  // histograms for cynematic of ZMuTracktagged muons when unMatched StandAlone has wrong charge
  h_zMuTrackMass_wrongStaCharge_ZMuTrackTagged = fs->make<TH1D>("zMuTrackMass_wrongStaCharge_ZMuTrackTagged","inv Mass ZMuTrack cand (global-track wrongUnMatcehdStaCharge)",100, 0., 200.);
  h_etaTrack_wrongStaCharge_ZMuTrackTagged = fs->make<TH1D>("etaTrack_wrongStaCharge_ZMuTrackTagged","eta of Track (global-track wrongUnMatchedStaCharge)",50, -2.5, 2.5);
  h_phiTrack_wrongStaCharge_ZMuTrackTagged = fs->make<TH1D>("phiTrack_wrongStaCharge_ZMuTrackTagged","phi of Track (global-track wrongUnMatchedStaCharge)",50, -3.1415, 3.1415);
  h_ptTrack_wrongStaCharge_ZMuTrackTagged = fs->make<TH1D>("ptTrack_wrongStaCharge_ZMuTrackTagged","pt of Track (global-track wrongUnMatchedStaCharge)",100, 0., 100.);
  h_DRTrack_wrongStaCharge_ZMuTrackTagged = fs->make<TH1D>("DRTrackSta_wrongStaCharge_ZMuTrackTagged","DR track-sta (global-track wrongUnMatchedStaCharge)",100, 0., 5.);

  // histograms for cynematic of ZMuStatagged muons with unMatched Track
  h_zMuStaMass_ZMuStaTagged = fs->make<TH1D>("zMuStaMass_ZMuStaTagged","inv Mass ZMuSta cand (global-sta)",100, 0., 200.);
  h_etaSta_ZMuStaTagged = fs->make<TH1D>("etaSta_ZMuStaTagged","eta of Sta (global-sta)",50, -2.5, 2.5);
  h_phiSta_ZMuStaTagged = fs->make<TH1D>("phiSta_ZMuStaTagged","phi of Sta (global-sta)",50, -3.1415, 3.1415);
  h_ptSta_ZMuStaTagged = fs->make<TH1D>("ptSta_ZMuStaTagged","pt of Sta (global-sta)",100, 0., 100.);

  // histograms for cynematic of ZMuStatagged muons when unMatched track has wrong charge
  h_zMuStaMass_wrongTrkCharge_ZMuStaTagged = fs->make<TH1D>("zMuStaMass_wrongTrkCharge_ZMuStaTagged","inv Mass ZMuSta cand (global-sta wrongUnMatcehdTrkCharge)",100, 0., 200.);
  h_etaSta_wrongTrkCharge_ZMuStaTagged = fs->make<TH1D>("etaSta_wrongTrkCharge_ZMuStaTagged","eta of Sta (global-sta wrongUnMatchedTrkCharge)",50, -2.5, 2.5);
  h_phiSta_wrongTrkCharge_ZMuStaTagged = fs->make<TH1D>("phiSta_wrongTrkCharge_ZMuStaTagged","phi of Sta (global-sta wrongUnMatchedTrkCharge)",50, -3.1415, 3.1415);
  h_ptSta_wrongTrkCharge_ZMuStaTagged = fs->make<TH1D>("ptSta_wrongTrkCharge_ZMuStaTagged","pt of Sta (global-sta wrongUnMatchedTrkCharge)",100, 0., 100.);

  // clear global counters
  totalNumberOfZfound=0;
  totalNumberOfZpassed=0;
  nZMuMuSameSign_mcMatched =  0;
  nZMuMuSameSign = 0;
  n_goodTrack_ZMuMutagged = 0;
  n_correctStaCharge_ZMuMutagged = 0;
  n_wrongStaCharge_ZMuMutagged = 0;
  n_goodSta_ZMuMutagged = 0;
  n_correctTrkCharge_ZMuMutagged = 0;
  n_wrongTrkCharge_ZMuMutagged = 0;
  n_goodTrack_ZMuTracktagged = 0;
  n_correctStaCharge_ZMuTracktagged = 0;
  n_wrongStaCharge_ZMuTracktagged = 0;
  n_StaNotFound_ZMuTracktagged=0;

  n_goodSta_ZMuStatagged = 0;
  n_correctTrkCharge_ZMuStatagged = 0;
  n_wrongTrkCharge_ZMuStatagged = 0;
  n_TrkNotFound_ZMuStatagged=0;

  n_OneGoodZMuTrack=0;
  n_MultipleGoodZMuTrack=0;
  numberOfMatchedZMuSta_=0;
  n_ZMuStaTaggedMatched=0;
}

void ZMuMuPerformances::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateView> zMuMu;
  Handle<GenParticleMatch> zMuMuMatchMap; //Map of Z made by Mu global + Mu global (can be used also for same sign Zmumu)
  Handle<CandidateView> zMuTrack;
  Handle<GenParticleMatch> zMuTrackMatchMap; //Map of Z made by Mu + Track
  Handle<CandidateView> zMuStandAlone;
  Handle<GenParticleMatch> zMuStandAloneMatchMap; //Map of Z made by Mu + StandAlone
  Handle<CandidateView> muons; //Collection of Muons
  Handle<GenParticleMatch> muonMatchMap;
  Handle<IsolationCollection> muonIso;
  Handle<CandidateView> tracks; //Collection of Tracks
  Handle<IsolationCollection> trackIso;
  Handle<GenParticleCollection> genParticles;  // Collection of Generatd Particles

  event.getByToken(zMuMuToken_, zMuMu);
  event.getByToken(zMuTrackToken_, zMuTrack);
  event.getByToken(zMuStandAloneToken_, zMuStandAlone);
  event.getByToken(muonsToken_, muons);
  event.getByToken(tracksToken_, tracks);
  event.getByToken(genParticlesToken_, genParticles);

  /*
  cout << "*********  zMuMu         size : " << zMuMu->size() << endl;
  cout << "*********  zMuMuSameSign size : " << zMuMuSameSign->size() << endl;
  cout << "*********  zMuStandAlone size : " << zMuStandAlone->size() << endl;
  cout << "*********  zMuTrack      size : " << zMuTrack->size() << endl;
  cout << "*********  muons         size : " << muons->size()<< endl;
  cout << "*********  standAlone    size : " << standAlone->size()<< endl;
  cout << "*********  tracks        size : " << tracks->size()<< endl;
  cout << "*********  generated     size : " << genParticles->size()<< endl;
  cout << "***************************************************" << endl;
  */

  int n_globalMuon_perEvent=0;
  int n_staOnlyMuon_perEvent=0;
  int n_trackerOnlyMuon_perEvent=0;
  int n_trackerStaOnlyMuon_perEvent=0;
  int n_globalMuon_perEvent_MCmatch=0;
  int n_staOnlyMuon_perEvent_MCmatch=0;
  int n_trackerOnlyMuon_perEvent_MCmatch=0;
  int n_trackerStaOnlyMuon_perEvent_MCmatch=0;

  for(unsigned int j = 0; j < muons->size() ; ++j) {
    CandidateBaseRef muCandRef = muons->refAt(j);
    const Candidate & muCand = (*muons)[j]; //the candidate
    const reco::Muon & muon = dynamic_cast<const reco::Muon &>(muCand);
    reco::TrackRef innerTrackRef = muon.track();
    reco::TrackRef outerTrackRef = muon.standAloneMuon();
    TrackRef muStaComponentRef = muCand.get<TrackRef,reco::StandAloneMuonTag>();  // standalone part of muon
    TrackRef muTrkComponentRef = muCand.get<TrackRef>();  // track part of muon
    GenParticleRef muonMatch = (*muonMatchMap)[muCandRef];
    if (muCandRef->isGlobalMuon()==1) n_globalMuon_perEvent++;
    if (muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==0 && muCandRef->isStandAloneMuon()==1) n_staOnlyMuon_perEvent++;
    if (muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==1 && muCandRef->isStandAloneMuon()==0) n_trackerOnlyMuon_perEvent++;
    if (muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==1 && muCandRef->isStandAloneMuon()==1) n_trackerStaOnlyMuon_perEvent++;

    if (muonMatch.isNonnull()) {
      if (muCandRef->isGlobalMuon()==1) n_globalMuon_perEvent_MCmatch++;
      if (muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==0 && muCandRef->isStandAloneMuon()==1) n_staOnlyMuon_perEvent_MCmatch++;
      if (muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==1 && muCandRef->isStandAloneMuon()==0) n_trackerOnlyMuon_perEvent_MCmatch++;
      if (muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==1 && muCandRef->isStandAloneMuon()==1) n_trackerStaOnlyMuon_perEvent_MCmatch++;
      double productCharge = muCandRef->charge() * muonMatch->charge();
      if (muCandRef->isGlobalMuon()==1) {
	h_GlobalMuonChargeTimeGenCharge->Fill(productCharge);
	h_GlobalMuonEtaMinusGenEta->Fill(muCandRef->eta() - muonMatch->eta());
	h_GlobalMuonPtMinusGenPt->Fill((muCandRef->pt() - muonMatch->pt())/muonMatch->pt());
	h_GlobalMuonStaComponentEtaMinusGenEta->Fill(muStaComponentRef->eta() - muonMatch->eta());
	h_GlobalMuonStaComponentPtMinusGenPt->Fill((muStaComponentRef->pt() - muonMatch->pt())/muonMatch->pt());
	h_DEtaGlobalGenvsEtaGen->Fill(muonMatch->eta(),muCandRef->eta() - muonMatch->eta());
	h_DPtGlobalGenvsPtGen->Fill(muonMatch->pt(),(muCandRef->pt() - muonMatch->pt())/muonMatch->pt());
	h_DEtaGlobalStaComponentGenvsEtaGen->Fill(muonMatch->eta(),muStaComponentRef->eta() - muonMatch->eta());
	h_DPtGlobalStaComponentGenvsPtGen->Fill(muonMatch->pt(),(muStaComponentRef->pt() - muonMatch->pt())/muonMatch->pt());
       }
      if (muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==1) {
	h_TrackerMuonChargeTimeGenCharge->Fill(productCharge);
	h_TrackerMuonEtaMinusGenEta->Fill(muCandRef->eta() - muonMatch->eta());
	h_TrackerMuonPtMinusGenPt->Fill((muCandRef->pt() - muonMatch->pt())/muonMatch->pt());
      }
    }
  }
  h_n_globalMuon_perEvent->Fill(n_globalMuon_perEvent);
  h_n_staOnlyMuon_perEvent->Fill(n_staOnlyMuon_perEvent);
  h_n_trackerOnlyMuon_perEvent->Fill(n_trackerOnlyMuon_perEvent);
  h_n_trackerStaOnlyMuon_perEvent->Fill(n_trackerStaOnlyMuon_perEvent);
  h_n_globalMuon_perEvent_MCmatch->Fill(n_globalMuon_perEvent_MCmatch);
  h_n_staOnlyMuon_perEvent_MCmatch->Fill(n_staOnlyMuon_perEvent_MCmatch);
  h_n_trackerOnlyMuon_perEvent_MCmatch->Fill(n_trackerOnlyMuon_perEvent_MCmatch);
  h_n_trackerStaOnlyMuon_perEvent_MCmatch->Fill(n_trackerStaOnlyMuon_perEvent_MCmatch);
  h_n_tracks_perEvent->Fill(tracks->size());

  h_n_zMuMu_perEvent->Fill(zMuMu->size());
  h_n_zMuSta_perEvent->Fill(zMuStandAlone->size());
  h_n_zMuTrack_perEvent->Fill(zMuTrack->size());

  //      std::cout<<"Run-> "<<event.id().run()<<std::endl;
  //      std::cout<<"Event-> "<<event.id().event()<<std::endl;


  // loop on ZMuMu
  if (zMuMu->size() > 0 ) {
    event.getByToken(zMuMuMatchMapToken_, zMuMuMatchMap);
    event.getByToken(muonIsoToken_, muonIso);
    event.getByToken(muonMatchMapToken_, muonMatchMap);
    float muGenplus_pt = 0, muGenminus_pt = 0, muGenplus_eta = 100, muGenminus_eta = 100;
    for(unsigned int i = 0; i < zMuMu->size(); ++i) { //loop on candidates
      const Candidate & zMuMuCand = (*zMuMu)[i]; //the candidate
      CandidateBaseRef zMuMuCandRef = zMuMu->refAt(i);
      GenParticleRef zMuMuMatch = (*zMuMuMatchMap)[zMuMuCandRef];
      bool isMCMatched = false;
      if(zMuMuMatch.isNonnull()) {
	isMCMatched = true;   // ZMuMu matched
	if(zMuMuMatch->pdgId() == 23 && zMuMuMatch->status()==3 && zMuMuMatch->numberOfDaughters() == 3) {
	                                     // Z0 decays in mu+ mu-, the 3rd daughter is the same Z0
	  const Candidate * dauGen0 = zMuMuMatch->daughter(0);
	  const Candidate * dauGen1 = zMuMuMatch->daughter(1);
	  const Candidate * dauGen2 = zMuMuMatch->daughter(2);
	  if (check_ifZmumu(dauGen0, dauGen1, dauGen2)) {         // Z0 in mu+ mu-
	    muGenplus_pt = getParticlePt(-13,dauGen0,dauGen1,dauGen2);
	    muGenminus_pt = getParticlePt(13,dauGen0,dauGen1,dauGen2);
	    muGenplus_eta = getParticleEta(-13,dauGen0,dauGen1,dauGen2);
	    muGenminus_eta = getParticleEta(13,dauGen0,dauGen1,dauGen2);
	    Particle::LorentzVector pZ(0, 0, 0, 0);
	    Particle::LorentzVector muplusp4 = getParticleP4(-13,dauGen0,dauGen1,dauGen2);
	    Particle::LorentzVector muminusp4 = getParticleP4(13,dauGen0,dauGen1,dauGen2);
	    pZ = muplusp4 + muminusp4;
	  }  // en if is Z
	}  // end if is Z->mumu

      }

      TrackRef as1 = zMuMuCand.daughter(0)->get<TrackRef,reco::StandAloneMuonTag>();  // standalone part of ZMuMu cand0
      TrackRef as2 = zMuMuCand.daughter(1)->get<TrackRef,reco::StandAloneMuonTag>();  // standalone part of ZMuMu cand1
      TrackRef a1 = zMuMuCand.daughter(0)->get<TrackRef,reco::CombinedMuonTag>();  // global part of ZMuMu cand0
      TrackRef a2 = zMuMuCand.daughter(1)->get<TrackRef,reco::CombinedMuonTag>();  // global part of ZMuMu cand1
      TrackRef at1 = zMuMuCand.daughter(0)->get<TrackRef>();  // tracker part of ZMuMu cand0
      TrackRef at2 = zMuMuCand.daughter(1)->get<TrackRef>();  // tracker part of ZMuMu cand1

      math::XYZTLorentzVector ps1(as1->px(),as1->py(),as1->pz(),as1->p());
      math::XYZTLorentzVector ps2(as2->px(),as2->py(),as2->pz(),as2->p());
      math::XYZTLorentzVector pg1(a1->px(),a1->py(),a1->pz(),a1->p());
      math::XYZTLorentzVector pg2(a2->px(),a2->py(),a2->pz(),a2->p());
      math::XYZTLorentzVector ptrk1(at1->px(),at1->py(),at1->pz(),at1->p());
      math::XYZTLorentzVector ptrk2(at2->px(),at2->py(),at2->pz(),at2->p());

      //      double mass2global = (pg1+pg2).mass();         // inv. Mass done with the two global muons (is th same like m)
      double massGlobalSta = (pg1+ps2).mass();       // inv. mass done with the global daughter(0) and the Sta part of Daughter(1)
      double massStaGlobal = (ps1+pg2).mass();       // inv. mass done with the global daughter(1) and the Sta part of Daughter(0)
      //      double mass2Tracker = (ptrk1+ptrk2).mass();    // inv. mass done with the two tracker compnents
      double massGlobalTracker = (pg1+ptrk2).mass(); // inv. mass done with the global daughter(0) and the tracker part of Daughter(1)
      double massTrackerGlobal = (ptrk1+pg2).mass(); // inv. mass done with the global daughter(1) and the tracker part of Daughter(0)
      double etaGlobal1 = a1->eta();
      double etaGlobal2 = a2->eta();
      double etaSta1 = as1->eta();
      double etaSta2 = as2->eta();
      double etaTracker1 = at1->eta();
      double etaTracker2 = at2->eta();
      //      double phiGlobal1 = a1->phi();
      //      double phiGlobal2 = a2->phi();
      double phiSta1 = as1->phi();
      double phiSta2 = as2->phi();
      double phiTracker1 = at1->phi();
      double phiTracker2 = at2->phi();
      double ptGlobal1 = a1->pt();
      double ptGlobal2 = a2->pt();
      double ptSta1 = as1->pt();
      double ptSta2 = as2->pt();
      double ptTracker1 = at1->pt();
      double ptTracker2 = at2->pt();
      double chargeGlobal1 = a1->charge();
      double chargeGlobal2 = a2->charge();
      double chargeSta1 = as1->charge();
      double chargeSta2 = as2->charge();
      double chargeTracker1 = at1->charge();
      double chargeTracker2 = at2->charge();
      double DR1 = deltaR(etaSta1, phiSta1, etaTracker1, phiTracker1);
      double DR2 = deltaR(etaSta2, phiSta2, etaTracker2, phiTracker2);

      if (chargeGlobal1 == chargeGlobal2) {
	nZMuMuSameSign++;
	h_zMuMuMassSameSign->Fill(zMuMuCand.mass());
	if (isMCMatched) {
	  nZMuMuSameSign_mcMatched++;
	  h_zMuMuMassSameSign_MCmatch->Fill(zMuMuCand.mass());

	}
      } else {
	h_zMuMuMassOppositeSign->Fill(zMuMuCand.mass());
      }

      bool etaCut = false;
      bool ptCut = false;
      //bool isoCut = false;
      bool massCut = false;

      // ******************************************************************************************************************************
      // Start study for StandAlone charge mis-id: select global-global events according to global1+track2 (or global2+track1)
      // *******************************************************************************************************************************

      // cynematical cuts for Zglobal1Track2
      if (abs(etaGlobal1)<etamax_ && abs(etaTracker2)<etamax_) etaCut = true;
      if (ptGlobal1>ptmin_ && ptTracker2>ptmin_) ptCut = true;
      if (massGlobalTracker>massMin_ && massGlobalTracker<massMax_) massCut = true;

      if (noCut_) {
	etaCut = true;
	ptCut = true;
	massCut = true;
      }

      if (etaCut && ptCut && massCut) {
	// check first global1-track2 if they have opposite charge and if global1 has consistent charge between sta and track
	if (chargeSta1 == chargeTracker1 && chargeTracker1 != chargeTracker2) {      // event tagged to study StandAlone2 charge
	  n_goodTrack_ZMuMutagged++;
	  h_zMuTrackMass_ZMuMuTagged->Fill(massGlobalTracker);  // inv mass global+tracker part
	  h_etaTrack_ZMuMuTagged->Fill(etaTracker2);      // eta of tagged track
	  h_phiTrack_ZMuMuTagged->Fill(phiTracker2);      // phi of tagged track
	  h_ptTrack_ZMuMuTagged->Fill(ptTracker2);        // pt of tagged track
	  h_DRTrack_ZMuMuTagged->Fill(DR2);               // DR between sta2 and tracker2 for tagged track

	  if (isMCMatched) {             // if MC match .. resolution plots of global1 respect to gen particles
	    double etaGen, ptGen;
	    if (chargeGlobal1==1) {
	      etaGen = muGenplus_eta;
	      ptGen = muGenplus_pt;
	    } else {
	      etaGen = muGenminus_eta;
	      ptGen = muGenminus_pt;
	    }
	    h_GlobalMuonEtaMinusGenEta_ZMuMuTagged->Fill(etaGlobal1 - etaGen);
	    h_GlobalMuonPtMinusGenPt_ZMuMuTagged->Fill((ptGlobal1 - ptGen)/ptGen);
	    h_GlobalMuonStaComponentEtaMinusGenEta_ZMuMuTagged->Fill(etaSta1 - etaGen);
	    h_GlobalMuonStaComponentPtMinusGenPt_ZMuMuTagged->Fill((ptSta1 - ptGen)/ptGen);
	    h_DEtaGlobalGenvsEtaGen_ZMuMuTagged->Fill(etaGen,etaGlobal1-etaGen);
	    h_DPtGlobalGenvsPtGen_ZMuMuTagged->Fill(ptGen,(ptGlobal1-ptGen)/ptGen);
	    h_DEtaGlobalStaComponentGenvsEtaGen_ZMuMuTagged->Fill(etaGen,etaSta1-etaGen);
	    h_DPtGlobalStaComponentGenvsPtGen_ZMuMuTagged->Fill(ptGen,(ptSta1-ptGen)/ptGen);
	    h_DPtGlobalGenvsEtaGen_ZMuMuTagged->Fill(etaGen,(ptGlobal1-ptGen)/ptGen);
	    h_DPtGlobalStaComponentGenvsEtaGen_ZMuMuTagged->Fill(etaGen,(ptSta1-ptGen)/ptGen);
	    h_DPtTrackGenvsPtGen_ZMuMuTagged->Fill(ptGen,(ptTracker1-ptGen)/ptGen);
	    h_DPtTrackGenvsEtaGen_ZMuMuTagged->Fill(etaGen,(ptTracker1-ptGen)/ptGen);

	  } // end if MC Match

	  if (chargeSta2 == chargeTracker2) {   // StandAlone2 has correct charge
	    n_correctStaCharge_ZMuMutagged++;
	    h_zMuStaMass_correctStaCharge_ZMuMuTagged->Fill(massGlobalSta);  // inv mass of global-Sta part for correct charge muons
	    h_ptStaMinusptTrack_correctStaCharge_ZMuMuTagged->Fill((ptSta2-ptTracker2)/ptTracker2);
	    h_ptStaMinusptTrack_vsEtaTracker_correctStaCharge_ZMuMuTagged->Fill(etaTracker2,(ptSta2-ptTracker2)/ptTracker2);
	    h_ptStaMinusptTrack_vsPtTracker_correctStaCharge_ZMuMuTagged->Fill(ptTracker2,(ptSta2-ptTracker2)/ptTracker2);
	    // qui posso aggiungere plot col MC match
	  }
	  if (chargeSta2 != chargeTracker2) {  // StandAlone2 has wrong charge
	    n_wrongStaCharge_ZMuMutagged++;
	    h_zMuTrackMass_wrongStaCharge_ZMuMuTagged->Fill(massGlobalTracker);  // inv mass global+tracker part (wrong Sta charge)
	    h_etaTrack_wrongStaCharge_ZMuMuTagged->Fill(etaTracker2);      // eta of tagged track (wrong Sta charge)
	    h_phiTrack_wrongStaCharge_ZMuMuTagged->Fill(phiTracker2);      // phi of tagged track (wrong Sta charge)
	    h_ptTrack_wrongStaCharge_ZMuMuTagged->Fill(ptTracker2);      // pt of tagged track (wrong Sta charge)
	    h_DRTrack_wrongStaCharge_ZMuMuTagged->Fill(DR2);            // DR between sta2 and tracker2 for tagged track (wrong Sta charge)
 	  }
	}  // end if check chrge global1-tracker2
      }  // end if cut selection

      // cynematical cuts for Zglobal2Track1
      etaCut = false;
      ptCut = false;
      massCut = false;
      //isoCut = false;
      if (abs(etaGlobal2)<etamax_ && abs(etaTracker1)<etamax_) etaCut = true;
      if (ptGlobal2>ptmin_ && ptTracker1>ptmin_) ptCut = true;
      if (massTrackerGlobal>massMin_ && massTrackerGlobal<massMax_) massCut = true;

      if (noCut_) {
	etaCut = true;
	ptCut = true;
	massCut = true;
      }

      if (etaCut && ptCut && massCut) {
	// check global2-track1 if they have opposite charge and if global2 has consistent charge between sta and track
	if (chargeSta2 == chargeTracker2 && chargeTracker1 != chargeTracker2) {      // event tagged to study StandAlone2 charge
	  n_goodTrack_ZMuMutagged++;
	  h_zMuTrackMass_ZMuMuTagged->Fill(massTrackerGlobal);  // inv mass global+tracker part
	  h_etaTrack_ZMuMuTagged->Fill(etaTracker1);      // eta of tagged track
	  h_phiTrack_ZMuMuTagged->Fill(phiTracker1);      // phi of tagged track
	  h_ptTrack_ZMuMuTagged->Fill(ptTracker1);        // pt of tagged track
	  h_DRTrack_ZMuMuTagged->Fill(DR1);               // DR between sta1 and tracker1 for tagged track

	  // qui posso aggiungere plot col MC match
	  if (isMCMatched) {             // if MC match .. resolution plots of global2 respect to gen particles
	    double etaGen, ptGen;
	    if (chargeGlobal2==1) {
	      etaGen = muGenplus_eta;
	      ptGen = muGenplus_pt;
	    } else {
	      etaGen = muGenminus_eta;
	      ptGen = muGenminus_pt;
	    }
	    h_GlobalMuonEtaMinusGenEta_ZMuMuTagged->Fill(etaGlobal2 - etaGen);
	    h_GlobalMuonPtMinusGenPt_ZMuMuTagged->Fill((ptGlobal2 - ptGen)/ptGen);
	    h_GlobalMuonStaComponentEtaMinusGenEta_ZMuMuTagged->Fill(etaSta2 - etaGen);
	    h_GlobalMuonStaComponentPtMinusGenPt_ZMuMuTagged->Fill((ptSta2 - ptGen)/ptGen);
	    h_DEtaGlobalGenvsEtaGen_ZMuMuTagged->Fill(etaGen,etaGlobal2-etaGen);
	    h_DPtGlobalGenvsPtGen_ZMuMuTagged->Fill(ptGen,(ptGlobal2-ptGen)/ptGen);
	    h_DEtaGlobalStaComponentGenvsEtaGen_ZMuMuTagged->Fill(etaGen,etaSta2-etaGen);
	    h_DPtGlobalStaComponentGenvsPtGen_ZMuMuTagged->Fill(ptGen,(ptSta2-ptGen)/ptGen);
	    h_DPtGlobalGenvsEtaGen_ZMuMuTagged->Fill(etaGen,(ptGlobal2-ptGen)/ptGen);
	    h_DPtGlobalStaComponentGenvsEtaGen_ZMuMuTagged->Fill(etaGen,(ptSta2-ptGen)/ptGen);
	    h_DPtTrackGenvsPtGen_ZMuMuTagged->Fill(ptGen,(ptTracker2-ptGen)/ptGen);
	    h_DPtTrackGenvsEtaGen_ZMuMuTagged->Fill(etaGen,(ptTracker2-ptGen)/ptGen);
	  } // end if MC Match


	  if (chargeSta1 == chargeTracker1) {   // StandAlone1 has correct charge
	    n_correctStaCharge_ZMuMutagged++;
	    h_zMuStaMass_correctStaCharge_ZMuMuTagged->Fill(massStaGlobal);  // inv mass of Sta-global part for correct charge muons
	    h_ptStaMinusptTrack_correctStaCharge_ZMuMuTagged->Fill((ptSta1-ptTracker1)/ptTracker1);
	    h_ptStaMinusptTrack_vsEtaTracker_correctStaCharge_ZMuMuTagged->Fill(etaTracker1,(ptSta1-ptTracker1)/ptTracker1);
	    h_ptStaMinusptTrack_vsPtTracker_correctStaCharge_ZMuMuTagged->Fill(ptTracker1,(ptSta1-ptTracker1)/ptTracker1);

	  }
	  if (chargeSta1 != chargeTracker1) {  // StandAlone2 has wrong charge
	    n_wrongStaCharge_ZMuMutagged++;
	    h_zMuTrackMass_wrongStaCharge_ZMuMuTagged->Fill(massTrackerGlobal);  // inv mass global+tracker part (wrong Sta charge)
	    h_etaTrack_wrongStaCharge_ZMuMuTagged->Fill(etaTracker1);      // eta of tagged track (wrong Sta charge)
	    h_phiTrack_wrongStaCharge_ZMuMuTagged->Fill(phiTracker1);      // phi of tagged track (wrong Sta charge)
	    h_ptTrack_wrongStaCharge_ZMuMuTagged->Fill(ptTracker1);      // pt of tagged track (wrong Sta charge)
	    h_DRTrack_wrongStaCharge_ZMuMuTagged->Fill(DR1);            // DR between sta1 and tracker1 for tagged track (wrong Sta charge)
 	  }
	}  // end if check charge global1-tracker2
      }  // end if cuts

      // ******************************************************************************************************************************
      // Start study for tracker charge mis-id: select global-global events according to global1+staComponent2 (or global2+staComponent1)
      // *******************************************************************************************************************************

      etaCut = false;
      ptCut = false;
      //isoCut = false;
      massCut = false;

     // cynematical cuts for Zglobal1Sta2
      if (abs(etaGlobal1)<etamax_ && abs(etaSta2)<etamax_) etaCut = true;
      if (ptGlobal1>ptmin_ && ptSta2>ptmin_) ptCut = true;
      if (massGlobalSta>massMin_ && massGlobalSta<massMax_) massCut = true;

      if (noCut_) {
	etaCut = true;
	ptCut = true;
	massCut = true;
      }

      if (etaCut && ptCut && massCut) {
	// check first global1-sta2 if they have opposite charge and if global1 has consistent charge between sta and track
	if (chargeSta1 == chargeTracker1 && chargeTracker1 != chargeSta2) {      // event tagged to study track2 charge
	  n_goodSta_ZMuMutagged++;
	  h_zMuStaMass_ZMuMuTagged->Fill(massGlobalSta);  // inv mass global+sta part
	  h_etaSta_ZMuMuTagged->Fill(etaSta2);            // eta of tagged sta
	  h_phiSta_ZMuMuTagged->Fill(phiSta2);      // phi of tagged sta
	  h_ptSta_ZMuMuTagged->Fill(ptSta2);        // pt of tagged sta
	  h_DRSta_ZMuMuTagged->Fill(DR2);               // DR between sta2 and tracker2 for tagged sta

	  if (chargeSta2 == chargeTracker2) {   // track2 has correct charge
	    n_correctTrkCharge_ZMuMutagged++;
	    // qui posso aggiungere plot col MC match
	  }
	  if (chargeSta2 != chargeTracker2) {  // track2 has wrong charge
	    n_wrongTrkCharge_ZMuMutagged++;
	    h_zMuStaMass_wrongTrkCharge_ZMuMuTagged->Fill(massGlobalSta);  // inv mass global+sta part (wrong Trk charge)
	    h_etaSta_wrongTrkCharge_ZMuMuTagged->Fill(etaSta2);      // eta of tagged sta (wrong trk charge)
	    h_phiSta_wrongTrkCharge_ZMuMuTagged->Fill(phiSta2);      // phi of tagged sta (wrong Trk charge)
	    h_ptSta_wrongTrkCharge_ZMuMuTagged->Fill(ptSta2);      // pt of tagged sta (wrong Trk charge)
	    h_DRSta_wrongTrkCharge_ZMuMuTagged->Fill(DR2);            // DR between sta2 and tracker2 for tagged sta (wrong trk charge)
 	  }
	}  // end if check chrge global1-sta2
      }  // end if cut selection

      etaCut = false;
      ptCut = false;
      //isoCut = false;
      massCut = false;

     // cynematical cuts for Zglobal2Sta1
      if (abs(etaGlobal2)<etamax_ && abs(etaSta1)<etamax_) etaCut = true;
      if (ptGlobal2>ptmin_ && ptSta1>ptmin_) ptCut = true;
      if (massStaGlobal>massMin_ && massStaGlobal<massMax_) massCut = true;

      if (noCut_) {
	etaCut = true;
	ptCut = true;
	massCut = true;
      }

      if (etaCut && ptCut && massCut) {
	// check first global2-sta1 if they have opposite charge and if global2 has consistent charge between sta and track
	if (chargeSta2 == chargeTracker2 && chargeTracker2 != chargeSta1) {      // event tagged to study track1 charge
	  n_goodSta_ZMuMutagged++;
	  h_zMuStaMass_ZMuMuTagged->Fill(massStaGlobal);  // inv mass global+sta part
	  h_etaSta_ZMuMuTagged->Fill(etaSta1);            // eta of tagged sta
	  h_phiSta_ZMuMuTagged->Fill(phiSta1);      // phi of tagged sta
	  h_ptSta_ZMuMuTagged->Fill(ptSta1);        // pt of tagged sta
	  h_DRSta_ZMuMuTagged->Fill(DR1);               // DR between sta1 and tracker1 for tagged sta

	  if (chargeSta1 == chargeTracker1) {   // track1 has correct charge
	    n_correctTrkCharge_ZMuMutagged++;
	    // qui posso aggiungere plot col MC match
	  }
	  if (chargeSta1 != chargeTracker1) {  // track1 has wrong charge
	    n_wrongTrkCharge_ZMuMutagged++;
	    h_zMuStaMass_wrongTrkCharge_ZMuMuTagged->Fill(massStaGlobal);  // inv mass global+sta part (wrong Trk charge)
	    h_etaSta_wrongTrkCharge_ZMuMuTagged->Fill(etaSta1);      // eta of tagged sta (wrong trk charge)
	    h_phiSta_wrongTrkCharge_ZMuMuTagged->Fill(phiSta1);      // phi of tagged sta (wrong Trk charge)
	    h_ptSta_wrongTrkCharge_ZMuMuTagged->Fill(ptSta1);      // pt of tagged sta (wrong Trk charge)
	    h_DRSta_wrongTrkCharge_ZMuMuTagged->Fill(DR1);            // DR between sta2 and tracker2 for tagged sta (wrong trk charge)
 	  }
	}  // end if check chrge global2-sta1
      }  // end if cut selection


    }  // end loop on ZMuMu cand
  }    // end if ZMuMu size > 0


  // loop on ZMuTrack in order to recover some unMatched StandAlone

  //double LargerDRCut=2.; // larger DR cut to recover unMatched Sta
  int taggedZ_index = -1; // index of Z with minimum DR respect to unMatched Sta
  int taggedMuon_index = -1; // index of Sta muon with minimum DR respect to unMatched track
  int n_ZMuTrackTagged_inEvent = 0;  // number of tagged Z in the event
  if (zMuTrack->size() > 0 && zMuMu->size()==0) {           // check ZMuTrack just if no ZMuMu has been found in the event
    event.getByToken(zMuTrackMatchMapToken_, zMuTrackMatchMap);
    for(unsigned int i = 0; i < zMuTrack->size(); ++i) { //loop on candidates
      const Candidate & zMuTrackCand = (*zMuTrack)[i]; //the candidate
      CandidateBaseRef zMuTrackCandRef = zMuTrack->refAt(i);
      GenParticleRef zMuTrackMatch = (*zMuTrackMatchMap)[zMuTrackCandRef];
      //bool isMCMatched = false;
      //if(zMuTrackMatch.isNonnull()) isMCMatched = true;   // ZMuTrack matched
      // forzo isMCMatched
      //      isMCMatched = true;

      double m = zMuTrackCand.mass();
      CandidateBaseRef zglobalDaughter = zMuTrackCand.daughter(0)->masterClone();
      CandidateBaseRef ztrackerDaughter = zMuTrackCand.daughter(1)->masterClone();
      TrackRef zglobalDaughter_StaComponentRef = zMuTrackCand.daughter(0)->get<TrackRef,reco::StandAloneMuonTag>();
                                                                                  // standalone part of global component of ZMuMu
      TrackRef zglobalDaughter_TrackComponentRef = zMuTrackCand.daughter(0)->get<TrackRef>();
                                                                                  // track part Of the global component of ZMuMu
      double ZtrackerDaughterCharge = ztrackerDaughter->charge();
      double ZtrackerDaughterPt = ztrackerDaughter->pt();
      double ZtrackerDaughterEta = ztrackerDaughter->eta();
      double ZtrackerDaughterPhi = ztrackerDaughter->phi();
      double ZglobalDaughterPt = zglobalDaughter->pt();
      double ZglobalDaughterEta = zglobalDaughter->eta();
      double ZglobalDaughter_StaComponentCharge = zglobalDaughter_StaComponentRef->charge();
      double ZglobalDaughter_TrackComponentCharge = zglobalDaughter_TrackComponentRef->charge();

      //*********************************************************************************************************************
      // study of standAlone charge mis-id and efficiency selecting ZMuTrack events (tag the index of Z and of muon)
      // for Sta charge mis-id just use unMatched standAlone muons trackerMuons that are standAlone Muons but no globalMuons
      // ********************************************************************************************************************
      // cynematical cuts for ZMuTrack
      bool etaCut = false;
      bool ptCut = false;
      //      bool isoCut = false;
      bool massCut = false;
      if (abs(ZglobalDaughterEta)<etamax_ && abs(ZtrackerDaughterEta)<etamax_) etaCut = true;
      if (ZglobalDaughterPt>ptmin_ && ZtrackerDaughterPt>ptmin_) ptCut = true;
      if (m>massMin_ && m<massMax_) massCut = true;

       if (noCut_) {
	etaCut = true;
	ptCut = true;
	massCut = true;
      }
      if (etaCut && ptCut && massCut && ZglobalDaughter_StaComponentCharge == ZglobalDaughter_TrackComponentCharge &&
	  ZglobalDaughter_TrackComponentCharge != ZtrackerDaughterCharge) {   // cynematic cuts and global charge consistent and opposite tracker charge
	n_ZMuTrackTagged_inEvent++;

	// posso inserire istogrammi eta e pt track per studio Sta efficiency
	// ...

	for(unsigned int j = 0; j < muons->size() ; ++j) {
	  CandidateBaseRef muCandRef = muons->refAt(j);
	  const Candidate & muCand = (*muons)[j]; //the candidate
	  TrackRef muStaComponentRef = muCand.get<TrackRef,reco::StandAloneMuonTag>();  // standalone part of muon
	  TrackRef muTrkComponentRef = muCand.get<TrackRef>();  // track part of muon

	  if (muCandRef->isStandAloneMuon()==1 && muCandRef->isGlobalMuon()==0 && muCandRef->isTrackerMuon()==1) {
	    double muEta = muCandRef->eta();
	    double muPhi = muCandRef->phi();
	    // check DeltaR between Sta muon and tracks of ZMuTrack
	    double DRmuSta_trackOfZ = deltaR(muEta, muPhi, ZtrackerDaughterEta, ZtrackerDaughterPhi);
	    if (DRmuSta_trackOfZ == 0) {  // match track track ... standalone-muTracker
	      taggedZ_index = i;
	      taggedMuon_index = j;
	    } // end check minimum DR
	  }  // end if isStandAlone
	}    // end loop on muon candidates
      } // end cynematic cuts

    }  // end loop on zMuTrack size
  }   // end if zMuTrack size > 0

  // analyze the tagged ZMuTRack and the Sta muons with minimal DR
  if (n_ZMuTrackTagged_inEvent>0) {   // at Least one ZMuTRack tagged

    if (taggedZ_index==-1) { // StandAlone inefficient
      n_StaNotFound_ZMuTracktagged++;
      //      h_etaTrack_StaNotFound_ZMuTrackTagged->Fill(ztrackerDaughter->eta());
    } else {
      const Candidate & zMuTrackCand = (*zMuTrack)[taggedZ_index]; //the candidate tagged
      CandidateBaseRef zMuTrackCandRef = zMuTrack->refAt(taggedZ_index);
      double m = zMuTrackCand.mass();
      CandidateBaseRef zglobalDaughter = zMuTrackCand.daughter(0)->masterClone();
      CandidateBaseRef ztrackerDaughter = zMuTrackCand.daughter(1)->masterClone();
      TrackRef zglobalDaughter_StaComponentRef = zMuTrackCand.daughter(0)->get<TrackRef,reco::StandAloneMuonTag>();
                                                                          // standalone part of global component of ZMuMu
      TrackRef zglobalDaughter_TrackComponentRef = zMuTrackCand.daughter(0)->get<TrackRef>();
                                                                          // track part Of the global component of ZMuMu
      double ZtrackerDaughterCharge = ztrackerDaughter->charge();
      double ZtrackerDaughterPt = ztrackerDaughter->pt();
      double ZtrackerDaughterEta = ztrackerDaughter->eta();
      double ZtrackerDaughterPhi = ztrackerDaughter->phi();

      CandidateBaseRef muCandRef = muons->refAt(taggedMuon_index);  // the tagged muon
      const Candidate & muCand = (*muons)[taggedMuon_index]; //the candidate
      TrackRef muStaComponentRef = muCand.get<TrackRef,reco::StandAloneMuonTag>();  // standalone part of muon
      TrackRef muTrkComponentRef = muCand.get<TrackRef>();  // track part of muon

      double muEta = muStaComponentRef->eta();
      double muPhi = muStaComponentRef->phi();
      double muCharge = muStaComponentRef->charge();
      // check DeltaR between Sta muon and tracks of ZMuTrack
      double DRmuSta_trackOfZ = deltaR(muEta, muPhi, ZtrackerDaughterEta, ZtrackerDaughterPhi);

      n_goodTrack_ZMuTracktagged++;
      h_zMuTrackMass_ZMuTrackTagged->Fill(m);         // inv mass ZMuTrack for tagged events
      h_etaTrack_ZMuTrackTagged->Fill(ZtrackerDaughterEta);   // eta of tagged track
      h_phiTrack_ZMuTrackTagged->Fill(ZtrackerDaughterPhi);      // phi of tagged track
      h_ptTrack_ZMuTrackTagged->Fill(ZtrackerDaughterPt);        // pt of tagged track
      h_DRTrack_ZMuTrackTagged->Fill(DRmuSta_trackOfZ);          // DR between sta1 and tracker1 for tagged track

      // check StandAlone charge
      if (muCharge != ZtrackerDaughterCharge) {  // wrong Sta charge
	n_wrongStaCharge_ZMuTracktagged++;                              // number of events wrong charge for unMatched Sta
	h_zMuTrackMass_wrongStaCharge_ZMuTrackTagged->Fill(m);         // inv mass ZMuTrack for tagged events wrong unMatched Sta charge
	h_etaTrack_wrongStaCharge_ZMuTrackTagged->Fill(ZtrackerDaughterEta);   // eta of tagged track wrong unMatched Sta charge
	h_phiTrack_wrongStaCharge_ZMuTrackTagged->Fill(ZtrackerDaughterPhi);      // phi of tagged track wrong unMatched Sta charge
	h_ptTrack_wrongStaCharge_ZMuTrackTagged->Fill(ZtrackerDaughterPt);        // pt of tagged track wrong unMatched Sta charge
	h_DRTrack_wrongStaCharge_ZMuTrackTagged->Fill(DRmuSta_trackOfZ);          // DR between unMatched Sta and tracker for wrong sta charge
      } else {  // correct Sta charge
	n_correctStaCharge_ZMuTracktagged++;                              // number of events correct charge for unMatched Sta
      }  // end if Sta charge check
    }   // end if StandAlone is present
  }  // end if zMuTrack tagged

  //*********************************************************************************************************************
  // study of track charge mis-id and efficiency selecting ZMuSta events
  // for Track charge mis-id just use unMatched standAlone muons trackerMuons that are standAlone Muons but no globalMuons
  // ********************************************************************************************************************

  // loop on ZMuSta in order to recover some unMatched StandAlone
  bool isZMuStaMatched=false;
  //LargerDRCut=2.; // larger DR cut to recover unMatched Sta
  taggedZ_index = -1; // index of Z with minimum DR respect to unMatched Sta
  taggedMuon_index = -1; // index of Sta muon with minimum DR respect to unMatched track
  int n_ZMuStaTagged_inEvent = 0;  // number of tagged Z in the event
  if (zMuStandAlone->size() > 0) {           // check ZMuSta just if no ZMuMu has been found in the event
    event.getByToken(zMuStandAloneMatchMapToken_, zMuStandAloneMatchMap);
    for(unsigned int i = 0; i < zMuStandAlone->size(); ++i) { //loop on candidates
      const Candidate & zMuStaCand = (*zMuStandAlone)[i]; //the candidate
      CandidateBaseRef zMuStaCandRef = zMuStandAlone->refAt(i);

      GenParticleRef zMuStaMatch = (*zMuStandAloneMatchMap)[zMuStaCandRef];
      if(zMuStaMatch.isNonnull()) {        // ZMuSta Macthed
	numberOfMatchedZMuSta_++;
	isZMuStaMatched = true;
      }

      double m = zMuStaCand.mass();
      CandidateBaseRef zglobalDaughter = zMuStaCand.daughter(0)->masterClone();
      CandidateBaseRef zstandaloneDaughter = zMuStaCand.daughter(1)->masterClone();
      int iglb = 0;
      int ista = 1;
      if (zglobalDaughter->isGlobalMuon()==0 && zstandaloneDaughter->isGlobalMuon()==1) {  // invert definition
	CandidateBaseRef buffer = zglobalDaughter;
	zglobalDaughter = zstandaloneDaughter;
	zstandaloneDaughter = buffer;
	iglb = 1;
	ista = 0;
      }
      TrackRef zglobalDaughter_StaComponentRef = zMuStaCand.daughter(iglb)->get<TrackRef,reco::StandAloneMuonTag>();
                                                                                  // standalone part of global component of ZMuMu
      TrackRef zglobalDaughter_TrackComponentRef = zMuStaCand.daughter(iglb)->get<TrackRef>();
                                                                                  // track part Of the global component of ZMuMu
      TrackRef zstaDaughter_StaComponentRef = zMuStaCand.daughter(ista)->get<TrackRef,reco::StandAloneMuonTag>();
                                                                                  // standalone part of global component of ZMuMu
      TrackRef zstaDaughter_TrackComponentRef = zMuStaCand.daughter(ista)->get<TrackRef>();
                                                                                  // track part Of the global component of ZMuMu
      double ZglobalDaughterPt = zglobalDaughter->pt();
      double ZglobalDaughterEta = zglobalDaughter->eta();

      double ZstaDaughter_StaComponentCharge = zstaDaughter_StaComponentRef->charge();
      double ZstaDaughter_StaComponentPt = zstaDaughter_StaComponentRef->pt();
      double ZstaDaughter_StaComponentEta = zstaDaughter_StaComponentRef->eta();
      double ZstaDaughter_StaComponentPhi = zstaDaughter_StaComponentRef->phi();
      double ZstaDaughter_TrackComponentCharge = zstaDaughter_TrackComponentRef->charge();

      double ZglobalDaughter_StaComponentCharge = zglobalDaughter_StaComponentRef->charge();
      double ZglobalDaughter_TrackComponentCharge = zglobalDaughter_TrackComponentRef->charge();

      // cynematical cuts for ZMuSta
      bool etaCut = false;
      bool ptCut = false;
      //      bool isoCut = false;
      bool massCut = false;
      if (abs(ZglobalDaughterEta)<etamax_ && abs(ZstaDaughter_StaComponentEta)<etamax_) etaCut = true;
      if (ZglobalDaughterPt>ptmin_ && ZstaDaughter_StaComponentPt>ptmin_) ptCut = true;
      if (m>massMin_ && m<massMax_) massCut = true;          // dovrei usare la massa fatta con la sola parte sta
                                                             // (Se è anche trackerMu non è cosi')
       if (noCut_) {
	etaCut = true;
	ptCut = true;
	massCut = true;
      }
      if (etaCut && ptCut && massCut && ZglobalDaughter_StaComponentCharge == ZglobalDaughter_TrackComponentCharge &&
	  ZglobalDaughter_StaComponentCharge != ZstaDaughter_StaComponentCharge) {   // cynematic cuts and global charge consistent and opposite sta charge
	n_ZMuStaTagged_inEvent++;
	if (isZMuStaMatched) n_ZMuStaTaggedMatched++;
	// posso inserire istogrammi eta e pt track per studio Sta efficiency
	// ...
	if (zstandaloneDaughter->isStandAloneMuon()==1 && zstandaloneDaughter->isTrackerMuon()==1) {  // track matched
	  n_goodSta_ZMuStatagged++;
	  h_zMuStaMass_ZMuStaTagged->Fill(m);         // inv mass ZMuSta for tagged events
	  h_etaSta_ZMuStaTagged->Fill(ZstaDaughter_StaComponentEta);   // eta of tagged sta
	  h_phiSta_ZMuStaTagged->Fill(ZstaDaughter_StaComponentPhi);  // phi of tagged sta
	  h_ptSta_ZMuStaTagged->Fill(ZstaDaughter_StaComponentPt);        // pt of tagged sta

	  // check Track charge
	  if (ZstaDaughter_StaComponentCharge != ZstaDaughter_TrackComponentCharge) {  // wrong Trk charge
	    n_wrongTrkCharge_ZMuStatagged++;                              // number of events wrong track charge for unMatched track
	    h_zMuStaMass_wrongTrkCharge_ZMuStaTagged->Fill(m);         // inv mass ZMuSta for tagged evts wrong unMatched track charge
	    h_etaSta_wrongTrkCharge_ZMuStaTagged->Fill(ZstaDaughter_StaComponentEta);   // eta of tagged sta wrong unMatched track charge
	    h_phiSta_wrongTrkCharge_ZMuStaTagged->Fill(ZstaDaughter_StaComponentPhi);   // phi of tagged sta wrong unMatched track charge
	    h_ptSta_wrongTrkCharge_ZMuStaTagged->Fill(ZstaDaughter_StaComponentPt);        // pt of tagged sta wrong unMatched track charge
	  } else {  // correct Sta charge
	    n_correctTrkCharge_ZMuStatagged++;                              // number of events correct charge for unMatched Sta
	  }  // end if Sta charge check

	} else {   // tracker inefficient
	  n_TrkNotFound_ZMuStatagged++;
	}
      } // end cynematic cuts
      if (n_ZMuStaTagged_inEvent==0) {
      }


    }  // end loop on zMuSta candidates
  }   // end check ZMuSta size

}       // end analyze

bool ZMuMuPerformances::check_ifZmumu(const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMuPerformances::getParticlePt(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMuPerformances::getParticleEta(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

float ZMuMuPerformances::getParticlePhi(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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

Particle::LorentzVector ZMuMuPerformances::getParticleP4(const int ipart, const Candidate * dauGen0, const Candidate * dauGen1, const Candidate * dauGen2)
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



void ZMuMuPerformances::endJob() {

  cout << "------------------------------------  Counters  --------------------------------" << endl;
  cout << "totalNumberOfZfound = " << totalNumberOfZfound << endl;
  cout << "totalNumberOfZpassed = " << totalNumberOfZpassed << endl;
  cout << "Number Of ZMuMu Same Sign (no cuts) " << nZMuMuSameSign << endl;
  cout << "Number Of ZMuMu Same Sign (no cuts) MC matched " << nZMuMuSameSign_mcMatched << endl;

  cout << "------------------------------------  Counters for standAlone charge mis-id studies --------------------------------" << endl;
  cout << " number of goodTracks tagged for ZMuMu collection = " << n_goodTrack_ZMuMutagged << endl;
  cout << " number of goodTracks tagged for ZMuMu collection (correct Sta charge) = " << n_correctStaCharge_ZMuMutagged << endl;
  cout << " number of goodTracks tagged for ZMuMu collection (wrong Sta charge) = " << n_wrongStaCharge_ZMuMutagged << endl<<endl;
  cout << " number of goodTracks tagged for ZMuTrack collection unMatchedSTA = " << n_goodTrack_ZMuTracktagged << endl;
  cout << " number of goodTracks tagged for ZMuTrack collection unMatchedSTA (correct Sta charge) = " << n_correctStaCharge_ZMuTracktagged << endl;
  cout << " number of goodTracks tagged for ZMuTrack collection unMatchedSTA (wrong Sta charge) = " << n_wrongStaCharge_ZMuTracktagged << endl<<endl;
  cout << " number of goodTracks tagged for ZMuTrack collection (No STA found) = " << n_StaNotFound_ZMuTracktagged << endl;

  cout << "------------------------------------  Counters for Track charge mis-id studies --------------------------------" << endl;
  cout << " number of goodStandAlone tagged for ZMuMu collection = " << n_goodSta_ZMuMutagged << endl;
  cout << " number of goodStandAlone tagged for ZMuMu collection (correct Trk charge) = " << n_correctTrkCharge_ZMuMutagged << endl;
  cout << " number of goodStandAlone tagged for ZMuMu collection (wrong Trk charge) = " << n_wrongTrkCharge_ZMuMutagged << endl<<endl;
  cout << " number of goodSta tagged for ZMuSta collection unMatchedTrk = " << n_goodSta_ZMuStatagged << endl;
  cout << " number of goodSta tagged for ZMuSta collection unMatchedTrk (correct Trk charge) = " << n_correctTrkCharge_ZMuStatagged << endl;
  cout << " number of goodSta tagged for ZMuSta collection unMatchedTrk (wrong Trk charge) = " << n_wrongTrkCharge_ZMuStatagged << endl<<endl;
  cout << " number of goodSta tagged for ZMuSta collection (No Trk found) = " << n_TrkNotFound_ZMuStatagged << endl;
  cout << " number of ZMuSta mactched = " << numberOfMatchedZMuSta_ << endl;
  cout << " number of ZMuSta Tagged matched = " << n_ZMuStaTaggedMatched << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuPerformances);

