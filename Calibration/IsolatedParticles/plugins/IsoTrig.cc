// -*- C++ -*-//
// Package:    IsoTrig
// Class:      IsoTrig
//
/**\class IsoTrig IsoTrig.cc IsoTrig/IsoTrig/src/IsoTrig.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ruchi Gupta
//         Created:  Fri May 25 12:02:48 CDT 2012
// $Id$
//
//

// system include files
#include <memory>

// Root objects
#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TMath.h"

// user include files
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
//Tracks
#include "DataFormats/TrackReco/interface/HitPattern.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

class IsoTrig : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit IsoTrig(const edm::ParameterSet &);
  ~IsoTrig() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override;
  void endJob() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override {}

  void clearMipCutTreeVectors();
  void clearChgIsolnTreeVectors();
  void pushChgIsolnTreeVecs(math::XYZTLorentzVector &Pixcand,
                            math::XYZTLorentzVector &Trkcand,
                            std::vector<double> &PixMaxP,
                            double &TrkMaxP,
                            bool &selTk);
  void pushMipCutTreeVecs(math::XYZTLorentzVector &NFcand,
                          math::XYZTLorentzVector &Trkcand,
                          double &EmipNFcand,
                          double &EmipTrkcand,
                          double &mindR,
                          double &mindP1,
                          std::vector<bool> &Flags,
                          double hCone);
  void StudyTrkEbyP(edm::Handle<reco::TrackCollection> &trkCollection);
  void studyTiming(const edm::Event &theEvent);
  void studyMipCut(edm::Handle<reco::TrackCollection> &trkCollection,
                   edm::Handle<reco::IsolatedPixelTrackCandidateCollection> &L2cands);
  void studyTrigger(edm::Handle<reco::TrackCollection> &, std::vector<reco::TrackCollection::const_iterator> &);
  void studyIsolation(edm::Handle<reco::TrackCollection> &, std::vector<reco::TrackCollection::const_iterator> &);
  void chgIsolation(double &etaTriggered,
                    double &phiTriggered,
                    edm::Handle<reco::TrackCollection> &trkCollection,
                    const edm::Event &theEvent);
  void getGoodTracks(const edm::Event &, edm::Handle<reco::TrackCollection> &);
  void fillHist(int, math::XYZTLorentzVector &);
  void fillDifferences(int, math::XYZTLorentzVector &, math::XYZTLorentzVector &, bool);
  void fillCuts(int, double, double, double, math::XYZTLorentzVector &, int, bool);
  void fillEnergy(int, int, double, double, math::XYZTLorentzVector &);
  double dEta(math::XYZTLorentzVector &, math::XYZTLorentzVector &);
  double dPhi(math::XYZTLorentzVector &, math::XYZTLorentzVector &);
  double dR(math::XYZTLorentzVector &, math::XYZTLorentzVector &);
  double dPt(math::XYZTLorentzVector &, math::XYZTLorentzVector &);
  double dP(math::XYZTLorentzVector &, math::XYZTLorentzVector &);
  double dinvPt(math::XYZTLorentzVector &, math::XYZTLorentzVector &);
  std::pair<double, double> etaPhiTrigger();
  std::pair<double, double> GetEtaPhiAtEcal(double etaIP, double phiIP, double pT, int charge, double vtxZ);
  double getDistInCM(double eta1, double phi1, double eta2, double phi2);

  // ----------member data ---------------------------
  HLTPrescaleProvider hltPrescaleProvider_;
  const std::vector<std::string> trigNames_;
  const edm::InputTag pixCandTag_, l1CandTag_, l2CandTag_;
  const std::vector<edm::InputTag> pixelTracksSources_;
  const bool doL2L3_, doTiming_, doMipCutTree_;
  const bool doTrkResTree_, doChgIsolTree_, doStudyIsol_;
  const int verbosity_;
  const std::vector<double> pixelIsolationConeSizeAtEC_;
  const double minPTrackValue_, vtxCutSeed_, vtxCutIsol_;
  const double tauUnbiasCone_, prelimCone_;
  std::string theTrackQuality_;
  const std::string processName_;
  double rEB_, zEE_, bfVal_;
  spr::trackSelectionParameters selectionParameters_;
  const double dr_L1_, a_coneR_, a_charIsoR_, a_neutIsoR_;
  const double a_mipR_, a_neutR1_, a_neutR2_, cutMip_;
  const double cutCharge_, cutNeutral_;
  const int minRunNo_, maxRunNo_;
  edm::EDGetTokenT<LumiDetails> tok_lumi_;
  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_hlt_;
  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<reco::VertexCollection> tok_verthb_, tok_verthe_;
  edm::EDGetTokenT<SeedingLayerSetsHits> tok_SeedingLayerHB_;
  edm::EDGetTokenT<SeedingLayerSetsHits> tok_SeedingLayerHE_;
  edm::EDGetTokenT<SiPixelRecHitCollection> tok_SiPixelRecHits_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> tok_pixtk_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_l1cand_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> tok_l2cand_;
  std::vector<edm::EDGetTokenT<reco::TrackCollection>> tok_pixtks_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;

  std::vector<reco::TrackRef> pixelTrackRefsHB_, pixelTrackRefsHE_;
  edm::Handle<HBHERecHitCollection> hbhe_;
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle_;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle_;
  edm::Handle<reco::BeamSpot> beamSpotH_;
  edm::Handle<reco::VertexCollection> recVtxs_;

  const MagneticField *bField_;
  const CaloGeometry *geo_;
  math::XYZPoint leadPV_;

  std::map<unsigned int, unsigned int> trigList_;
  std::map<unsigned int, const std::pair<int, int>> trigPreList_;
  bool changed_;
  double pLimits_[6];
  edm::Service<TFileService> fs_;
  TTree *MipCutTree_, *ChgIsolnTree_, *TrkResTree_, *TimingTree_;
  std::vector<double> *t_timeL2Prod;
  std::vector<int> *t_nPixCand;
  std::vector<int> *t_nPixSeed;
  std::vector<int> *t_nGoodTk;

  std::vector<double> *t_TrkhCone;
  std::vector<double> *t_TrkP;
  std::vector<bool> *t_TrkselTkFlag;
  std::vector<bool> *t_TrkqltyFlag;
  std::vector<bool> *t_TrkMissFlag;
  std::vector<bool> *t_TrkPVFlag;
  std::vector<bool> *t_TrkNuIsolFlag;

  std::vector<double> *t_PixcandP;
  std::vector<double> *t_PixcandPt;
  std::vector<double> *t_PixcandEta;
  std::vector<double> *t_PixcandPhi;
  std::vector<std::vector<double>> *t_PixcandMaxP;
  std::vector<double> *t_PixTrkcandP;
  std::vector<double> *t_PixTrkcandPt;
  std::vector<double> *t_PixTrkcandEta;
  std::vector<double> *t_PixTrkcandPhi;
  std::vector<double> *t_PixTrkcandMaxP;
  std::vector<bool> *t_PixTrkcandselTk;

  std::vector<double> *t_NFcandP;
  std::vector<double> *t_NFcandPt;
  std::vector<double> *t_NFcandEta;
  std::vector<double> *t_NFcandPhi;
  std::vector<double> *t_NFcandEmip;
  std::vector<double> *t_NFTrkcandP;
  std::vector<double> *t_NFTrkcandPt;
  std::vector<double> *t_NFTrkcandEta;
  std::vector<double> *t_NFTrkcandPhi;
  std::vector<double> *t_NFTrkcandEmip;
  std::vector<double> *t_NFTrkMinDR;
  std::vector<double> *t_NFTrkMinDP1;
  std::vector<bool> *t_NFTrkselTkFlag;
  std::vector<bool> *t_NFTrkqltyFlag;
  std::vector<bool> *t_NFTrkMissFlag;
  std::vector<bool> *t_NFTrkPVFlag;
  std::vector<bool> *t_NFTrkPropFlag;
  std::vector<bool> *t_NFTrkChgIsoFlag;
  std::vector<bool> *t_NFTrkNeuIsoFlag;
  std::vector<bool> *t_NFTrkMipFlag;
  std::vector<double> *t_ECone;

  TH1D *h_EnIn, *h_EnOut;
  TH2D *h_MipEnMatch, *h_MipEnNoMatch;
  TH1I *h_nHLT, *h_HLT, *h_PreL1, *h_PreHLT;
  TH1I *h_Pre, *h_nL3Objs, *h_Filters;
  TH1D *h_PreL1wt, *h_PreHLTwt, *h_L1ObjEnergy;
  TH1D *h_p[20], *h_pt[20], *h_eta[20], *h_phi[20];
  TH1D *h_dEtaL1[2], *h_dPhiL1[2], *h_dRL1[2];
  TH1D *h_dEta[9], *h_dPhi[9], *h_dPt[9], *h_dP[9];
  TH1D *h_dinvPt[9], *h_mindR[9], *h_eMip[2];
  TH1D *h_eMaxNearP[2], *h_eNeutIso[2];
  TH1D *h_etaCalibTracks[5][2][2], *h_etaMipTracks[5][2][2];
  TH1D *h_eHcal[5][6][48], *h_eCalo[5][6][48];
  TH1I *g_Pre, *g_PreL1, *g_PreHLT, *g_Accepts;
  std::vector<math::XYZTLorentzVector> vec_[3];
};

IsoTrig::IsoTrig(const edm::ParameterSet &iConfig)
    : hltPrescaleProvider_(iConfig, consumesCollector(), *this),
      trigNames_(iConfig.getUntrackedParameter<std::vector<std::string>>("Triggers")),
      pixCandTag_(iConfig.getUntrackedParameter<edm::InputTag>("pixCandTag")),
      l1CandTag_(iConfig.getUntrackedParameter<edm::InputTag>("l1CandTag")),
      l2CandTag_(iConfig.getUntrackedParameter<edm::InputTag>("l2CandTag")),
      pixelTracksSources_(iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("pixelTracksSources")),
      doL2L3_(iConfig.getUntrackedParameter<bool>("doL2L3", true)),
      doTiming_(iConfig.getUntrackedParameter<bool>("doTimingTree", true)),
      doMipCutTree_(iConfig.getUntrackedParameter<bool>("doMipCutTree", true)),
      doTrkResTree_(iConfig.getUntrackedParameter<bool>("doTrkResTree", true)),
      doChgIsolTree_(iConfig.getUntrackedParameter<bool>("doChgIsolTree", true)),
      doStudyIsol_(iConfig.getUntrackedParameter<bool>("doStudyIsol", true)),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      pixelIsolationConeSizeAtEC_(iConfig.getUntrackedParameter<std::vector<double>>("pixelIsolationConeSizeAtEC")),
      minPTrackValue_(iConfig.getUntrackedParameter<double>("minPTrackValue")),
      vtxCutSeed_(iConfig.getUntrackedParameter<double>("vertexCutSeed")),
      vtxCutIsol_(iConfig.getUntrackedParameter<double>("vertexCutIsol")),
      tauUnbiasCone_(iConfig.getUntrackedParameter<double>("tauUnbiasCone")),
      prelimCone_(iConfig.getUntrackedParameter<double>("prelimCone")),
      theTrackQuality_(iConfig.getUntrackedParameter<std::string>("trackQuality", "highPurity")),
      processName_(iConfig.getUntrackedParameter<std::string>("processName", "HLT")),
      dr_L1_(iConfig.getUntrackedParameter<double>("isolationL1", 1.0)),
      a_coneR_(iConfig.getUntrackedParameter<double>("coneRadius", 34.98)),
      a_charIsoR_(a_coneR_ + 28.9),
      a_neutIsoR_(a_charIsoR_ * 0.726),
      a_mipR_(iConfig.getUntrackedParameter<double>("coneRadiusMIP", 14.0)),
      a_neutR1_(iConfig.getUntrackedParameter<double>("coneRadiusNeut1", 21.0)),
      a_neutR2_(iConfig.getUntrackedParameter<double>("coneRadiusNeut2", 29.0)),
      cutMip_(iConfig.getUntrackedParameter<double>("cutMIP", 1.0)),
      cutCharge_(iConfig.getUntrackedParameter<double>("chargeIsolation", 2.0)),
      cutNeutral_(iConfig.getUntrackedParameter<double>("neutralIsolation", 2.0)),
      minRunNo_(iConfig.getUntrackedParameter<int>("minRun")),
      maxRunNo_(iConfig.getUntrackedParameter<int>("maxRun")),
      changed_(false),
      t_timeL2Prod(nullptr),
      t_nPixCand(nullptr),
      t_nPixSeed(nullptr),
      t_nGoodTk(nullptr),
      t_TrkhCone(nullptr),
      t_TrkP(nullptr),
      t_TrkselTkFlag(nullptr),
      t_TrkqltyFlag(nullptr),
      t_TrkMissFlag(nullptr),
      t_TrkPVFlag(nullptr),
      t_TrkNuIsolFlag(nullptr),
      t_PixcandP(nullptr),
      t_PixcandPt(nullptr),
      t_PixcandEta(nullptr),
      t_PixcandPhi(nullptr),
      t_PixcandMaxP(nullptr),
      t_PixTrkcandP(nullptr),
      t_PixTrkcandPt(nullptr),
      t_PixTrkcandEta(nullptr),
      t_PixTrkcandPhi(nullptr),
      t_PixTrkcandMaxP(nullptr),
      t_PixTrkcandselTk(nullptr),
      t_NFcandP(nullptr),
      t_NFcandPt(nullptr),
      t_NFcandEta(nullptr),
      t_NFcandPhi(nullptr),
      t_NFcandEmip(nullptr),
      t_NFTrkcandP(nullptr),
      t_NFTrkcandPt(nullptr),
      t_NFTrkcandEta(nullptr),
      t_NFTrkcandPhi(nullptr),
      t_NFTrkcandEmip(nullptr),
      t_NFTrkMinDR(nullptr),
      t_NFTrkMinDP1(nullptr),
      t_NFTrkselTkFlag(nullptr),
      t_NFTrkqltyFlag(nullptr),
      t_NFTrkMissFlag(nullptr),
      t_NFTrkPVFlag(nullptr),
      t_NFTrkPropFlag(nullptr),
      t_NFTrkChgIsoFlag(nullptr),
      t_NFTrkNeuIsoFlag(nullptr),
      t_NFTrkMipFlag(nullptr),
      t_ECone(nullptr) {
  usesResource(TFileService::kSharedResource);

  //now do whatever initialization is needed
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameters_.minPt = iConfig.getUntrackedParameter<double>("minTrackPt", 10.0);
  selectionParameters_.minQuality = trackQuality_;
  selectionParameters_.maxDxyPV = iConfig.getUntrackedParameter<double>("maxDxyPV", 0.2);
  selectionParameters_.maxDzPV = iConfig.getUntrackedParameter<double>("maxDzPV", 5.0);
  selectionParameters_.maxChi2 = iConfig.getUntrackedParameter<double>("maxChi2", 5.0);
  selectionParameters_.maxDpOverP = iConfig.getUntrackedParameter<double>("maxDpOverP", 0.1);
  selectionParameters_.minOuterHit = iConfig.getUntrackedParameter<int>("minOuterHit", 4);
  selectionParameters_.minLayerCrossed = iConfig.getUntrackedParameter<int>("minLayerCrossed", 8);
  selectionParameters_.maxInMiss = iConfig.getUntrackedParameter<int>("maxInMiss", 0);
  selectionParameters_.maxOutMiss = iConfig.getUntrackedParameter<int>("maxOutMiss", 0);

  // define tokens for access
  tok_lumi_ = consumes<LumiDetails, edm::InLumi>(edm::InputTag("lumiProducer"));
  edm::InputTag triggerEvent_("hltTriggerSummaryAOD", "", processName_);
  tok_trigEvt_ = consumes<trigger::TriggerEvent>(triggerEvent_);
  edm::InputTag theTriggerResultsLabel("TriggerResults", "", processName_);
  tok_trigRes_ = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  tok_pixtk_ = consumes<reco::IsolatedPixelTrackCandidateCollection>(pixCandTag_);
  tok_l1cand_ = consumes<trigger::TriggerFilterObjectWithRefs>(l1CandTag_);
  tok_l2cand_ = consumes<reco::IsolatedPixelTrackCandidateCollection>(l2CandTag_);
  if (doTiming_) {
    tok_verthb_ = consumes<reco::VertexCollection>(edm::InputTag("hltHITPixelVerticesHB"));
    tok_verthe_ = consumes<reco::VertexCollection>(edm::InputTag("hltHITPixelVerticesHB"));
    tok_hlt_ = consumes<trigger::TriggerFilterObjectWithRefs>(edm::InputTag("hltL1sL1SingleJet68"));
    tok_SeedingLayerHB_ = consumes<SeedingLayerSetsHits>(edm::InputTag("hltPixelLayerTripletsHITHB"));
    tok_SeedingLayerHE_ = consumes<SeedingLayerSetsHits>(edm::InputTag("hltPixelLayerTripletsHITHE"));
    tok_SiPixelRecHits_ = consumes<SiPixelRecHitCollection>(edm::InputTag("hltSiPixelRecHits"));
  }
  if (doChgIsolTree_) {
    for (unsigned int k = 0; k < pixelTracksSources_.size(); ++k) {
      //      edm::InputTag  pix (pixelTracksSources_[k],"",processName_);
      //      tok_pixtks_.push_back(consumes<reco::TrackCollection>(pix));
      tok_pixtks_.push_back(consumes<reco::TrackCollection>(pixelTracksSources_[k]));
    }
  }
  if (verbosity_ >= 0) {
    edm::LogVerbatim("IsoTrack") << "Parameters read from config file \n"
                                 << "\t minPt " << selectionParameters_.minPt << "\t theTrackQuality "
                                 << theTrackQuality_ << "\t minQuality " << selectionParameters_.minQuality
                                 << "\t maxDxyPV " << selectionParameters_.maxDxyPV << "\t maxDzPV "
                                 << selectionParameters_.maxDzPV << "\t maxChi2 " << selectionParameters_.maxChi2
                                 << "\t maxDpOverP " << selectionParameters_.maxDpOverP << "\t minOuterHit "
                                 << selectionParameters_.minOuterHit << "\t minLayerCrossed "
                                 << selectionParameters_.minLayerCrossed << "\t maxInMiss "
                                 << selectionParameters_.maxInMiss << "\t maxOutMiss "
                                 << selectionParameters_.maxOutMiss << "\t a_coneR " << a_coneR_ << "\t a_charIsoR "
                                 << a_charIsoR_ << "\t a_neutIsoR " << a_neutIsoR_ << "\t a_mipR " << a_mipR_
                                 << "\t a_neutR " << a_neutR1_ << ":" << a_neutR2_ << "\t cuts (MIP " << cutMip_
                                 << " : Charge " << cutCharge_ << " : Neutral " << cutNeutral_ << ")";
    edm::LogVerbatim("IsoTrack") << "Charge Isolation parameters:"
                                 << "\t minPTrackValue " << minPTrackValue_ << "\t vtxCutSeed " << vtxCutSeed_
                                 << "\t vtxCutIsol " << vtxCutIsol_ << "\t tauUnbiasCone " << tauUnbiasCone_
                                 << "\t prelimCone " << prelimCone_ << "\t pixelIsolationConeSizeAtEC";
    for (unsigned int k = 0; k < pixelIsolationConeSizeAtEC_.size(); ++k)
      edm::LogVerbatim("IsoTrack") << "[" << k << "] " << pixelIsolationConeSizeAtEC_[k];
  }
  double pl[] = {20, 30, 40, 60, 80, 120};
  for (int i = 0; i < 6; ++i)
    pLimits_[i] = pl[i];
  rEB_ = 123.8;
  zEE_ = 317.0;

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
}

IsoTrig::~IsoTrig() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  if (t_timeL2Prod)
    delete t_timeL2Prod;
  if (t_nPixCand)
    delete t_nPixCand;
  if (t_nPixSeed)
    delete t_nPixSeed;
  if (t_nGoodTk)
    delete t_nGoodTk;
  if (t_TrkhCone)
    delete t_TrkhCone;
  if (t_TrkP)
    delete t_TrkP;
  if (t_TrkselTkFlag)
    delete t_TrkselTkFlag;
  if (t_TrkqltyFlag)
    delete t_TrkqltyFlag;
  if (t_TrkMissFlag)
    delete t_TrkMissFlag;
  if (t_TrkPVFlag)
    delete t_TrkPVFlag;
  if (t_TrkNuIsolFlag)
    delete t_TrkNuIsolFlag;
  if (t_PixcandP)
    delete t_PixcandP;
  if (t_PixcandPt)
    delete t_PixcandPt;
  if (t_PixcandEta)
    delete t_PixcandEta;
  if (t_PixcandPhi)
    delete t_PixcandPhi;
  if (t_PixcandMaxP)
    delete t_PixcandMaxP;
  if (t_PixTrkcandP)
    delete t_PixTrkcandP;
  if (t_PixTrkcandPt)
    delete t_PixTrkcandPt;
  if (t_PixTrkcandEta)
    delete t_PixTrkcandEta;
  if (t_PixTrkcandPhi)
    delete t_PixTrkcandPhi;
  if (t_PixTrkcandMaxP)
    delete t_PixTrkcandMaxP;
  if (t_PixTrkcandselTk)
    delete t_PixTrkcandselTk;
  if (t_NFcandP)
    delete t_NFcandP;
  if (t_NFcandPt)
    delete t_NFcandPt;
  if (t_NFcandEta)
    delete t_NFcandEta;
  if (t_NFcandPhi)
    delete t_NFcandPhi;
  if (t_NFcandEmip)
    delete t_NFcandEmip;
  if (t_NFTrkcandP)
    delete t_NFTrkcandP;
  if (t_NFTrkcandPt)
    delete t_NFTrkcandPt;
  if (t_NFTrkcandEta)
    delete t_NFTrkcandEta;
  if (t_NFTrkcandPhi)
    delete t_NFTrkcandPhi;
  if (t_NFTrkcandEmip)
    delete t_NFTrkcandEmip;
  if (t_NFTrkMinDR)
    delete t_NFTrkMinDR;
  if (t_NFTrkMinDP1)
    delete t_NFTrkMinDP1;
  if (t_NFTrkselTkFlag)
    delete t_NFTrkselTkFlag;
  if (t_NFTrkqltyFlag)
    delete t_NFTrkqltyFlag;
  if (t_NFTrkMissFlag)
    delete t_NFTrkMissFlag;
  if (t_NFTrkPVFlag)
    delete t_NFTrkPVFlag;
  if (t_NFTrkPropFlag)
    delete t_NFTrkPropFlag;
  if (t_NFTrkChgIsoFlag)
    delete t_NFTrkChgIsoFlag;
  if (t_NFTrkNeuIsoFlag)
    delete t_NFTrkNeuIsoFlag;
  if (t_NFTrkMipFlag)
    delete t_NFTrkMipFlag;
  if (t_ECone)
    delete t_ECone;
}

void IsoTrig::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> triggers = {"HLT_IsoTrackHB"};
  std::vector<edm::InputTag> tags = {edm::InputTag("hltHITPixelTracksHB"), edm::InputTag("hltHITPixelTracksHE")};
  std::vector<double> cones = {35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 63.9, 70.0};
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("Triggers", triggers);
  desc.addUntracked<edm::InputTag>("pixCandTag", edm::InputTag(" "));
  desc.addUntracked<edm::InputTag>("l1CandTag", edm::InputTag("hltL1sV0SingleJet60"));
  desc.addUntracked<edm::InputTag>("l2CandTag", edm::InputTag("isolEcalPixelTrackProd"));
  desc.addUntracked<bool>("doL2L3", false);
  desc.addUntracked<bool>("doTimingTree", false);
  desc.addUntracked<bool>("doMipCutTree", false);
  desc.addUntracked<bool>("doTrkResTree", true);
  desc.addUntracked<bool>("doChgIsolTree", false);
  desc.addUntracked<bool>("doStudyIsol", false);
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<std::string>("processName", "HLT");
  desc.addUntracked<std::string>("trackQuality", "highPurity");
  desc.addUntracked<double>("minTrackPt", 10.0);
  desc.addUntracked<double>("maxDxyPV", 0.02);
  desc.addUntracked<double>("maxDzPV", 0.02);
  desc.addUntracked<double>("maxChi2", 5.0);
  desc.addUntracked<double>("maxDpOverP", 0.1);
  desc.addUntracked<int>("minOuterHit", 4);
  desc.addUntracked<int>("minLayerCrossed", 8);
  desc.addUntracked<int>("maxInMiss", 0);
  desc.addUntracked<int>("maxOutMiss", 0);
  desc.addUntracked<double>("isolationL1", 1.0);
  desc.addUntracked<double>("coneRadius", 34.98);
  desc.addUntracked<double>("coneRadiusMIP", 14.0);
  desc.addUntracked<double>("coneRadiusNeut1", 21.0);
  desc.addUntracked<double>("coneRadiusNeut2", 29.0);
  desc.addUntracked<double>("cutMIP", 1.0);
  desc.addUntracked<double>("chargeIsolation", 2.0);
  desc.addUntracked<double>("neutralIsolation", 2.0);
  desc.addUntracked<int>("minRun", 190456);
  desc.addUntracked<int>("maxRun", 203002);
  desc.addUntracked<std::vector<edm::InputTag>>("pixelTracksSources", tags);
  desc.addUntracked<std::vector<double>>("pixelIsolationConeSizeAtEC", cones);
  desc.addUntracked<double>("minPTrackValue", 0.0);
  desc.addUntracked<double>("vertexCutSeed", 101.0);
  desc.addUntracked<double>("vertexCutIsol", 101.0);
  desc.addUntracked<double>("tauUnbiasCone", 1.2);
  desc.addUntracked<double>("prelimCone", 1.0);
  desc.add<unsigned int>("stageL1Trigger", 1);
  descriptions.add("isoTrigDefault", desc);
}

void IsoTrig::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "Event starts====================================";

  int RunNo = iEvent.id().run();

  HLTConfigProvider const &hltConfig = hltPrescaleProvider_.hltConfigProvider();

  bField_ = &iSetup.getData(tok_magField_);
  geo_ = &iSetup.getData(tok_geom_);
  GlobalVector BField = bField_->inTesla(GlobalPoint(0, 0, 0));
  bfVal_ = BField.mag();

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt_, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product hltTriggerSummaryAOD";

  } else {
    triggerEvent = *(triggerEventHandle.product());
  }
  const trigger::TriggerObjectCollection &TOC(triggerEvent.getObjects());
  /////////////////////////////TriggerResults
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(tok_trigRes_, triggerResults);

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);

  iEvent.getByToken(tok_EB_, barrelRecHitsHandle_);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle_);

  iEvent.getByToken(tok_hbhe_, hbhe_);

  iEvent.getByToken(tok_recVtx_, recVtxs_);
  iEvent.getByToken(tok_bs_, beamSpotH_);
  if (!recVtxs_->empty() && !((*recVtxs_)[0].isFake())) {
    leadPV_ = math::XYZPoint((*recVtxs_)[0].x(), (*recVtxs_)[0].y(), (*recVtxs_)[0].z());
  } else if (beamSpotH_.isValid()) {
    leadPV_ = beamSpotH_->position();
  }

  if ((verbosity_ / 100) % 10 > 0) {
    edm::LogVerbatim("IsoTrack") << "Primary Vertex " << leadPV_;
    if (beamSpotH_.isValid())
      edm::LogVerbatim("IsoTrack") << "Beam Spot " << beamSpotH_->position();
  }
  pixelTrackRefsHE_.clear();
  pixelTrackRefsHB_.clear();
  for (unsigned int iPix = 0; iPix < pixelTracksSources_.size(); iPix++) {
    edm::Handle<reco::TrackCollection> iPixCol;
    iEvent.getByToken(tok_pixtks_[iPix], iPixCol);
    if (iPixCol.isValid()) {
      for (reco::TrackCollection::const_iterator pit = iPixCol->begin(); pit != iPixCol->end(); pit++) {
        if (iPix == 0)
          pixelTrackRefsHB_.push_back(reco::TrackRef(iPixCol, pit - iPixCol->begin()));
        pixelTrackRefsHE_.push_back(reco::TrackRef(iPixCol, pit - iPixCol->begin()));
      }
    }
  }
  if (doTiming_)
    getGoodTracks(iEvent, trkCollection);

  for (unsigned int ifilter = 0; ifilter < triggerEvent.sizeFilters(); ++ifilter) {
    std::string FilterNames[7] = {"hltL1sL1SingleJet68",
                                  "hltIsolPixelTrackL2FilterHE",
                                  "ecalIsolPixelTrackFilterHE",
                                  "hltIsolPixelTrackL3FilterHE",
                                  "hltIsolPixelTrackL2FilterHB",
                                  "ecalIsolPixelTrackFilterHB",
                                  "hltIsolPixelTrackL3FilterHB"};
    std::string label = triggerEvent.filterTag(ifilter).label();
    for (int i = 0; i < 7; i++) {
      if (label == FilterNames[i])
        h_Filters->Fill(i);
    }
  }
  edm::InputTag lumiProducer("LumiProducer", "", "RECO");
  edm::Handle<LumiDetails> Lumid;
  iEvent.getLuminosityBlock().getByToken(tok_lumi_, Lumid);
  float mybxlumi = -1;
  if (Lumid.isValid())
    mybxlumi = Lumid->lumiValue(LumiDetails::kOCC1, iEvent.bunchCrossing()) * 6.37;
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "RunNo " << RunNo << " EvtNo " << iEvent.id().event() << " Lumi "
                                 << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing() << " mybxlumi "
                                 << mybxlumi;
  if (!triggerResults.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product triggerResults";
    //      std::shared_ptr<cms::Exception> const & error = triggerResults.whyFailed();
    //      edm::LogWarning(error->category()) << error->what();
  } else {
    std::vector<std::string> modules;
    h_nHLT->Fill(triggerResults->size());
    const edm::TriggerNames &triggerNames = iEvent.triggerNames(*triggerResults);

    const std::vector<std::string> &triggerNames_ = triggerNames.triggerNames();
    if (verbosity_ % 10 > 1)
      edm::LogVerbatim("IsoTrack") << "number of HLTs " << triggerNames_.size();
    int hlt(-1), preL1(-1), preHLT(-1), prescale(-1);
    for (unsigned int i = 0; i < triggerResults->size(); i++) {
      unsigned int triggerindx = hltConfig.triggerIndex(triggerNames_[i]);
      const std::vector<std::string> &moduleLabels(hltConfig.moduleLabels(triggerindx));

      for (unsigned int in = 0; in < trigNames_.size(); ++in) {
        //	  if (triggerNames_[i].find(trigNames_[in].c_str())!=std::string::npos || triggerNames_[i]==" ") {
        if (triggerNames_[i].find(trigNames_[in]) != std::string::npos) {
          if (verbosity_ % 10 > 0)
            edm::LogVerbatim("IsoTrack") << "trigger that i want " << triggerNames_[i] << " accept "
                                         << triggerResults->accept(i);
          hlt = triggerResults->accept(i);
          h_HLT->Fill(hlt);
          //	    if (hlt>0 || triggerNames_[i]==" ") {
          if (hlt > 0) {
            edm::Handle<reco::IsolatedPixelTrackCandidateCollection> Pixcands;
            iEvent.getByToken(tok_pixtk_, Pixcands);
            edm::Handle<trigger::TriggerFilterObjectWithRefs> L1cands;
            iEvent.getByToken(tok_l1cand_, L1cands);

            const std::pair<int, int> prescales(hltPrescaleProvider_.prescaleValues(iEvent, iSetup, triggerNames_[i]));
            preL1 = prescales.first;
            preHLT = prescales.second;
            prescale = preL1 * preHLT;
            if (verbosity_ % 10 > 0)
              edm::LogVerbatim("IsoTrack")
                  << triggerNames_[i] << " accept " << hlt << " preL1 " << preL1 << " preHLT " << preHLT;
            for (int iv = 0; iv < 3; ++iv)
              vec_[iv].clear();
            if (trigList_.find(RunNo) != trigList_.end()) {
              trigList_[RunNo] += 1;
            } else {
              trigList_.insert(std::pair<unsigned int, unsigned int>(RunNo, 1));
              trigPreList_.insert(std::pair<unsigned int, std::pair<int, int>>(RunNo, prescales));
            }
            //loop over all trigger filters in event (i.e. filters passed)
            for (unsigned int ifilter = 0; ifilter < triggerEvent.sizeFilters(); ++ifilter) {
              std::vector<int> Keys;
              std::string label = triggerEvent.filterTag(ifilter).label();
              //loop over keys to objects passing this filter
              for (unsigned int imodule = 0; imodule < moduleLabels.size(); imodule++) {
                if (label.find(moduleLabels[imodule]) != std::string::npos) {
                  if (verbosity_ % 10 > 0)
                    edm::LogVerbatim("IsoTrack") << "FILTERNAME " << label;
                  for (unsigned int ifiltrKey = 0; ifiltrKey < triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
                    Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
                    const trigger::TriggerObject &TO(TOC[Keys[ifiltrKey]]);
                    math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
                    if (label.find("L2Filter") != std::string::npos) {
                      vec_[1].push_back(v4);
                    } else if (label.find("L3Filter") != std::string::npos) {
                      vec_[2].push_back(v4);
                    } else {
                      vec_[0].push_back(v4);
                      h_L1ObjEnergy->Fill(TO.energy());
                    }
                    if (verbosity_ % 10 > 0)
                      edm::LogVerbatim("IsoTrack") << "key " << ifiltrKey << " : pt " << TO.pt() << " eta " << TO.eta()
                                                   << " phi " << TO.phi() << " mass " << TO.mass() << " Id " << TO.id();
                  }
                }
              }
            }
            std::vector<reco::TrackCollection::const_iterator> goodTks;
            if (doL2L3_) {
              h_nL3Objs->Fill(vec_[2].size());
              studyTrigger(trkCollection, goodTks);
            } else {
              if (trkCollection.isValid()) {
                reco::TrackCollection::const_iterator trkItr;
                for (trkItr = trkCollection->begin(); trkItr != trkCollection->end(); trkItr++)
                  goodTks.push_back(trkItr);
              }
            }
            // Now study isolation etc
            if (doStudyIsol_)
              studyIsolation(trkCollection, goodTks);
            if (doTrkResTree_)
              StudyTrkEbyP(trkCollection);

            std::pair<double, double> etaphi = etaPhiTrigger();
            edm::Handle<reco::IsolatedPixelTrackCandidateCollection> L2cands;
            iEvent.getByToken(tok_l2cand_, L2cands);
            if (!L2cands.isValid()) {
              if (verbosity_ % 10 > 0)
                edm::LogVerbatim("IsoTrack") << " trigCand is not valid ";
            } else {
              if (doMipCutTree_)
                studyMipCut(trkCollection, L2cands);
            }
            if (!pixelTracksSources_.empty())
              if (doChgIsolTree_ && !pixelTrackRefsHE_.empty())
                chgIsolation(etaphi.first, etaphi.second, trkCollection, iEvent);
          }
          break;
        }
      }
    }
    h_PreL1->Fill(preL1);
    h_PreHLT->Fill(preHLT);
    h_Pre->Fill(prescale);
    h_PreL1wt->Fill(preL1, mybxlumi);
    h_PreHLTwt->Fill(preHLT, mybxlumi);

    // check if trigger names in (new) config
    //      edm::LogVerbatim("IsoTrack") << "changed " << changed_;
    if (changed_) {
      changed_ = false;
      if ((verbosity_ / 10) % 10 > 1) {
        edm::LogVerbatim("IsoTrack") << "New trigger menu found !!!";
        const unsigned int n(hltConfig.size());
        for (unsigned itrig = 0; itrig < triggerNames_.size(); itrig++) {
          unsigned int triggerindx = hltConfig.triggerIndex(triggerNames_[itrig]);
          if (triggerindx >= n)
            edm::LogVerbatim("IsoTrack") << triggerNames_[itrig] << " " << triggerindx << " does not exist in"
                                         << " the current menu";
          else
            edm::LogVerbatim("IsoTrack") << triggerNames_[itrig] << " " << triggerindx << " exists";
        }
      }
    }
  }
  if (doTiming_)
    studyTiming(iEvent);
}

void IsoTrig::clearChgIsolnTreeVectors() {
  t_PixcandP->clear();
  t_PixcandPt->clear();
  t_PixcandEta->clear();
  t_PixcandPhi->clear();
  for (unsigned int i = 0; i < t_PixcandMaxP->size(); i++)
    t_PixcandMaxP[i].clear();
  t_PixcandMaxP->clear();
  t_PixTrkcandP->clear();
  t_PixTrkcandPt->clear();
  t_PixTrkcandEta->clear();
  t_PixTrkcandPhi->clear();
  t_PixTrkcandMaxP->clear();
  t_PixTrkcandselTk->clear();
}

void IsoTrig::clearMipCutTreeVectors() {
  t_NFcandP->clear();
  t_NFcandPt->clear();
  t_NFcandEta->clear();
  t_NFcandPhi->clear();
  t_NFcandEmip->clear();
  t_NFTrkcandP->clear();
  t_NFTrkcandPt->clear();
  t_NFTrkcandEta->clear();
  t_NFTrkcandPhi->clear();
  t_NFTrkcandEmip->clear();
  t_NFTrkMinDR->clear();
  t_NFTrkMinDP1->clear();
  t_NFTrkselTkFlag->clear();
  t_NFTrkqltyFlag->clear();
  t_NFTrkMissFlag->clear();
  t_NFTrkPVFlag->clear();
  t_NFTrkPropFlag->clear();
  t_NFTrkChgIsoFlag->clear();
  t_NFTrkNeuIsoFlag->clear();
  t_NFTrkMipFlag->clear();
  t_ECone->clear();
}

void IsoTrig::pushChgIsolnTreeVecs(math::XYZTLorentzVector &Pixcand,
                                   math::XYZTLorentzVector &Trkcand,
                                   std::vector<double> &PixMaxP,
                                   double &TrkMaxP,
                                   bool &selTk) {
  t_PixcandP->push_back(Pixcand.r());
  t_PixcandPt->push_back(Pixcand.pt());
  t_PixcandEta->push_back(Pixcand.eta());
  t_PixcandPhi->push_back(Pixcand.phi());
  t_PixcandMaxP->push_back(PixMaxP);
  t_PixTrkcandP->push_back(Trkcand.r());
  t_PixTrkcandPt->push_back(Trkcand.pt());
  t_PixTrkcandEta->push_back(Trkcand.eta());
  t_PixTrkcandPhi->push_back(Trkcand.phi());
  t_PixTrkcandMaxP->push_back(TrkMaxP);
  t_PixTrkcandselTk->push_back(selTk);
}

void IsoTrig::pushMipCutTreeVecs(math::XYZTLorentzVector &NFcand,
                                 math::XYZTLorentzVector &Trkcand,
                                 double &EmipNFcand,
                                 double &EmipTrkcand,
                                 double &mindR,
                                 double &mindP1,
                                 std::vector<bool> &Flags,
                                 double hCone) {
  t_NFcandP->push_back(NFcand.r());
  t_NFcandPt->push_back(NFcand.pt());
  t_NFcandEta->push_back(NFcand.eta());
  t_NFcandPhi->push_back(NFcand.phi());
  t_NFcandEmip->push_back(EmipNFcand);
  t_NFTrkcandP->push_back(Trkcand.r());
  t_NFTrkcandPt->push_back(Trkcand.pt());
  t_NFTrkcandEta->push_back(Trkcand.eta());
  t_NFTrkcandPhi->push_back(Trkcand.phi());
  t_NFTrkcandEmip->push_back(EmipTrkcand);
  t_NFTrkMinDR->push_back(mindR);
  t_NFTrkMinDP1->push_back(mindP1);
  t_NFTrkselTkFlag->push_back(Flags[0]);
  t_NFTrkqltyFlag->push_back(Flags[1]);
  t_NFTrkMissFlag->push_back(Flags[2]);
  t_NFTrkPVFlag->push_back(Flags[3]);
  t_NFTrkPropFlag->push_back(Flags[4]);
  t_NFTrkChgIsoFlag->push_back(Flags[5]);
  t_NFTrkNeuIsoFlag->push_back(Flags[6]);
  t_NFTrkMipFlag->push_back(Flags[7]);
  t_ECone->push_back(hCone);
}

void IsoTrig::beginJob() {
  char hname[100], htit[100];
  std::string levels[20] = {"L1",        "L2",          "L3",        "Reco",      "RecoMatch",     "RecoNoMatch",
                            "L2Match",   "L2NoMatch",   "L3Match",   "L3NoMatch", "HLTTrk",        "HLTGoodTrk",
                            "HLTIsoTrk", "HLTMip",      "HLTSelect", "nonHLTTrk", "nonHLTGoodTrk", "nonHLTIsoTrk",
                            "nonHLTMip", "nonHLTSelect"};
  if (doTiming_) {
    TimingTree_ = fs_->make<TTree>("TimingTree", "TimingTree");
    t_timeL2Prod = new std::vector<double>();
    t_nPixCand = new std::vector<int>();
    t_nPixSeed = new std::vector<int>();
    t_nGoodTk = new std::vector<int>();

    TimingTree_->Branch("t_timeL2Prod", "std::vector<double>", &t_timeL2Prod);
    TimingTree_->Branch("t_nPixCand", "std::vector<int>", &t_nPixCand);
    TimingTree_->Branch("t_nPixSeed", "std::vector<int>", &t_nPixSeed);
    TimingTree_->Branch("t_nGoodTk", "std::vector<int>", &t_nGoodTk);
  }
  if (doTrkResTree_) {
    TrkResTree_ = fs_->make<TTree>("TrkRestree", "TrkResTree");
    t_TrkhCone = new std::vector<double>();
    t_TrkP = new std::vector<double>();
    t_TrkselTkFlag = new std::vector<bool>();
    t_TrkqltyFlag = new std::vector<bool>();
    t_TrkMissFlag = new std::vector<bool>();
    t_TrkPVFlag = new std::vector<bool>();
    t_TrkNuIsolFlag = new std::vector<bool>();

    TrkResTree_->Branch("t_TrkhCone", "std::vector<double>", &t_TrkhCone);
    TrkResTree_->Branch("t_TrkP", "std::vector<double>", &t_TrkP);
    TrkResTree_->Branch("t_TrkselTkFlag", "std::vector<bool>", &t_TrkselTkFlag);
    TrkResTree_->Branch("t_TrkqltyFlag", "std::vector<bool>", &t_TrkqltyFlag);
    TrkResTree_->Branch("t_TrkMissFlag", "std::vector<bool>", &t_TrkMissFlag);
    TrkResTree_->Branch("t_TrkPVFlag", "std::vector<bool>", &t_TrkPVFlag);
    TrkResTree_->Branch("t_TrkNuIsolFlag", "std::vector<bool>", &t_TrkNuIsolFlag);
  }
  if (doChgIsolTree_) {
    ChgIsolnTree_ = fs_->make<TTree>("ChgIsolnTree", "ChgIsolnTree");
    t_PixcandP = new std::vector<double>();
    t_PixcandPt = new std::vector<double>();
    t_PixcandEta = new std::vector<double>();
    t_PixcandPhi = new std::vector<double>();
    t_PixcandMaxP = new std::vector<std::vector<double>>();
    t_PixTrkcandP = new std::vector<double>();
    t_PixTrkcandPt = new std::vector<double>();
    t_PixTrkcandEta = new std::vector<double>();
    t_PixTrkcandPhi = new std::vector<double>();
    t_PixTrkcandMaxP = new std::vector<double>();
    t_PixTrkcandselTk = new std::vector<bool>();

    ChgIsolnTree_->Branch("t_PixcandP", "std::vector<double>", &t_PixcandP);
    ChgIsolnTree_->Branch("t_PixcandPt", "std::vector<double>", &t_PixcandPt);
    ChgIsolnTree_->Branch("t_PixcandEta", "std::vector<double>", &t_PixcandEta);
    ChgIsolnTree_->Branch("t_PixcandPhi", "std::vector<double>", &t_PixcandPhi);
    ChgIsolnTree_->Branch("t_PixcandMaxP", "std::vector<std::vector<double> >", &t_PixcandMaxP);
    ChgIsolnTree_->Branch("t_PixTrkcandP", "std::vector<double>", &t_PixTrkcandP);
    ChgIsolnTree_->Branch("t_PixTrkcandPt", "std::vector<double>", &t_PixTrkcandPt);
    ChgIsolnTree_->Branch("t_PixTrkcandEta", "std::vector<double>", &t_PixTrkcandEta);
    ChgIsolnTree_->Branch("t_PixTrkcandPhi", "std::vector<double>", &t_PixTrkcandPhi);
    ChgIsolnTree_->Branch("t_PixTrkcandMaxP", "std::vector<double>", &t_PixTrkcandMaxP);
    ChgIsolnTree_->Branch("t_PixTrkcandselTk", "std::vector<bool>", &t_PixTrkcandselTk);
  }
  if (doMipCutTree_) {
    MipCutTree_ = fs_->make<TTree>("MipCutTree", "MipCutTree");
    t_NFcandP = new std::vector<double>();
    t_NFcandPt = new std::vector<double>();
    t_NFcandEta = new std::vector<double>();
    t_NFcandPhi = new std::vector<double>();
    t_NFcandEmip = new std::vector<double>();
    t_NFTrkcandP = new std::vector<double>();
    t_NFTrkcandPt = new std::vector<double>();
    t_NFTrkcandEta = new std::vector<double>();
    t_NFTrkcandPhi = new std::vector<double>();
    t_NFTrkcandEmip = new std::vector<double>();
    t_NFTrkMinDR = new std::vector<double>();
    t_NFTrkMinDP1 = new std::vector<double>();
    t_NFTrkselTkFlag = new std::vector<bool>();
    t_NFTrkqltyFlag = new std::vector<bool>();
    t_NFTrkMissFlag = new std::vector<bool>();
    t_NFTrkPVFlag = new std::vector<bool>();
    t_NFTrkPropFlag = new std::vector<bool>();
    t_NFTrkChgIsoFlag = new std::vector<bool>();
    t_NFTrkNeuIsoFlag = new std::vector<bool>();
    t_NFTrkMipFlag = new std::vector<bool>();
    t_ECone = new std::vector<double>();

    MipCutTree_->Branch("t_NFcandP", "std::vector<double>", &t_NFcandP);
    MipCutTree_->Branch("t_NFcandPt", "std::vector<double>", &t_NFcandPt);
    MipCutTree_->Branch("t_NFcandEta", "std::vector<double>", &t_NFcandEta);
    MipCutTree_->Branch("t_NFcandPhi", "std::vector<double>", &t_NFcandPhi);
    MipCutTree_->Branch("t_NFcandEmip", "std::vector<double>", &t_NFcandEmip);
    MipCutTree_->Branch("t_NFTrkcandP", "std::vector<double>", &t_NFTrkcandP);
    MipCutTree_->Branch("t_NFTrkcandPt", "std::vector<double>", &t_NFTrkcandPt);
    MipCutTree_->Branch("t_NFTrkcandEta", "std::vector<double>", &t_NFTrkcandEta);
    MipCutTree_->Branch("t_NFTrkcandPhi", "std::vector<double>", &t_NFTrkcandPhi);
    MipCutTree_->Branch("t_NFTrkcandEmip", "std::vector<double>", &t_NFTrkcandEmip);
    MipCutTree_->Branch("t_NFTrkMinDR", "std::vector<double>", &t_NFTrkMinDR);
    MipCutTree_->Branch("t_NFTrkMinDP1", "std::vector<double>", &t_NFTrkMinDP1);
    MipCutTree_->Branch("t_NFTrkselTkFlag", "std::vector<bool>", &t_NFTrkselTkFlag);
    MipCutTree_->Branch("t_NFTrkqltyFlag", "std::vector<bool>", &t_NFTrkqltyFlag);
    MipCutTree_->Branch("t_NFTrkMissFlag", "std::vector<bool>", &t_NFTrkMissFlag);
    MipCutTree_->Branch("t_NFTrkPVFlag", "std::vector<bool>", &t_NFTrkPVFlag);
    MipCutTree_->Branch("t_NFTrkPropFlag", "std::vector<bool>", &t_NFTrkPropFlag);
    MipCutTree_->Branch("t_NFTrkChgIsoFlag", "std::vector<bool>", &t_NFTrkChgIsoFlag);
    MipCutTree_->Branch("t_NFTrkNeuIsoFlag", "std::vector<bool>", &t_NFTrkNeuIsoFlag);
    MipCutTree_->Branch("t_NFTrkMipFlag", "std::vector<bool>", &t_NFTrkMipFlag);
    MipCutTree_->Branch("t_ECone", "std::vector<double>", &t_ECone);
  }
  h_Filters = fs_->make<TH1I>("h_Filters", "Filter Accepts", 10, 0, 10);
  std::string FilterNames[7] = {"hltL1sL1SingleJet68",
                                "hltIsolPixelTrackL2FilterHE",
                                "ecalIsolPixelTrackFilterHE",
                                "hltIsolPixelTrackL3FilterHE",
                                "hltIsolPixelTrackL2FilterHB",
                                "ecalIsolPixelTrackFilterHB",
                                "hltIsolPixelTrackL3FilterHB"};
  for (int i = 0; i < 7; i++) {
    h_Filters->GetXaxis()->SetBinLabel(i + 1, FilterNames[i].c_str());
  }

  h_nHLT = fs_->make<TH1I>("h_nHLT", "Size of trigger Names", 1000, 1, 1000);
  h_HLT = fs_->make<TH1I>("h_HLT", "HLT accept", 3, -1, 2);
  h_PreL1 = fs_->make<TH1I>("h_PreL1", "L1 Prescale", 500, 0, 500);
  h_PreHLT = fs_->make<TH1I>("h_PreHLT", "HLT Prescale", 50, 0, 50);
  h_Pre = fs_->make<TH1I>("h_Pre", "Prescale", 3000, 0, 3000);

  h_PreL1wt = fs_->make<TH1D>("h_PreL1wt", "Weighted L1 Prescale", 500, 0, 500);
  h_PreHLTwt = fs_->make<TH1D>("h_PreHLTwt", "Weighted HLT Prescale", 500, 0, 100);
  h_L1ObjEnergy = fs_->make<TH1D>("h_L1ObjEnergy", "Energy of L1Object", 500, 0.0, 500.0);

  h_EnIn = fs_->make<TH1D>("h_EnInEcal", "EnergyIn Ecal", 200, 0.0, 20.0);
  h_EnOut = fs_->make<TH1D>("h_EnOutEcal", "EnergyOut Ecal", 200, 0.0, 20.0);
  h_MipEnMatch =
      fs_->make<TH2D>("h_MipEnMatch", "MipEn: HLT level vs Reco Level (Matched)", 200, 0.0, 20.0, 200, 0.0, 20.0);
  h_MipEnNoMatch = fs_->make<TH2D>(
      "h_MipEnNoMatch", "MipEn: HLT level vs Reco Level (No Match Found)", 200, 0.0, 20.0, 200, 0.0, 20.0);

  if (doL2L3_) {
    h_nL3Objs = fs_->make<TH1I>("h_nL3Objs", "Number of L3 objects", 10, 0, 10);

    std::string pairs[9] = {"L2L3",
                            "L2L3Match",
                            "L2L3NoMatch",
                            "L3Reco",
                            "L3RecoMatch",
                            "L3RecoNoMatch",
                            "NewFilterReco",
                            "NewFilterRecoMatch",
                            "NewFilterRecoNoMatch"};
    for (int ipair = 0; ipair < 9; ipair++) {
      sprintf(hname, "h_dEta%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta#eta for %s", pairs[ipair].c_str());
      h_dEta[ipair] = fs_->make<TH1D>(hname, htit, 200, -10.0, 10.0);
      h_dEta[ipair]->GetXaxis()->SetTitle("d#eta");

      sprintf(hname, "h_dPhi%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta#phi for %s", pairs[ipair].c_str());
      h_dPhi[ipair] = fs_->make<TH1D>(hname, htit, 140, -7.0, 7.0);
      h_dPhi[ipair]->GetXaxis()->SetTitle("d#phi");

      sprintf(hname, "h_dPt%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta dp_{T} for %s objects", pairs[ipair].c_str());
      h_dPt[ipair] = fs_->make<TH1D>(hname, htit, 400, -200.0, 200.0);
      h_dPt[ipair]->GetXaxis()->SetTitle("dp_{T} (GeV)");

      sprintf(hname, "h_dP%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta p for %s objects", pairs[ipair].c_str());
      h_dP[ipair] = fs_->make<TH1D>(hname, htit, 400, -200.0, 200.0);
      h_dP[ipair]->GetXaxis()->SetTitle("dP (GeV)");

      sprintf(hname, "h_dinvPt%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta (1/p_{T}) for %s objects", pairs[ipair].c_str());
      h_dinvPt[ipair] = fs_->make<TH1D>(hname, htit, 500, -0.4, 0.1);
      h_dinvPt[ipair]->GetXaxis()->SetTitle("d(1/p_{T})");
      sprintf(hname, "h_mindR%s", pairs[ipair].c_str());
      sprintf(htit, "min(#Delta R) for %s objects", pairs[ipair].c_str());
      h_mindR[ipair] = fs_->make<TH1D>(hname, htit, 500, 0.0, 1.0);
      h_mindR[ipair]->GetXaxis()->SetTitle("dR");
    }

    for (int lvl = 0; lvl < 2; lvl++) {
      sprintf(hname, "h_dEtaL1%s", levels[lvl + 1].c_str());
      sprintf(htit, "#Delta#eta for L1 and %s objects", levels[lvl + 1].c_str());
      h_dEtaL1[lvl] = fs_->make<TH1D>(hname, htit, 400, -10.0, 10.0);

      sprintf(hname, "h_dPhiL1%s", levels[lvl + 1].c_str());
      sprintf(htit, "#Delta#phi for L1 and %s objects", levels[lvl + 1].c_str());
      h_dPhiL1[lvl] = fs_->make<TH1D>(hname, htit, 280, -7.0, 7.0);

      sprintf(hname, "h_dRL1%s", levels[lvl + 1].c_str());
      sprintf(htit, "#Delta R for L1 and %s objects", levels[lvl + 1].c_str());
      h_dRL1[lvl] = fs_->make<TH1D>(hname, htit, 100, 0.0, 10.0);
    }
  }

  int levmin = (doL2L3_ ? 0 : 10);
  for (int ilevel = levmin; ilevel < 20; ilevel++) {
    sprintf(hname, "h_p%s", levels[ilevel].c_str());
    sprintf(htit, "p for %s objects", levels[ilevel].c_str());
    h_p[ilevel] = fs_->make<TH1D>(hname, htit, 100, 0.0, 500.0);
    h_p[ilevel]->GetXaxis()->SetTitle("p (GeV)");

    sprintf(hname, "h_pt%s", levels[ilevel].c_str());
    sprintf(htit, "p_{T} for %s objects", levels[ilevel].c_str());
    h_pt[ilevel] = fs_->make<TH1D>(hname, htit, 100, 0.0, 500.0);
    h_pt[ilevel]->GetXaxis()->SetTitle("p_{T} (GeV)");

    sprintf(hname, "h_eta%s", levels[ilevel].c_str());
    sprintf(htit, "#eta for %s objects", levels[ilevel].c_str());
    h_eta[ilevel] = fs_->make<TH1D>(hname, htit, 100, -5.0, 5.0);
    h_eta[ilevel]->GetXaxis()->SetTitle("#eta");

    sprintf(hname, "h_phi%s", levels[ilevel].c_str());
    sprintf(htit, "#phi for %s objects", levels[ilevel].c_str());
    h_phi[ilevel] = fs_->make<TH1D>(hname, htit, 70, -3.5, 3.50);
    h_phi[ilevel]->GetXaxis()->SetTitle("#phi");
  }

  std::string cuts[2] = {"HLTMatched", "HLTNotMatched"};
  std::string cuts2[2] = {"All", "Away from L1"};
  for (int icut = 0; icut < 2; icut++) {
    sprintf(hname, "h_eMip%s", cuts[icut].c_str());
    sprintf(htit, "eMip for %s tracks", cuts[icut].c_str());
    h_eMip[icut] = fs_->make<TH1D>(hname, htit, 200, 0.0, 10.0);
    h_eMip[icut]->GetXaxis()->SetTitle("E_{Mip} (GeV)");

    sprintf(hname, "h_eMaxNearP%s", cuts[icut].c_str());
    sprintf(htit, "eMaxNearP for %s tracks", cuts[icut].c_str());
    h_eMaxNearP[icut] = fs_->make<TH1D>(hname, htit, 240, -2.0, 10.0);
    h_eMaxNearP[icut]->GetXaxis()->SetTitle("E_{MaxNearP} (GeV)");

    sprintf(hname, "h_eNeutIso%s", cuts[icut].c_str());
    sprintf(htit, "eNeutIso for %s ", cuts[icut].c_str());
    h_eNeutIso[icut] = fs_->make<TH1D>(hname, htit, 200, 0.0, 10.0);
    h_eNeutIso[icut]->GetXaxis()->SetTitle("E_{NeutIso} (GeV)");

    for (int kcut = 0; kcut < 2; ++kcut) {
      for (int lim = 0; lim < 5; ++lim) {
        sprintf(hname, "h_etaCalibTracks%sCut%dLim%d", cuts[icut].c_str(), kcut, lim);
        sprintf(htit,
                "#eta for %s isolated MIP tracks (%4.1f < p < %5.1f Gev/c %s)",
                cuts[icut].c_str(),
                pLimits_[lim],
                pLimits_[lim + 1],
                cuts2[kcut].c_str());
        h_etaCalibTracks[lim][icut][kcut] = fs_->make<TH1D>(hname, htit, 60, -30.0, 30.0);
        h_etaCalibTracks[lim][icut][kcut]->GetXaxis()->SetTitle("i#eta");

        sprintf(hname, "h_etaMipTracks%sCut%dLim%d", cuts[icut].c_str(), kcut, lim);
        sprintf(htit,
                "#eta for %s charge isolated MIP tracks (%4.1f < p < %5.1f Gev/c %s)",
                cuts[icut].c_str(),
                pLimits_[lim],
                pLimits_[lim + 1],
                cuts2[kcut].c_str());
        h_etaMipTracks[lim][icut][kcut] = fs_->make<TH1D>(hname, htit, 60, -30.0, 30.0);
        h_etaMipTracks[lim][icut][kcut]->GetXaxis()->SetTitle("i#eta");
      }
    }
  }

  std::string ecut1[3] = {"all", "HLTMatched", "HLTNotMatched"};
  std::string ecut2[2] = {"without", "with"};
  int etac[48] = {-1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10, -11, -12, -13, -14, -15, -16,
                  -17, -18, -19, -20, -21, -22, -23, -24, 1,  2,   3,   4,   5,   6,   7,   8,
                  9,   10,  11,  12,  13,  14,  15,  16,  17, 18,  19,  20,  21,  22,  23,  24};
  for (int icut = 0; icut < 6; icut++) {
    //    int i1 = (icut>3 ? 1 : 0);
    int i1 = (icut > 2 ? 1 : 0);
    int i2 = icut - i1 * 3;
    for (int kcut = 0; kcut < 48; kcut++) {
      for (int lim = 0; lim < 5; ++lim) {
        sprintf(hname, "h_eta%dEnHcal%s%s%d", etac[kcut], ecut1[i2].c_str(), ecut2[i1].c_str(), lim);
        sprintf(htit,
                "HCAL energy for #eta=%d for %s tracks (p=%4.1f:%5.1f Gev) %s neutral isolation",
                etac[kcut],
                ecut1[i2].c_str(),
                pLimits_[lim],
                pLimits_[lim + 1],
                ecut2[i1].c_str());
        h_eHcal[lim][icut][kcut] = fs_->make<TH1D>(hname, htit, 750, 0.0, 150.0);
        h_eHcal[lim][icut][kcut]->GetXaxis()->SetTitle("Energy (GeV)");
        sprintf(hname, "h_eta%dEnCalo%s%s%d", etac[kcut], ecut1[i2].c_str(), ecut2[i1].c_str(), lim);
        sprintf(htit,
                "Calorimter energy for #eta=%d for %s tracks (p=%4.1f:%5.1f Gev) %s neutral isolation",
                etac[kcut],
                ecut1[i2].c_str(),
                pLimits_[lim],
                pLimits_[lim + 1],
                ecut2[i1].c_str());
        h_eCalo[lim][icut][kcut] = fs_->make<TH1D>(hname, htit, 750, 0.0, 150.0);
        h_eCalo[lim][icut][kcut]->GetXaxis()->SetTitle("Energy (GeV)");
      }
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void IsoTrig::endJob() {
  unsigned int preL1, preHLT;
  std::map<unsigned int, unsigned int>::iterator itr;
  std::map<unsigned int, const std::pair<int, int>>::iterator itrPre;
  edm::LogWarning("IsoTrack") << trigNames_.size() << "Triggers were run. RunNo vs HLT accepts for";
  for (unsigned int i = 0; i < trigNames_.size(); ++i)
    edm::LogWarning("IsoTrack") << "[" << i << "]: " << trigNames_[i];
  unsigned int n = maxRunNo_ - minRunNo_ + 1;
  g_Pre = fs_->make<TH1I>("h_PrevsRN", "PreScale Vs Run Number", n, minRunNo_, maxRunNo_);
  g_PreL1 = fs_->make<TH1I>("h_PreL1vsRN", "L1 PreScale Vs Run Number", n, minRunNo_, maxRunNo_);
  g_PreHLT = fs_->make<TH1I>("h_PreHLTvsRN", "HLT PreScale Vs Run Number", n, minRunNo_, maxRunNo_);
  g_Accepts = fs_->make<TH1I>("h_HLTAcceptsvsRN", "HLT Accepts Vs Run Number", n, minRunNo_, maxRunNo_);

  for (itr = trigList_.begin(), itrPre = trigPreList_.begin(); itr != trigList_.end(); itr++, itrPre++) {
    preL1 = (itrPre->second).first;
    preHLT = (itrPre->second).second;
    edm::LogVerbatim("IsoTrack") << itr->first << " " << itr->second << " " << itrPre->first << " " << preL1 << " "
                                 << preHLT;
    g_Accepts->Fill(itr->first, itr->second);
    g_PreL1->Fill(itr->first, preL1);
    g_PreHLT->Fill(itr->first, preHLT);
    g_Pre->Fill(itr->first, preL1 * preHLT);
  }
}

void IsoTrig::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  edm::LogWarning("IsoTrack") << "Run " << iRun.run() << " hltconfig.init "
                              << hltPrescaleProvider_.init(iRun, iSetup, processName_, changed_);
}

void IsoTrig::StudyTrkEbyP(edm::Handle<reco::TrackCollection> &trkCollection) {
  t_TrkselTkFlag->clear();
  t_TrkqltyFlag->clear();
  t_TrkMissFlag->clear();
  t_TrkPVFlag->clear();
  t_TrkNuIsolFlag->clear();
  t_TrkhCone->clear();
  t_TrkP->clear();

  if (!trkCollection.isValid()) {
    edm::LogVerbatim("IsoTrack") << "trkCollection.isValid is false";
  } else {
    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
    std::vector<spr::propagatedTrackDirection> trkCaloDirections1;
    spr::propagateCALO(
        trkCollection, geo_, bField_, theTrackQuality_, trkCaloDirections1, ((verbosity_ / 100) % 10 > 2));
    unsigned int nTracks = 0;
    int nRH_eMipDR = 0, nNearTRKs = 0;
    std::vector<bool> selFlags;
    for (trkDetItr = trkCaloDirections1.begin(); trkDetItr != trkCaloDirections1.end(); trkDetItr++, nTracks++) {
      double conehmaxNearP = 0, hCone = 0, eMipDR = 0.0;
      const reco::Track *pTrack = &(*(trkDetItr->trkItr));
      if (verbosity_ % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "track no. " << nTracks << " p(): " << pTrack->p();
      if (pTrack->p() > 20) {
        math::XYZTLorentzVector v2(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
        eMipDR = spr::eCone_ecal(geo_,
                                 barrelRecHitsHandle_,
                                 endcapRecHitsHandle_,
                                 trkDetItr->pointHCAL,
                                 trkDetItr->pointECAL,
                                 a_mipR_,
                                 trkDetItr->directionECAL,
                                 nRH_eMipDR);
        bool selectTk = spr::goodTrack(pTrack, leadPV_, selectionParameters_, ((verbosity_ / 100) % 10 > 1));
        spr::trackSelectionParameters oneCutParameters = selectionParameters_;
        oneCutParameters.maxDxyPV = 10;
        oneCutParameters.maxDzPV = 100;
        oneCutParameters.maxInMiss = 2;
        oneCutParameters.maxOutMiss = 2;
        bool qltyFlag = spr::goodTrack(pTrack, leadPV_, oneCutParameters, ((verbosity_ / 100) % 10 > 1));
        oneCutParameters = selectionParameters_;
        oneCutParameters.maxDxyPV = 10;
        oneCutParameters.maxDzPV = 100;
        bool qltyMissFlag = spr::goodTrack(pTrack, leadPV_, oneCutParameters, ((verbosity_ / 100) % 10 > 1));
        oneCutParameters = selectionParameters_;
        oneCutParameters.maxInMiss = 2;
        oneCutParameters.maxOutMiss = 2;
        bool qltyPVFlag = spr::goodTrack(pTrack, leadPV_, oneCutParameters, ((verbosity_ / 100) % 10 > 1));
        if ((verbosity_ / 1000) % 10 > 1)
          edm::LogVerbatim("IsoTrack") << "sel " << selectTk << "ntracks " << nTracks << " a_charIsoR " << a_charIsoR_
                                       << " nNearTRKs " << nNearTRKs;
        conehmaxNearP = spr::chargeIsolationCone(
            nTracks, trkCaloDirections1, a_charIsoR_, nNearTRKs, ((verbosity_ / 100) % 10 > 1));
        if ((verbosity_ / 1000) % 10 > 1)
          edm::LogVerbatim("IsoTrack") << "coneh " << conehmaxNearP << "ok " << trkDetItr->okECAL << " "
                                       << trkDetItr->okHCAL;
        double e1 = spr::eCone_ecal(geo_,
                                    barrelRecHitsHandle_,
                                    endcapRecHitsHandle_,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_neutR1_,
                                    trkDetItr->directionECAL,
                                    nRH_eMipDR);
        double e2 = spr::eCone_ecal(geo_,
                                    barrelRecHitsHandle_,
                                    endcapRecHitsHandle_,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_neutR2_,
                                    trkDetItr->directionECAL,
                                    nRH_eMipDR);
        double e_inCone = e2 - e1;
        bool chgIsolFlag = (conehmaxNearP < cutCharge_);
        bool mipFlag = (eMipDR < cutMip_);
        bool neuIsolFlag = (e_inCone < cutNeutral_);
        bool trkpropFlag = ((trkDetItr->okECAL) && (trkDetItr->okHCAL));
        selFlags.clear();
        selFlags.push_back(selectTk);
        selFlags.push_back(qltyFlag);
        selFlags.push_back(qltyMissFlag);
        selFlags.push_back(qltyPVFlag);
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "emip: " << eMipDR << "<" << cutMip_ << "(" << mipFlag << ")"
                                       << " ; ok: " << trkDetItr->okECAL << "/" << trkDetItr->okHCAL
                                       << " ; chgiso: " << conehmaxNearP << "<" << cutCharge_ << "(" << chgIsolFlag
                                       << ")";

        if (chgIsolFlag && mipFlag && trkpropFlag) {
          double distFromHotCell = -99.0;
          int nRecHitsCone = -99, ietaHotCell = -99, iphiHotCell = -99;
          GlobalPoint gposHotCell(0., 0., 0.);
          std::vector<DetId> coneRecHitDetIds;
          hCone = spr::eCone_hcal(geo_,
                                  hbhe_,
                                  trkDetItr->pointHCAL,
                                  trkDetItr->pointECAL,
                                  a_coneR_,
                                  trkDetItr->directionHCAL,
                                  nRecHitsCone,
                                  coneRecHitDetIds,
                                  distFromHotCell,
                                  ietaHotCell,
                                  iphiHotCell,
                                  gposHotCell,
                                  -1);
          // push vectors into the Tree
          t_TrkselTkFlag->push_back(selFlags[0]);
          t_TrkqltyFlag->push_back(selFlags[1]);
          t_TrkMissFlag->push_back(selFlags[2]);
          t_TrkPVFlag->push_back(selFlags[3]);
          t_TrkNuIsolFlag->push_back(neuIsolFlag);
          t_TrkhCone->push_back(hCone);
          t_TrkP->push_back(pTrack->p());
        }
      }
    }
    if (verbosity_ % 10 > 0)
      edm::LogVerbatim("IsoTrack") << "Filling " << t_TrkP->size() << " tracks in TrkRestree out of " << nTracks;
  }
  TrkResTree_->Fill();
}

void IsoTrig::studyTiming(const edm::Event &theEvent) {
  t_timeL2Prod->clear();
  t_nPixCand->clear();
  t_nPixSeed->clear();

  if (verbosity_ % 10 > 0) {
    edm::Handle<SeedingLayerSetsHits> hblayers, helayers;
    theEvent.getByToken(tok_SeedingLayerHB_, hblayers);
    theEvent.getByToken(tok_SeedingLayerHE_, helayers);
    const SeedingLayerSetsHits *layershb = hblayers.product();
    const SeedingLayerSetsHits *layershe = helayers.product();
    edm::LogVerbatim("IsoTrack") << "size of Seeding TripletLayers hb/he " << layershb->size() << "/"
                                 << layershe->size();
    edm::Handle<SiPixelRecHitCollection> rchts;
    theEvent.getByToken(tok_SiPixelRecHits_, rchts);
    const SiPixelRecHitCollection *rechits = rchts.product();
    edm::LogVerbatim("IsoTrack") << "size of SiPixelRechits " << rechits->size();
  }
  int nCandHB = pixelTrackRefsHB_.size(), nCandHE = pixelTrackRefsHE_.size();
  int nSeedHB = 0, nSeedHE = 0;

  if (nCandHE > 0) {
    edm::Handle<reco::VertexCollection> pVertHB, pVertHE;
    theEvent.getByToken(tok_verthb_, pVertHB);
    theEvent.getByToken(tok_verthe_, pVertHE);
    edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
    theEvent.getByToken(tok_l1cand_, l1trigobj);

    std::vector<edm::Ref<l1extra::L1JetParticleCollection>> l1tauobjref;
    std::vector<edm::Ref<l1extra::L1JetParticleCollection>> l1jetobjref;
    std::vector<edm::Ref<l1extra::L1JetParticleCollection>> l1forjetobjref;

    l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
    l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
    l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);

    double ptTriggered = -10;
    double etaTriggered = -100;
    double phiTriggered = -100;
    for (unsigned int p = 0; p < l1tauobjref.size(); p++) {
      if (l1tauobjref[p]->pt() > ptTriggered) {
        ptTriggered = l1tauobjref[p]->pt();
        phiTriggered = l1tauobjref[p]->phi();
        etaTriggered = l1tauobjref[p]->eta();
      }
    }
    for (unsigned int p = 0; p < l1jetobjref.size(); p++) {
      if (l1jetobjref[p]->pt() > ptTriggered) {
        ptTriggered = l1jetobjref[p]->pt();
        phiTriggered = l1jetobjref[p]->phi();
        etaTriggered = l1jetobjref[p]->eta();
      }
    }
    for (unsigned int p = 0; p < l1forjetobjref.size(); p++) {
      if (l1forjetobjref[p]->pt() > ptTriggered) {
        ptTriggered = l1forjetobjref[p]->pt();
        phiTriggered = l1forjetobjref[p]->phi();
        etaTriggered = l1forjetobjref[p]->eta();
      }
    }
    for (unsigned iS = 0; iS < pixelTrackRefsHE_.size(); iS++) {
      reco::VertexCollection::const_iterator vitSel;
      double minDZ = 100;
      bool vtxMatch;
      for (reco::VertexCollection::const_iterator vit = pVertHE->begin(); vit != pVertHE->end(); vit++) {
        if (fabs(pixelTrackRefsHE_[iS]->dz(vit->position())) < minDZ) {
          minDZ = fabs(pixelTrackRefsHE_[iS]->dz(vit->position()));
          vitSel = vit;
        }
      }
      //cut on dYX:
      if ((fabs(pixelTrackRefsHE_[iS]->dxy(vitSel->position())) < vtxCutSeed_) || (minDZ == 100))
        vtxMatch = true;

      //select tracks not matched to triggered L1 jet
      double R = deltaR(etaTriggered, phiTriggered, pixelTrackRefsHE_[iS]->eta(), pixelTrackRefsHE_[iS]->phi());
      if (R > tauUnbiasCone_ && vtxMatch)
        nSeedHE++;
    }
    for (unsigned iS = 0; iS < pixelTrackRefsHB_.size(); iS++) {
      reco::VertexCollection::const_iterator vitSel;
      double minDZ = 100;
      bool vtxMatch(false);
      for (reco::VertexCollection::const_iterator vit = pVertHB->begin(); vit != pVertHB->end(); vit++) {
        if (fabs(pixelTrackRefsHB_[iS]->dz(vit->position())) < minDZ) {
          minDZ = fabs(pixelTrackRefsHB_[iS]->dz(vit->position()));
          vitSel = vit;
        }
      }
      //cut on dYX:
      if ((fabs(pixelTrackRefsHB_[iS]->dxy(vitSel->position())) < 101.0) || (minDZ == 100))
        vtxMatch = true;

      //select tracks not matched to triggered L1 jet
      double R = deltaR(etaTriggered, phiTriggered, pixelTrackRefsHB_[iS]->eta(), pixelTrackRefsHB_[iS]->phi());
      if (R > 1.2 && vtxMatch)
        nSeedHB++;
    }

    if (verbosity_ % 10 > 0) {
      edm::LogVerbatim("IsoTrack") << "(HB/HE) nCand: " << nCandHB << "/" << nCandHE << "nSeed: " << nSeedHB << "/"
                                   << nSeedHE;
    }
  }
  t_nPixSeed->push_back(nSeedHB);
  t_nPixSeed->push_back(nSeedHE);
  t_nPixCand->push_back(nCandHB);
  t_nPixCand->push_back(nCandHE);

  TimingTree_->Fill();
}
void IsoTrig::studyMipCut(edm::Handle<reco::TrackCollection> &trkCollection,
                          edm::Handle<reco::IsolatedPixelTrackCandidateCollection> &L2cands) {
  clearMipCutTreeVectors();
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "inside studymipcut";
  if (!trkCollection.isValid()) {
    edm::LogWarning("IsoTrack") << "trkCollection.isValid is false";
  } else {
    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
    std::vector<spr::propagatedTrackDirection> trkCaloDirections1;
    spr::propagateCALO(
        trkCollection, geo_, bField_, theTrackQuality_, trkCaloDirections1, ((verbosity_ / 100) % 10 > 2));
    if (verbosity_ % 10 > 0)
      edm::LogVerbatim("IsoTrack") << "Number of L2cands:" << L2cands->size() << " to be matched to something out of "
                                   << trkCaloDirections1.size() << " reco tracks";
    for (unsigned int i = 0; i < L2cands->size(); i++) {
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref =
          edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(L2cands, i);
      double enIn = candref->energyIn();
      h_EnIn->Fill(candref->energyIn());
      h_EnOut->Fill(candref->energyOut());
      math::XYZTLorentzVector v1(
          candref->track()->px(), candref->track()->py(), candref->track()->pz(), candref->track()->p());
      if (verbosity_ % 10 > 1)
        edm::LogVerbatim("IsoTrack") << "HLT Level Candidate eta/phi/pt/enIn:" << candref->track()->eta() << "/"
                                     << candref->track()->phi() << "/" << candref->track()->pt() << "/"
                                     << candref->energyIn();
      math::XYZTLorentzVector mindRvec;
      double mindR = 999.9, mindP1 = 999.9, eMipDR = 0.0;
      std::vector<bool> selFlags;
      unsigned int nTracks = 0;
      double conehmaxNearP = 0, hCone = 0;
      for (trkDetItr = trkCaloDirections1.begin(); trkDetItr != trkCaloDirections1.end(); trkDetItr++, nTracks++) {
        const reco::Track *pTrack = &(*(trkDetItr->trkItr));
        math::XYZTLorentzVector v2(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
        double dr = dR(v1, v2);
        double dp1 = std::abs(1. / v1.r() - 1. / v2.r());
        if (verbosity_ % 1000 > 0)
          edm::LogVerbatim("IsoTrack") << "This recotrack(eta/phi/pt) " << pTrack->eta() << "/" << pTrack->phi() << "/"
                                       << pTrack->pt() << " has dr/dp= " << dr << "/" << dp1;
        if (dr < mindR) {
          mindR = dr;
          mindP1 = dp1;
          mindRvec = v2;
          int nRH_eMipDR = 0, nNearTRKs = 0;
          eMipDR = spr::eCone_ecal(geo_,
                                   barrelRecHitsHandle_,
                                   endcapRecHitsHandle_,
                                   trkDetItr->pointHCAL,
                                   trkDetItr->pointECAL,
                                   a_mipR_,
                                   trkDetItr->directionECAL,
                                   nRH_eMipDR);
          bool selectTk = spr::goodTrack(pTrack, leadPV_, selectionParameters_, ((verbosity_ / 100) % 10 > 1));
          spr::trackSelectionParameters oneCutParameters = selectionParameters_;
          oneCutParameters.maxDxyPV = 10;
          oneCutParameters.maxDzPV = 100;
          oneCutParameters.maxInMiss = 2;
          oneCutParameters.maxOutMiss = 2;
          bool qltyFlag = spr::goodTrack(pTrack, leadPV_, oneCutParameters, ((verbosity_ / 100) % 10 > 1));
          oneCutParameters = selectionParameters_;
          oneCutParameters.maxDxyPV = 10;
          oneCutParameters.maxDzPV = 100;
          bool qltyMissFlag = spr::goodTrack(pTrack, leadPV_, oneCutParameters, ((verbosity_ / 100) % 10 > 1));
          oneCutParameters = selectionParameters_;
          oneCutParameters.maxInMiss = 2;
          oneCutParameters.maxOutMiss = 2;
          bool qltyPVFlag = spr::goodTrack(pTrack, leadPV_, oneCutParameters, ((verbosity_ / 100) % 10 > 1));
          conehmaxNearP = spr::chargeIsolationCone(
              nTracks, trkCaloDirections1, a_charIsoR_, nNearTRKs, ((verbosity_ / 100) % 10 > 1));
          double e1 = spr::eCone_ecal(geo_,
                                      barrelRecHitsHandle_,
                                      endcapRecHitsHandle_,
                                      trkDetItr->pointHCAL,
                                      trkDetItr->pointECAL,
                                      a_neutR1_,
                                      trkDetItr->directionECAL,
                                      nRH_eMipDR);
          double e2 = spr::eCone_ecal(geo_,
                                      barrelRecHitsHandle_,
                                      endcapRecHitsHandle_,
                                      trkDetItr->pointHCAL,
                                      trkDetItr->pointECAL,
                                      a_neutR2_,
                                      trkDetItr->directionECAL,
                                      nRH_eMipDR);
          double e_inCone = e2 - e1;
          bool chgIsolFlag = (conehmaxNearP < cutCharge_);
          bool mipFlag = (eMipDR < cutMip_);
          bool neuIsolFlag = (e_inCone < cutNeutral_);
          bool trkpropFlag = ((trkDetItr->okECAL) && (trkDetItr->okHCAL));
          selFlags.clear();
          selFlags.push_back(selectTk);
          selFlags.push_back(qltyFlag);
          selFlags.push_back(qltyMissFlag);
          selFlags.push_back(qltyPVFlag);
          selFlags.push_back(trkpropFlag);
          selFlags.push_back(chgIsolFlag);
          selFlags.push_back(neuIsolFlag);
          selFlags.push_back(mipFlag);
          double distFromHotCell = -99.0;
          int nRecHitsCone = -99, ietaHotCell = -99, iphiHotCell = -99;
          GlobalPoint gposHotCell(0., 0., 0.);
          std::vector<DetId> coneRecHitDetIds;
          hCone = spr::eCone_hcal(geo_,
                                  hbhe_,
                                  trkDetItr->pointHCAL,
                                  trkDetItr->pointECAL,
                                  a_coneR_,
                                  trkDetItr->directionHCAL,
                                  nRecHitsCone,
                                  coneRecHitDetIds,
                                  distFromHotCell,
                                  ietaHotCell,
                                  iphiHotCell,
                                  gposHotCell,
                                  -1);
        }
      }
      pushMipCutTreeVecs(v1, mindRvec, enIn, eMipDR, mindR, mindP1, selFlags, hCone);
      fillDifferences(6, v1, mindRvec, (verbosity_ % 10 > 0));
      if (mindR >= 0.05) {
        fillDifferences(8, v1, mindRvec, (verbosity_ % 10 > 0));
        h_MipEnNoMatch->Fill(candref->energyIn(), eMipDR);
      } else {
        fillDifferences(7, v1, mindRvec, (verbosity_ % 10 > 0));
        h_MipEnMatch->Fill(candref->energyIn(), eMipDR);
      }
    }
  }
  MipCutTree_->Fill();
}

void IsoTrig::studyTrigger(edm::Handle<reco::TrackCollection> &trkCollection,
                           std::vector<reco::TrackCollection::const_iterator> &goodTks) {
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "Inside StudyTrigger";
  //// Filling Pt, eta, phi of L1, L2 and L3 objects
  for (int j = 0; j < 3; j++) {
    for (unsigned int k = 0; k < vec_[j].size(); k++) {
      if (verbosity_ % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "vec[" << j << "][" << k << "] pt " << vec_[j][k].pt() << " eta "
                                     << vec_[j][k].eta() << " phi " << vec_[j][k].phi();
      fillHist(j, vec_[j][k]);
    }
  }

  double deta, dphi, dr;
  //// deta, dphi and dR for leading L1 object with L2 and L3 objects
  for (int lvl = 1; lvl < 3; lvl++) {
    for (unsigned int i = 0; i < vec_[lvl].size(); i++) {
      deta = dEta(vec_[0][0], vec_[lvl][i]);
      dphi = dPhi(vec_[0][0], vec_[lvl][i]);
      dr = dR(vec_[0][0], vec_[lvl][i]);
      if (verbosity_ % 10 > 1)
        edm::LogVerbatim("IsoTrack") << "lvl " << lvl << " i " << i << " deta " << deta << " dphi " << dphi << " dR "
                                     << dr;
      h_dEtaL1[lvl - 1]->Fill(deta);
      h_dPhiL1[lvl - 1]->Fill(dphi);
      h_dRL1[lvl - 1]->Fill(std::sqrt(dr));
    }
  }

  math::XYZTLorentzVector mindRvec;
  double mindR;
  for (unsigned int k = 0; k < vec_[2].size(); ++k) {
    //// Find min of deta/dphi/dR for each of L3 objects with L2 objects
    mindR = 999.9;
    if (verbosity_ % 10 > 1)
      edm::LogVerbatim("IsoTrack") << "L3obj: pt " << vec_[2][k].pt() << " eta " << vec_[2][k].eta() << " phi "
                                   << vec_[2][k].phi();
    for (unsigned int j = 0; j < vec_[1].size(); j++) {
      dr = dR(vec_[2][k], vec_[1][j]);
      if (dr < mindR) {
        mindR = dr;
        mindRvec = vec_[1][j];
      }
    }
    fillDifferences(0, vec_[2][k], mindRvec, (verbosity_ % 10 > 0));
    if (mindR < 0.03) {
      fillDifferences(1, vec_[2][k], mindRvec, (verbosity_ % 10 > 0));
      fillHist(6, mindRvec);
      fillHist(8, vec_[2][k]);
    } else {
      fillDifferences(2, vec_[2][k], mindRvec, (verbosity_ % 10 > 0));
      fillHist(7, mindRvec);
      fillHist(9, vec_[2][k]);
    }

    ////// Minimum deltaR for each of L3 objs with Reco::tracks
    mindR = 999.9;
    if (verbosity_ % 10 > 0)
      edm::LogVerbatim("IsoTrack") << "Now Matching L3 track with reco: L3 Track (eta, phi) " << vec_[2][k].eta() << ":"
                                   << vec_[2][k].phi() << " L2 Track " << mindRvec.eta() << ":" << mindRvec.phi()
                                   << " dR " << mindR;
    reco::TrackCollection::const_iterator goodTk = trkCollection->end();
    if (trkCollection.isValid()) {
      double mindP(9999.9);
      reco::TrackCollection::const_iterator trkItr;
      for (trkItr = trkCollection->begin(); trkItr != trkCollection->end(); trkItr++) {
        math::XYZTLorentzVector v4(trkItr->px(), trkItr->py(), trkItr->pz(), trkItr->p());
        double deltaR = dR(v4, vec_[2][k]);
        double dp = std::abs(v4.r() / vec_[2][k].r() - 1.0);
        if (deltaR < mindR) {
          mindR = deltaR;
          mindP = dp;
          mindRvec = v4;
          goodTk = trkItr;
        }
        if ((verbosity_ / 10) % 10 > 1 && deltaR < 1.0)
          edm::LogVerbatim("IsoTrack") << "reco track: pt " << v4.pt() << " eta " << v4.eta() << " phi " << v4.phi()
                                       << " DR " << deltaR;
      }
      if (verbosity_ % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "Now Matching at Reco level in step 1 DR: " << mindR << ":" << mindP
                                     << " eta:phi " << mindRvec.eta() << ":" << mindRvec.phi();
      if (mindR < 0.03 && mindP > 0.1) {
        for (trkItr = trkCollection->begin(); trkItr != trkCollection->end(); trkItr++) {
          math::XYZTLorentzVector v4(trkItr->px(), trkItr->py(), trkItr->pz(), trkItr->p());
          double deltaR = dR(v4, vec_[2][k]);
          double dp = std::abs(v4.r() / vec_[2][k].r() - 1.0);
          if (dp < mindP && deltaR < 0.03) {
            mindR = deltaR;
            mindP = dp;
            mindRvec = v4;
            goodTk = trkItr;
          }
        }
        if (verbosity_ % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "Now Matching at Reco level in step 2 DR: " << mindR << ":" << mindP
                                       << " eta:phi " << mindRvec.eta() << ":" << mindRvec.phi();
      }
      fillDifferences(3, vec_[2][k], mindRvec, (verbosity_ % 10 > 0));
      fillHist(3, mindRvec);
      if (mindR < 0.03) {
        fillDifferences(4, vec_[2][k], mindRvec, (verbosity_ % 10 > 0));
        fillHist(4, mindRvec);
      } else {
        fillDifferences(5, vec_[2][k], mindRvec, (verbosity_ % 10 > 0));
        fillHist(5, mindRvec);
      }
      if (goodTk != trkCollection->end())
        goodTks.push_back(goodTk);
    }
  }
}

void IsoTrig::studyIsolation(edm::Handle<reco::TrackCollection> &trkCollection,
                             std::vector<reco::TrackCollection::const_iterator> &goodTks) {
  if (trkCollection.isValid()) {
    std::vector<spr::propagatedTrackDirection> trkCaloDirections;
    spr::propagateCALO(
        trkCollection, geo_, bField_, theTrackQuality_, trkCaloDirections, ((verbosity_ / 100) % 10 > 2));

    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
    if ((verbosity_ / 1000) % 10 > 1) {
      edm::LogVerbatim("IsoTrack") << "n of barrelRecHitsHandle " << barrelRecHitsHandle_->size();
      for (EcalRecHitCollection::const_iterator hit = barrelRecHitsHandle_->begin(); hit != barrelRecHitsHandle_->end();
           ++hit) {
        edm::LogVerbatim("IsoTrack") << "hit : detid(ieta,iphi) " << (EBDetId)(hit->id()) << " time " << hit->time()
                                     << " energy " << hit->energy();
      }
      edm::LogVerbatim("IsoTrack") << "n of endcapRecHitsHandle " << endcapRecHitsHandle_->size();
      for (EcalRecHitCollection::const_iterator hit = endcapRecHitsHandle_->begin(); hit != endcapRecHitsHandle_->end();
           ++hit) {
        edm::LogVerbatim("IsoTrack") << "hit : detid(ieta,iphi) " << (EEDetId)(hit->id()) << " time " << hit->time()
                                     << " energy " << hit->energy();
      }
      edm::LogVerbatim("IsoTrack") << "n of hbhe " << hbhe_->size();
      for (HBHERecHitCollection::const_iterator hit = hbhe_->begin(); hit != hbhe_->end(); ++hit) {
        edm::LogVerbatim("IsoTrack") << "hit : detid(ieta,iphi) " << hit->id() << " time " << hit->time() << " energy "
                                     << hit->energy();
      }
    }
    unsigned int nTracks = 0, ngoodTk = 0, nselTk = 0;
    int ieta = 999;
    for (trkDetItr = trkCaloDirections.begin(); trkDetItr != trkCaloDirections.end(); trkDetItr++, nTracks++) {
      bool l3Track = (std::find(goodTks.begin(), goodTks.end(), trkDetItr->trkItr) != goodTks.end());
      const reco::Track *pTrack = &(*(trkDetItr->trkItr));
      math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
      bool selectTk = spr::goodTrack(pTrack, leadPV_, selectionParameters_, ((verbosity_ / 100) % 10 > 1));
      double eMipDR = 9999., e_inCone = 0, conehmaxNearP = 0, mindR = 999.9, hCone = 0;
      if (trkDetItr->okHCAL) {
        HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
        ieta = detId.ieta();
      }
      for (unsigned k = 0; k < vec_[0].size(); ++k) {
        double deltaR = dR(v4, vec_[0][k]);
        if (deltaR < mindR)
          mindR = deltaR;
      }
      if ((verbosity_ / 100) % 10 > 1)
        edm::LogVerbatim("IsoTrack") << "Track ECAL " << trkDetItr->okECAL << " HCAL " << trkDetItr->okHCAL << " Flag "
                                     << selectTk;
      if (selectTk && trkDetItr->okECAL && trkDetItr->okHCAL) {
        ngoodTk++;
        int nRH_eMipDR = 0, nNearTRKs = 0;
        double e1 = spr::eCone_ecal(geo_,
                                    barrelRecHitsHandle_,
                                    endcapRecHitsHandle_,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_neutR1_,
                                    trkDetItr->directionECAL,
                                    nRH_eMipDR);
        double e2 = spr::eCone_ecal(geo_,
                                    barrelRecHitsHandle_,
                                    endcapRecHitsHandle_,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_neutR2_,
                                    trkDetItr->directionECAL,
                                    nRH_eMipDR);
        eMipDR = spr::eCone_ecal(geo_,
                                 barrelRecHitsHandle_,
                                 endcapRecHitsHandle_,
                                 trkDetItr->pointHCAL,
                                 trkDetItr->pointECAL,
                                 a_mipR_,
                                 trkDetItr->directionECAL,
                                 nRH_eMipDR);
        conehmaxNearP =
            spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, ((verbosity_ / 100) % 10 > 1));
        e_inCone = e2 - e1;
        double distFromHotCell = -99.0;
        int nRecHitsCone = -99, ietaHotCell = -99, iphiHotCell = -99;
        GlobalPoint gposHotCell(0., 0., 0.);
        std::vector<DetId> coneRecHitDetIds;
        hCone = spr::eCone_hcal(geo_,
                                hbhe_,
                                trkDetItr->pointHCAL,
                                trkDetItr->pointECAL,
                                a_coneR_,
                                trkDetItr->directionHCAL,
                                nRecHitsCone,
                                coneRecHitDetIds,
                                distFromHotCell,
                                ietaHotCell,
                                iphiHotCell,
                                gposHotCell,
                                -1);
        if (eMipDR < 1.0)
          nselTk++;
      }
      if (l3Track) {
        fillHist(10, v4);
        if (selectTk) {
          fillHist(11, v4);
          fillCuts(0, eMipDR, conehmaxNearP, e_inCone, v4, ieta, (mindR > dr_L1_));
          if (conehmaxNearP < cutCharge_) {
            fillHist(12, v4);
            if (eMipDR < cutMip_) {
              fillHist(13, v4);
              fillEnergy(1, ieta, hCone, eMipDR, v4);
              fillEnergy(0, ieta, hCone, eMipDR, v4);
              if (e_inCone < cutNeutral_) {
                fillHist(14, v4);
                fillEnergy(4, ieta, hCone, eMipDR, v4);
                fillEnergy(3, ieta, hCone, eMipDR, v4);
              }
            }
          }
        }
      } else if (doL2L3_) {
        fillHist(15, v4);
        if (selectTk) {
          fillHist(16, v4);
          fillCuts(1, eMipDR, conehmaxNearP, e_inCone, v4, ieta, (mindR > dr_L1_));
          if (conehmaxNearP < cutCharge_) {
            fillHist(17, v4);
            if (eMipDR < cutMip_) {
              fillHist(18, v4);
              fillEnergy(2, ieta, hCone, eMipDR, v4);
              fillEnergy(0, ieta, hCone, eMipDR, v4);
              if (e_inCone < cutNeutral_) {
                fillHist(19, v4);
                fillEnergy(5, ieta, hCone, eMipDR, v4);
                fillEnergy(3, ieta, hCone, eMipDR, v4);
              }
            }
          }
        }
      }
    }
    //   edm::LogVerbatim("IsoTrack") << "Number of tracks selected offline " << nselTk;
  }
}

void IsoTrig::chgIsolation(double &etaTriggered,
                           double &phiTriggered,
                           edm::Handle<reco::TrackCollection> &trkCollection,
                           const edm::Event &theEvent) {
  clearChgIsolnTreeVectors();
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "Inside chgIsolation() with eta/phi Triggered: " << etaTriggered << "/"
                                 << phiTriggered;
  std::vector<double> maxP;

  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  std::vector<spr::propagatedTrackDirection> trkCaloDirections1;
  spr::propagateCALO(trkCollection, geo_, bField_, theTrackQuality_, trkCaloDirections1, ((verbosity_ / 100) % 10 > 2));
  if (verbosity_ % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "Propagated TrkCollection";
  for (unsigned int k = 0; k < pixelIsolationConeSizeAtEC_.size(); ++k)
    maxP.push_back(0);
  unsigned i = pixelTrackRefsHE_.size();
  std::vector<std::pair<unsigned int, std::pair<double, double>>> VecSeedsatEC;
  //loop to select isolated tracks
  for (unsigned iS = 0; iS < pixelTrackRefsHE_.size(); iS++) {
    if (pixelTrackRefsHE_[iS]->p() > minPTrackValue_) {
      bool vtxMatch = false;
      //associate to vertex (in Z)
      unsigned int ivSel = recVtxs_->size();
      double minDZ = 100;
      for (unsigned int iv = 0; iv < recVtxs_->size(); ++iv) {
        if (fabs(pixelTrackRefsHE_[iS]->dz((*recVtxs_)[iv].position())) < minDZ) {
          minDZ = fabs(pixelTrackRefsHE_[iS]->dz((*recVtxs_)[iv].position()));
          ivSel = iv;
        }
      }
      //cut on dYX:
      if (ivSel == recVtxs_->size()) {
        vtxMatch = true;
      } else if (fabs(pixelTrackRefsHE_[iS]->dxy((*recVtxs_)[ivSel].position())) < vtxCutSeed_) {
        vtxMatch = true;
      }
      //select tracks not matched to triggered L1 jet
      double R = deltaR(etaTriggered, phiTriggered, pixelTrackRefsHE_[iS]->eta(), pixelTrackRefsHE_[iS]->phi());
      if (R > tauUnbiasCone_ && vtxMatch) {
        //propagate seed track to ECAL surface:
        std::pair<double, double> seedCooAtEC;
        // in case vertex is found:
        if (minDZ != 100)
          seedCooAtEC = GetEtaPhiAtEcal(pixelTrackRefsHE_[iS]->eta(),
                                        pixelTrackRefsHE_[iS]->phi(),
                                        pixelTrackRefsHE_[iS]->pt(),
                                        pixelTrackRefsHE_[iS]->charge(),
                                        (*recVtxs_)[ivSel].z());
        //in case vertex is not found:
        else
          seedCooAtEC = GetEtaPhiAtEcal(pixelTrackRefsHE_[iS]->eta(),
                                        pixelTrackRefsHE_[iS]->phi(),
                                        pixelTrackRefsHE_[iS]->pt(),
                                        pixelTrackRefsHE_[iS]->charge(),
                                        0);
        VecSeedsatEC.push_back(std::make_pair(iS, seedCooAtEC));
      }
    }
  }
  for (unsigned int l = 0; l < VecSeedsatEC.size(); l++) {
    unsigned int iSeed = VecSeedsatEC[l].first;
    math::XYZTLorentzVector v1(pixelTrackRefsHE_[iSeed]->px(),
                               pixelTrackRefsHE_[iSeed]->py(),
                               pixelTrackRefsHE_[iSeed]->pz(),
                               pixelTrackRefsHE_[iSeed]->p());

    for (unsigned int j = 0; j < VecSeedsatEC.size(); j++) {
      unsigned int iSurr = VecSeedsatEC[j].first;
      if (iSeed != iSurr) {
        //define preliminary cone around seed track impact point from which tracks will be extrapolated:
        //	edm::Ref<reco::IsolatedPixelTrackCandidateCollection> cand2ref =
        //	  edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(L2cands, iSurr);
        if (deltaR(pixelTrackRefsHE_[iSeed]->eta(),
                   pixelTrackRefsHE_[iSeed]->phi(),
                   pixelTrackRefsHE_[iSurr]->eta(),
                   pixelTrackRefsHE_[iSurr]->phi()) < prelimCone_) {
          unsigned int ivSel = recVtxs_->size();
          double minDZ2 = 100;
          for (unsigned int iv = 0; iv < recVtxs_->size(); ++iv) {
            if (fabs(pixelTrackRefsHE_[iSurr]->dz((*recVtxs_)[iv].position())) < minDZ2) {
              minDZ2 = fabs(pixelTrackRefsHE_[iSurr]->dz((*recVtxs_)[iv].position()));
              ivSel = iv;
            }
          }
          //cut ot dXY:
          if (minDZ2 == 100 || fabs(pixelTrackRefsHE_[iSurr]->dxy((*recVtxs_)[ivSel].position())) < vtxCutIsol_) {
            //calculate distance at ECAL surface and update isolation:
            double dist = getDistInCM(VecSeedsatEC[i].second.first,
                                      VecSeedsatEC[i].second.second,
                                      VecSeedsatEC[j].second.first,
                                      VecSeedsatEC[j].second.second);
            for (unsigned int k = 0; k < pixelIsolationConeSizeAtEC_.size(); ++k) {
              if (dist < pixelIsolationConeSizeAtEC_[k]) {
                if (pixelTrackRefsHE_[iSurr]->p() > maxP[k])
                  maxP[k] = pixelTrackRefsHE_[iSurr]->p();
              }
            }
          }
        }
      }
    }

    double conehmaxNearP = -1;
    bool selectTk = false;
    double mindR = 999.9;
    int nTracks = 0;
    math::XYZTLorentzVector mindRvec;
    for (trkDetItr = trkCaloDirections1.begin(); trkDetItr != trkCaloDirections1.end(); trkDetItr++, nTracks++) {
      int nNearTRKs = 0;
      const reco::Track *pTrack = &(*(trkDetItr->trkItr));
      math::XYZTLorentzVector v2(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
      double dr = dR(v1, v2);
      if (dr < mindR) {
        selectTk = spr::goodTrack(pTrack, leadPV_, selectionParameters_, ((verbosity_ / 100) % 10 > 1));
        conehmaxNearP = spr::chargeIsolationCone(
            nTracks, trkCaloDirections1, a_charIsoR_, nNearTRKs, ((verbosity_ / 100) % 10 > 1));
        mindR = dr;
        mindRvec = v2;
      }
    }
    pushChgIsolnTreeVecs(v1, mindRvec, maxP, conehmaxNearP, selectTk);
  }
  ChgIsolnTree_->Fill();
}

void IsoTrig::getGoodTracks(const edm::Event &iEvent, edm::Handle<reco::TrackCollection> &trkCollection) {
  t_nGoodTk->clear();
  std::vector<int> nGood(4, 0);
  if (trkCollection.isValid()) {
    std::vector<spr::propagatedTrackDirection> trkCaloDirections;
    spr::propagateCALO(
        trkCollection, geo_, bField_, theTrackQuality_, trkCaloDirections, ((verbosity_ / 100) % 10 > 2));

    // get the trigger jet
    edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
    iEvent.getByToken(tok_l1cand_, l1trigobj);

    std::vector<edm::Ref<l1extra::L1JetParticleCollection>> l1tauobjref;
    l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
    std::vector<edm::Ref<l1extra::L1JetParticleCollection>> l1jetobjref;
    l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
    std::vector<edm::Ref<l1extra::L1JetParticleCollection>> l1forjetobjref;
    l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);

    double ptTriggered(-10), etaTriggered(-100), phiTriggered(-100);
    for (unsigned int p = 0; p < l1tauobjref.size(); p++) {
      if (l1tauobjref[p]->pt() > ptTriggered) {
        ptTriggered = l1tauobjref[p]->pt();
        phiTriggered = l1tauobjref[p]->phi();
        etaTriggered = l1tauobjref[p]->eta();
      }
    }
    for (unsigned int p = 0; p < l1jetobjref.size(); p++) {
      if (l1jetobjref[p]->pt() > ptTriggered) {
        ptTriggered = l1jetobjref[p]->pt();
        phiTriggered = l1jetobjref[p]->phi();
        etaTriggered = l1jetobjref[p]->eta();
      }
    }
    for (unsigned int p = 0; p < l1forjetobjref.size(); p++) {
      if (l1forjetobjref[p]->pt() > ptTriggered) {
        ptTriggered = l1forjetobjref[p]->pt();
        phiTriggered = l1forjetobjref[p]->phi();
        etaTriggered = l1forjetobjref[p]->eta();
      }
    }
    double pTriggered = ptTriggered * cosh(etaTriggered);
    math::XYZTLorentzVector pTrigger(
        ptTriggered * cos(phiTriggered), ptTriggered * sin(phiTriggered), pTriggered * tanh(etaTriggered), pTriggered);

    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
    unsigned int nTracks(0);
    for (trkDetItr = trkCaloDirections.begin(); trkDetItr != trkCaloDirections.end(); trkDetItr++, nTracks++) {
      const reco::Track *pTrack = &(*(trkDetItr->trkItr));
      math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
      bool selectTk = spr::goodTrack(pTrack, leadPV_, selectionParameters_, ((verbosity_ / 100) % 10 > 1));
      double mindR = dR(v4, pTrigger);
      if ((verbosity_ / 100) % 10 > 1)
        edm::LogVerbatim("IsoTrack") << "Track ECAL " << trkDetItr->okECAL << " HCAL " << trkDetItr->okHCAL << " Flag "
                                     << selectTk;
      if (selectTk && trkDetItr->okECAL && trkDetItr->okHCAL && mindR > 1.0) {
        int nRH_eMipDR(0), nNearTRKs(0);
        double eMipDR = spr::eCone_ecal(geo_,
                                        barrelRecHitsHandle_,
                                        endcapRecHitsHandle_,
                                        trkDetItr->pointHCAL,
                                        trkDetItr->pointECAL,
                                        a_mipR_,
                                        trkDetItr->directionECAL,
                                        nRH_eMipDR);
        double conehmaxNearP =
            spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, ((verbosity_ / 100) % 10 > 1));
        if (conehmaxNearP < 2.0 && eMipDR < 1.0) {
          if (pTrack->p() >= 20 && pTrack->p() < 30) {
            ++nGood[0];
          } else if (pTrack->p() >= 30 && pTrack->p() < 40) {
            ++nGood[1];
          } else if (pTrack->p() >= 40 && pTrack->p() < 60) {
            ++nGood[2];
          } else if (pTrack->p() >= 60 && pTrack->p() < 100) {
            ++nGood[3];
          }
        }
      }
    }
  }

  for (unsigned int ii = 0; ii < nGood.size(); ++ii)
    t_nGoodTk->push_back(nGood[ii]);
}

void IsoTrig::fillHist(int indx, math::XYZTLorentzVector &vec) {
  h_p[indx]->Fill(vec.r());
  h_pt[indx]->Fill(vec.pt());
  h_eta[indx]->Fill(vec.eta());
  h_phi[indx]->Fill(vec.phi());
}

void IsoTrig::fillDifferences(int indx, math::XYZTLorentzVector &vec1, math::XYZTLorentzVector &vec2, bool debug) {
  double dr = dR(vec1, vec2);
  double deta = dEta(vec1, vec2);
  double dphi = dPhi(vec1, vec2);
  double dpt = dPt(vec1, vec2);
  double dp = dP(vec1, vec2);
  double dinvpt = dinvPt(vec1, vec2);
  h_dEta[indx]->Fill(deta);
  h_dPhi[indx]->Fill(dphi);
  h_dPt[indx]->Fill(dpt);
  h_dP[indx]->Fill(dp);
  h_dinvPt[indx]->Fill(dinvpt);
  h_mindR[indx]->Fill(dr);
  if (debug)
    edm::LogVerbatim("IsoTrack") << "mindR for index " << indx << " is " << dr << " deta " << deta << " dphi " << dphi
                                 << " dpt " << dpt << " dinvpt " << dinvpt;
}

void IsoTrig::fillCuts(
    int indx, double eMipDR, double conehmaxNearP, double e_inCone, math::XYZTLorentzVector &vec, int ieta, bool cut) {
  h_eMip[indx]->Fill(eMipDR);
  h_eMaxNearP[indx]->Fill(conehmaxNearP);
  h_eNeutIso[indx]->Fill(e_inCone);
  if ((conehmaxNearP < cutCharge_) && (eMipDR < cutMip_)) {
    for (int lim = 0; lim < 5; ++lim) {
      if ((vec.r() > pLimits_[lim]) && (vec.r() <= pLimits_[lim + 1])) {
        h_etaMipTracks[lim][indx][0]->Fill((double)(ieta));
        if (cut)
          h_etaMipTracks[lim][indx][1]->Fill((double)(ieta));
        if (e_inCone < cutNeutral_) {
          h_etaCalibTracks[lim][indx][0]->Fill((double)(ieta));
          if (cut)
            h_etaCalibTracks[lim][indx][1]->Fill((double)(ieta));
        }
      }
    }
  }
}

void IsoTrig::fillEnergy(int indx, int ieta, double hCone, double eMipDR, math::XYZTLorentzVector &vec) {
  int kk(-1);
  if (ieta > 0 && ieta < 25)
    kk = 23 + ieta;
  else if (ieta > -25 && ieta < 0)
    kk = -(ieta + 1);
  if (kk >= 0 && eMipDR > 0.01 && hCone > 1.0) {
    for (int lim = 0; lim < 5; ++lim) {
      if ((vec.r() > pLimits_[lim]) && (vec.r() <= pLimits_[lim + 1])) {
        h_eHcal[lim][indx][kk]->Fill(hCone);
        h_eCalo[lim][indx][kk]->Fill(hCone + eMipDR);
      }
    }
  }
}

double IsoTrig::dEta(math::XYZTLorentzVector &vec1, math::XYZTLorentzVector &vec2) { return (vec1.eta() - vec2.eta()); }

double IsoTrig::dPhi(math::XYZTLorentzVector &vec1, math::XYZTLorentzVector &vec2) {
  double phi1 = vec1.phi();
  if (phi1 < 0)
    phi1 += 2.0 * M_PI;
  double phi2 = vec2.phi();
  if (phi2 < 0)
    phi2 += 2.0 * M_PI;
  double dphi = phi1 - phi2;
  if (dphi > M_PI)
    dphi -= 2. * M_PI;
  else if (dphi < -M_PI)
    dphi += 2. * M_PI;
  return dphi;
}

double IsoTrig::dR(math::XYZTLorentzVector &vec1, math::XYZTLorentzVector &vec2) {
  double deta = dEta(vec1, vec2);
  double dphi = dPhi(vec1, vec2);
  return std::sqrt(deta * deta + dphi * dphi);
}

double IsoTrig::dPt(math::XYZTLorentzVector &vec1, math::XYZTLorentzVector &vec2) { return (vec1.pt() - vec2.pt()); }

double IsoTrig::dP(math::XYZTLorentzVector &vec1, math::XYZTLorentzVector &vec2) {
  return (std::abs(vec1.r() - vec2.r()));
}

double IsoTrig::dinvPt(math::XYZTLorentzVector &vec1, math::XYZTLorentzVector &vec2) {
  return ((1 / vec1.pt()) - (1 / vec2.pt()));
}

std::pair<double, double> IsoTrig::etaPhiTrigger() {
  double eta(0), phi(0), ptmax(0);
  for (unsigned int k = 0; k < vec_[0].size(); ++k) {
    if (k == 0) {
      eta = vec_[0][k].eta();
      phi = vec_[0][k].phi();
      ptmax = vec_[0][k].pt();
    } else if (vec_[0][k].pt() > ptmax) {
      eta = vec_[0][k].eta();
      phi = vec_[0][k].phi();
      ptmax = vec_[0][k].pt();
    }
  }
  return std::pair<double, double>(eta, phi);
}

std::pair<double, double> IsoTrig::GetEtaPhiAtEcal(double etaIP, double phiIP, double pT, int charge, double vtxZ) {
  double deltaPhi = 0;
  double etaEC = 100;
  double phiEC = 100;

  double Rcurv = 9999999;
  if (bfVal_ != 0)
    Rcurv = pT * 33.3 * 100 / (bfVal_ * 10);  //r(m)=pT(GeV)*33.3/B(kG)

  double ecDist = zEE_;
  double ecRad = rEB_;  //radius of ECAL barrel (cm)
  double theta = 2 * atan(exp(-etaIP));
  double zNew = 0;
  if (theta > 0.5 * M_PI)
    theta = M_PI - theta;
  if (fabs(etaIP) < 1.479) {
    if ((0.5 * ecRad / Rcurv) > 1) {
      etaEC = 10000;
      deltaPhi = 0;
    } else {
      deltaPhi = -charge * asin(0.5 * ecRad / Rcurv);
      double alpha1 = 2 * asin(0.5 * ecRad / Rcurv);
      double z = ecRad / tan(theta);
      if (etaIP > 0)
        zNew = z * (Rcurv * alpha1) / ecRad + vtxZ;  //new z-coordinate of track
      else
        zNew = -z * (Rcurv * alpha1) / ecRad + vtxZ;  //new z-coordinate of track
      double zAbs = fabs(zNew);
      if (zAbs < ecDist) {
        etaEC = -log(tan(0.5 * atan(ecRad / zAbs)));
        deltaPhi = -charge * asin(0.5 * ecRad / Rcurv);
      }
      if (zAbs > ecDist) {
        zAbs = (fabs(etaIP) / etaIP) * ecDist;
        double Zflight = fabs(zAbs - vtxZ);
        double alpha = (Zflight * ecRad) / (z * Rcurv);
        double Rec = 2 * Rcurv * sin(alpha / 2);
        deltaPhi = -charge * alpha / 2;
        etaEC = -log(tan(0.5 * atan(Rec / ecDist)));
      }
    }
  } else {
    zNew = (fabs(etaIP) / etaIP) * ecDist;
    double Zflight = fabs(zNew - vtxZ);
    double Rvirt = fabs(Zflight * tan(theta));
    double Rec = 2 * Rcurv * sin(Rvirt / (2 * Rcurv));
    deltaPhi = -(charge) * (Rvirt / (2 * Rcurv));
    etaEC = -log(tan(0.5 * atan(Rec / ecDist)));
  }

  if (zNew < 0)
    etaEC = -etaEC;
  phiEC = phiIP + deltaPhi;

  if (phiEC < -M_PI)
    phiEC += 2 * M_PI;
  if (phiEC > M_PI)
    phiEC -= 2 * M_PI;

  std::pair<double, double> retVal(etaEC, phiEC);
  return retVal;
}

double IsoTrig::getDistInCM(double eta1, double phi1, double eta2, double phi2) {
  double Rec;
  double theta1 = 2 * atan(exp(-eta1));
  double theta2 = 2 * atan(exp(-eta2));
  if (fabs(eta1) < 1.479)
    Rec = rEB_;  //radius of ECAL barrel
  else if (fabs(eta1) > 1.479 && fabs(eta1) < 7.0)
    Rec = tan(theta1) * zEE_;  //distance from IP to ECAL endcap
  else
    return 1000;

  //|vect| times tg of acos(scalar product)
  double angle =
      acos((sin(theta1) * sin(theta2) * (sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2)) + cos(theta1) * cos(theta2)));
  if (angle < 0.5 * M_PI)
    return fabs((Rec / sin(theta1)) * tan(angle));
  else
    return 1000;
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrig);
