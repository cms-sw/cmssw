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
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
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

class IsoTrig : public edm::EDAnalyzer {

public:
  explicit IsoTrig(const edm::ParameterSet&);
  ~IsoTrig();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  void clearMipCutTreeVectors();
  void clearChgIsolnTreeVectors();
  void pushChgIsolnTreeVecs(math::XYZTLorentzVector &Pixcand, 
			    math::XYZTLorentzVector &Trkcand, 
			    std::vector<double> &PixMaxP, double &TrkMaxP, bool &selTk);
  void pushMipCutTreeVecs(math::XYZTLorentzVector &NFcand,
			  math::XYZTLorentzVector &Trkcand,
			  double &EmipNFcand, double &EmipTrkcand, 
			  double &mindR, double &mindP1,
			  std::vector<bool> &Flags, double hCone);
  void StudyTrkEbyP(edm::Handle<reco::TrackCollection>& trkCollection);
  void studyTiming(const edm::Event& theEvent);
  void studyMipCut(edm::Handle<reco::TrackCollection>& trkCollection,
		   edm::Handle<reco::IsolatedPixelTrackCandidateCollection>& L2cands);
  void studyTrigger(edm::Handle<reco::TrackCollection>&,
		    std::vector<reco::TrackCollection::const_iterator>&);
  void studyIsolation(edm::Handle<reco::TrackCollection>&,
		      std::vector<reco::TrackCollection::const_iterator>&);
  void chgIsolation(double& etaTriggered, double& phiTriggered,
		    edm::Handle<reco::TrackCollection>& trkCollection, 
		    const edm::Event& theEvent);
  void getGoodTracks(const edm::Event&, edm::Handle<reco::TrackCollection>&);
  void fillHist(int, math::XYZTLorentzVector&);
  void fillDifferences(int, math::XYZTLorentzVector&, math::XYZTLorentzVector&, bool);
  void fillCuts(int, double, double, double, math::XYZTLorentzVector&, int, bool);
  void fillEnergy(int, int, double, double, math::XYZTLorentzVector&);
  double dEta(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dPhi(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dP(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dinvPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  std::pair<double,double> etaPhiTrigger();
  std::pair<double,double> GetEtaPhiAtEcal(double etaIP, double phiIP, 
					   double pT, int charge, double vtxZ);
  double getDistInCM(double eta1,double phi1, double eta2,double phi2);

  // ----------member data ---------------------------
  HLTPrescaleProvider hltPrescaleProvider_;
  std::vector<std::string>   trigNames;
  edm::InputTag              PixcandTag_, L1candTag_, L2candTag_;
  std::vector<edm::InputTag> pixelTracksSources_;
  bool                       doL2L3, doTiming, doMipCutTree;
  bool                       doTrkResTree, doChgIsolTree, doStudyIsol;
  int                        verbosity;
  double                     rEB_, zEE_, bfVal;
  std::vector<double>        pixelIsolationConeSizeAtEC_;
  double                     minPTrackValue_, vtxCutSeed_, vtxCutIsol_;
  double                     tauUnbiasCone_, prelimCone_;
  spr::trackSelectionParameters selectionParameters;
  std::string                theTrackQuality, processName;
  double                     dr_L1, a_mipR, a_coneR, a_charIsoR, a_neutIsoR;
  double                     a_neutR1, a_neutR2, cutMip, cutCharge, cutNeutral;
  int                        minRunNo, maxRunNo;
  double                     pLimits[6];
  edm::EDGetTokenT<LumiDetails>            tok_lumi;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_hlt_;
  edm::EDGetTokenT<reco::TrackCollection>  tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>         tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>   tok_hbhe_;
  edm::EDGetTokenT<reco::VertexCollection> tok_verthb_, tok_verthe_;
  edm::EDGetTokenT<SeedingLayerSetsHits> tok_SeedingLayerhb, tok_SeedingLayerhe;
  edm::EDGetTokenT<SiPixelRecHitCollection> tok_SiPixelRecHits;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> tok_pixtk_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>        tok_l1cand_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> tok_l2cand_;
  std::vector<edm::EDGetTokenT<reco::TrackCollection> >         tok_pixtks_;

  std::vector<reco::TrackRef> pixelTrackRefsHB, pixelTrackRefsHE;
  edm::ESHandle<MagneticField> bFieldH;  
  edm::ESHandle<CaloGeometry> pG;
  edm::Handle<HBHERecHitCollection> hbhe;
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  edm::Handle<reco::BeamSpot> beamSpotH; 
  edm::Handle<reco::VertexCollection> recVtxs;
  math::XYZPoint leadPV;

  std::map<unsigned int, unsigned int> TrigList;
  std::map<unsigned int, const std::pair<int, int>> TrigPreList;
  bool                 changed;
  edm::Service<TFileService> fs;
  TTree               *MipCutTree, *ChgIsolnTree, *TrkResTree, *TimingTree;
  std::vector<double> *t_timeL2Prod;
  std::vector<int>    *t_nPixCand;
  std::vector<int>    *t_nPixSeed;
  std::vector<int>    *t_nGoodTk;

  std::vector<double> *t_TrkhCone;
  std::vector<double> *t_TrkP;
  std::vector<bool>   *t_TrkselTkFlag;
  std::vector<bool>   *t_TrkqltyFlag;
  std::vector<bool>   *t_TrkMissFlag;
  std::vector<bool>   *t_TrkPVFlag;
  std::vector<bool>   *t_TrkNuIsolFlag;

  std::vector<double> *t_PixcandP;
  std::vector<double> *t_PixcandPt;
  std::vector<double> *t_PixcandEta;
  std::vector<double> *t_PixcandPhi;
  std::vector<std::vector<double> > *t_PixcandMaxP;
  std::vector<double> *t_PixTrkcandP;
  std::vector<double> *t_PixTrkcandPt;
  std::vector<double> *t_PixTrkcandEta;
  std::vector<double> *t_PixTrkcandPhi;
  std::vector<double> *t_PixTrkcandMaxP;
  std::vector<bool>   *t_PixTrkcandselTk;

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
  std::vector<bool>   *t_NFTrkselTkFlag;
  std::vector<bool>   *t_NFTrkqltyFlag;
  std::vector<bool>   *t_NFTrkMissFlag;
  std::vector<bool>   *t_NFTrkPVFlag;
  std::vector<bool>   *t_NFTrkPropFlag;
  std::vector<bool>   *t_NFTrkChgIsoFlag;
  std::vector<bool>   *t_NFTrkNeuIsoFlag;
  std::vector<bool>   *t_NFTrkMipFlag;
  std::vector<double> *t_ECone;

  TH1D                      *h_EnIn, *h_EnOut;
  TH2D                      *h_MipEnMatch, *h_MipEnNoMatch;
  TH1I                      *h_nHLT, *h_HLT, *h_PreL1, *h_PreHLT; 
  TH1I                      *h_Pre, *h_nL3Objs, *h_Filters;
  TH1D                      *h_PreL1wt, *h_PreHLTwt, *h_L1ObjEnergy;
  TH1D                      *h_p[20], *h_pt[20], *h_eta[20], *h_phi[20];
  TH1D                      *h_dEtaL1[2], *h_dPhiL1[2], *h_dRL1[2];
  TH1D                      *h_dEta[9], *h_dPhi[9], *h_dPt[9], *h_dP[9];
  TH1D                      *h_dinvPt[9], *h_mindR[9], *h_eMip[2];
  TH1D                      *h_eMaxNearP[2], *h_eNeutIso[2];
  TH1D                      *h_etaCalibTracks[5][2][2],*h_etaMipTracks[5][2][2];
  TH1D                      *h_eHcal[5][6][48], *h_eCalo[5][6][48];
  TH1I                      *g_Pre, *g_PreL1, *g_PreHLT, *g_Accepts;
  std::vector<math::XYZTLorentzVector> vec[3];

};
