// system include files
#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TInterpreter.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
//Generator information
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

//#define EDM_ML_DEBUG

class HcalIsoTrackStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HcalIsoTrackStudy(edm::ParameterSet const&);
  ~HcalIsoTrackStudy() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  std::array<int, 3> fillTree(std::vector<math::XYZTLorentzVector>& vecL1,
                              std::vector<math::XYZTLorentzVector>& vecL3,
                              math::XYZPoint& leadPV,
                              std::vector<spr::propagatedTrackDirection>& trkCaloDirections,
                              std::vector<spr::propagatedTrackID>& trkCaloDets,
                              const CaloGeometry* geo,
                              const CaloTopology* topo,
                              const HcalTopology* theHBHETopology,
                              const EcalChannelStatus* theEcalChStatus,
                              const EcalSeverityLevelAlgo* theEcalSevlv,
                              edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle,
                              edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle,
                              edm::Handle<HBHERecHitCollection>& hbhe,
                              edm::Handle<CaloTowerCollection>& towerHandle,
                              edm::Handle<reco::GenParticleCollection>& genParticles,
                              const HcalRespCorrs* respCorrs);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double trackP(const reco::Track*, const edm::Handle<reco::GenParticleCollection>&);
  double trackE(const reco::Track*, const edm::Handle<reco::GenParticleCollection>&);
  double rhoh(const edm::Handle<CaloTowerCollection>&);
  void storeEnergy(int indx,
                   const HcalRespCorrs* respCorrs,
                   const std::vector<DetId>& ids,
                   std::vector<double>& edet,
                   double& eHcal,
                   std::vector<unsigned int>* detIds,
                   std::vector<double>* hitEnergies);
  void fillECALmatrix(const DetId& detId,
                      edm::Handle<EcalRecHitCollection>& hitsEB,
                      edm::Handle<EcalRecHitCollection>& hitsEE,
                      const CaloGeometry* geo,
                      const CaloTopology* caloTopology,
                      const reco::Track* pTrack,
                      int ieta,
                      int iphi);
  void fillHCALmatrix(const HcalTopology* topology,
                      const DetId& detId,
                      edm::Handle<HBHERecHitCollection>& hbhe,
                      const CaloGeometry* geo,
                      const reco::Track* pTrack,
                      int ieta,
                      int iphi);
  void TrackMap(unsigned int trkIndex, std::vector<spr::propagatedTrackDirection>& trkDirs, double dR);

  l1t::L1TGlobalUtil* l1GtUtils_;
  edm::Service<TFileService> fs;
  HLTConfigProvider hltConfig_;
  const std::vector<std::string> trigNames_;
  spr::trackSelectionParameters selectionParameter_;
  const std::string theTrackQuality_;
  const std::string processName_, l1Filter_;
  const std::string l2Filter_, l3Filter_;
  const double a_coneR_, a_mipR_, pTrackMin_, eEcalMax_;
  const double maxRestrictionP_, slopeRestrictionP_;
  const double hcalScale_, eIsolate1_, eIsolate2_;
  const double pTrackLow_, pTrackHigh_;
  const int prescaleLow_, prescaleHigh_;
  const int useRaw_, dataType_;
  const bool ignoreTrigger_, useL1Trigger_;
  const bool unCorrect_, collapseDepth_;
  const double hitEthrEB_, hitEthrEE0_, hitEthrEE1_;
  const double hitEthrEE2_, hitEthrEE3_;
  const double hitEthrEELo_, hitEthrEEHi_;
  const edm::InputTag triggerEvent_, theTriggerResultsLabel_;
  const std::string labelGenTrack_, labelRecVtx_, labelEB_;
  const std::string labelEE_, labelHBHE_, labelTower_, l1TrigName_;
  const int matrixECAL_, matrixHCAL_;
  const double mapR_;
  const bool get2Ddist_;
  unsigned int nRun_, nLow_, nHigh_;
  double a_charIsoR_, a_coneR1_, a_coneR2_;
  const HcalDDDRecConstants* hdc_;
  std::vector<double> etabins_, phibins_;
  double etadist_, phidist_, etahalfdist_, phihalfdist_;
  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_parts_;
  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<CaloTowerCollection> tok_cala_;
  edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;
  edm::EDGetTokenT<BXVector<GlobalAlgBlk>> tok_alg_;

  TTree *tree, *tree2;
  unsigned int t_RunNo, t_EventNo;
  int t_Run, t_Event, t_DataType, t_ieta, t_iphi;
  int t_goodPV, t_nVtx, t_nTrk;
  double t_EventWeight, t_p, t_pt, t_phi;
  std::vector<double>*t_mapP, *t_mapPt, *t_mapEta, *t_mapPhi;
  double t_l1pt, t_l1eta, t_l1phi;
  double t_l3pt, t_l3eta, t_l3phi;
  double t_mindR1, t_mindR2;
  double t_eMipDR, t_hmaxNearP, t_gentrackP, t_gentrackE;
  double t_emaxNearP, t_eAnnular, t_hAnnular;
  double t_eHcal, t_eHcal10, t_eHcal30, t_rhoh;
  bool t_selectTk, t_qltyFlag, t_qltyMissFlag;
  bool t_qltyPVFlag, t_TrigPass, t_TrigPassSel;
  std::vector<unsigned int>*t_DetIds, *t_DetIds1, *t_DetIds3;
  std::vector<double>*t_HitEnergies, *t_HitEnergies1, *t_HitEnergies3;
  std::vector<bool>*t_trgbits, *t_hltbits;
  bool t_L1Bit;
  int t_Tracks, t_TracksProp, t_TracksSaved;
  int t_TracksLoose, t_TracksTight, t_allvertex;
  std::vector<int>*t_ietaAll, *t_ietaGood, *t_trackType;
  std::vector<unsigned int>*t_DetIdEC, *t_DetIdHC;
  std::vector<double>*t_HitEnergyEC, *t_HitDistEC;
  std::vector<double>*t_HitEnergyHC, *t_HitDistHC;
};

HcalIsoTrackStudy::HcalIsoTrackStudy(const edm::ParameterSet& iConfig)
    : trigNames_(iConfig.getParameter<std::vector<std::string>>("triggers")),
      theTrackQuality_(iConfig.getParameter<std::string>("trackQuality")),
      processName_(iConfig.getParameter<std::string>("processName")),
      l1Filter_(iConfig.getParameter<std::string>("l1Filter")),
      l2Filter_(iConfig.getParameter<std::string>("l2Filter")),
      l3Filter_(iConfig.getParameter<std::string>("l3Filter")),
      a_coneR_(iConfig.getParameter<double>("coneRadius")),
      a_mipR_(iConfig.getParameter<double>("coneRadiusMIP")),
      pTrackMin_(iConfig.getParameter<double>("minimumTrackP")),
      eEcalMax_(iConfig.getParameter<double>("maximumEcalEnergy")),
      maxRestrictionP_(iConfig.getParameter<double>("maxTrackP")),
      slopeRestrictionP_(iConfig.getParameter<double>("slopeTrackP")),
      hcalScale_(iConfig.getUntrackedParameter<double>("hHcalScale", 1.0)),
      eIsolate1_(iConfig.getParameter<double>("isolationEnergyTight")),
      eIsolate2_(iConfig.getParameter<double>("isolationEnergyLoose")),
      pTrackLow_(iConfig.getParameter<double>("momentumLow")),
      pTrackHigh_(iConfig.getParameter<double>("momentumHigh")),
      prescaleLow_(iConfig.getParameter<int>("prescaleLow")),
      prescaleHigh_(iConfig.getParameter<int>("prescaleHigh")),
      useRaw_(iConfig.getUntrackedParameter<int>("useRaw", 0)),
      dataType_(iConfig.getUntrackedParameter<int>("dataType", 0)),
      ignoreTrigger_(iConfig.getUntrackedParameter<bool>("ignoreTriggers", false)),
      useL1Trigger_(iConfig.getUntrackedParameter<bool>("useL1Trigger", false)),
      unCorrect_(iConfig.getUntrackedParameter<bool>("unCorrect", false)),
      collapseDepth_(iConfig.getUntrackedParameter<bool>("collapseDepth", false)),
      hitEthrEB_(iConfig.getParameter<double>("EBHitEnergyThreshold")),
      hitEthrEE0_(iConfig.getParameter<double>("EEHitEnergyThreshold0")),
      hitEthrEE1_(iConfig.getParameter<double>("EEHitEnergyThreshold1")),
      hitEthrEE2_(iConfig.getParameter<double>("EEHitEnergyThreshold2")),
      hitEthrEE3_(iConfig.getParameter<double>("EEHitEnergyThreshold3")),
      hitEthrEELo_(iConfig.getParameter<double>("EEHitEnergyThresholdLow")),
      hitEthrEEHi_(iConfig.getParameter<double>("EEHitEnergyThresholdHigh")),
      triggerEvent_(iConfig.getParameter<edm::InputTag>("labelTriggerEvent")),
      theTriggerResultsLabel_(iConfig.getParameter<edm::InputTag>("labelTriggerResult")),
      labelGenTrack_(iConfig.getParameter<std::string>("labelTrack")),
      labelRecVtx_(iConfig.getParameter<std::string>("labelVertex")),
      labelEB_(iConfig.getParameter<std::string>("labelEBRecHit")),
      labelEE_(iConfig.getParameter<std::string>("labelEERecHit")),
      labelHBHE_(iConfig.getParameter<std::string>("labelHBHERecHit")),
      labelTower_(iConfig.getParameter<std::string>("labelCaloTower")),
      l1TrigName_(iConfig.getUntrackedParameter<std::string>("l1TrigName", "L1_SingleJet60")),
      matrixECAL_(iConfig.getUntrackedParameter<int>("matrixECAL", 5)),
      matrixHCAL_(iConfig.getUntrackedParameter<int>("matrixHCAL", 3)),
      mapR_(iConfig.getUntrackedParameter<double>("mapRadius", 34.98)),
      get2Ddist_(iConfig.getUntrackedParameter<bool>("get2Ddist", false)),
      nRun_(0),
      nLow_(0),
      nHigh_(0),
      hdc_(nullptr) {
  usesResource(TFileService::kSharedResource);

  //now do whatever initialization is needed
  const double isolationRadius(28.9), innerR(10.0), outerR(30.0);
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameter_.minPt = iConfig.getParameter<double>("minTrackPt");
  ;
  selectionParameter_.minQuality = trackQuality_;
  selectionParameter_.maxDxyPV = iConfig.getParameter<double>("maxDxyPV");
  selectionParameter_.maxDzPV = iConfig.getParameter<double>("maxDzPV");
  selectionParameter_.maxChi2 = iConfig.getParameter<double>("maxChi2");
  selectionParameter_.maxDpOverP = iConfig.getParameter<double>("maxDpOverP");
  selectionParameter_.minOuterHit = iConfig.getParameter<int>("minOuterHit");
  selectionParameter_.minLayerCrossed = iConfig.getParameter<int>("minLayerCrossed");
  selectionParameter_.maxInMiss = iConfig.getParameter<int>("maxInMiss");
  selectionParameter_.maxOutMiss = iConfig.getParameter<int>("maxOutMiss");
  a_charIsoR_ = a_coneR_ + isolationRadius;
  a_coneR1_ = a_coneR_ + innerR;
  a_coneR2_ = a_coneR_ + outerR;
  // Different isolation cuts are described in DN-2016/029
  // Tight cut uses 2 GeV; Loose cut uses 10 GeV
  // Eta dependent cut uses (maxRestrictionP_ * exp(|ieta|*log(2.5)/18))
  // with the factor for exponential slopeRestrictionP_ = log(2.5)/18
  // maxRestrictionP_ = 8 GeV as came from a study
  std::string labelBS = iConfig.getParameter<std::string>("labelBeamSpot");
  std::string modnam = iConfig.getUntrackedParameter<std::string>("moduleName", "");
  std::string prdnam = iConfig.getUntrackedParameter<std::string>("producerName", "");
  edm::InputTag algTag = iConfig.getParameter<edm::InputTag>("algInputTag");
  edm::InputTag extTag = iConfig.getParameter<edm::InputTag>("extInputTag");
  l1GtUtils_ = new l1t::L1TGlobalUtil(iConfig, consumesCollector(), *this, algTag, extTag, l1t::UseEventSetupIn::Event);
  // define tokens for access
  tok_trigEvt_ = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes_ = consumes<edm::TriggerResults>(theTriggerResultsLabel_);
  tok_bs_ = consumes<reco::BeamSpot>(labelBS);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_ew_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  tok_parts_ = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));
  tok_cala_ = consumes<CaloTowerCollection>(labelTower_);
  tok_alg_ = consumes<BXVector<GlobalAlgBlk>>(algTag);

  if (modnam.empty()) {
    tok_recVtx_ = consumes<reco::VertexCollection>(labelRecVtx_);
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", labelEB_));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", labelEE_));
    tok_hbhe_ = consumes<HBHERecHitCollection>(labelHBHE_);
    edm::LogVerbatim("HcalIsoTrack") << "Labels used " << triggerEvent_ << " " << theTriggerResultsLabel_ << " "
                                     << labelBS << " " << labelRecVtx_ << " " << labelGenTrack_ << " "
                                     << edm::InputTag("ecalRecHit", labelEB_) << " "
                                     << edm::InputTag("ecalRecHit", labelEE_) << " " << labelHBHE_ << " "
                                     << labelTower_;
  } else {
    tok_recVtx_ = consumes<reco::VertexCollection>(edm::InputTag(modnam, labelRecVtx_, prdnam));
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag(modnam, labelEB_, prdnam));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag(modnam, labelEE_, prdnam));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag(modnam, labelHBHE_, prdnam));
    edm::LogVerbatim("HcalIsoTrack") << "Labels used " << triggerEvent_ << " " << theTriggerResultsLabel_ << " "
                                     << labelBS << " " << edm::InputTag(modnam, labelRecVtx_, prdnam) << " "
                                     << labelGenTrack_ << " " << edm::InputTag(modnam, labelEB_, prdnam) << " "
                                     << edm::InputTag(modnam, labelEE_, prdnam) << " "
                                     << edm::InputTag(modnam, labelHBHE_, prdnam) << " " << labelTower_;
  }

  edm::LogVerbatim("HcalIsoTrack")
      << "Parameters read from config file \n"
      << "\t minPt " << selectionParameter_.minPt << "\t theTrackQuality " << theTrackQuality_ << "\t minQuality "
      << selectionParameter_.minQuality << "\t maxDxyPV " << selectionParameter_.maxDxyPV << "\t maxDzPV "
      << selectionParameter_.maxDzPV << "\t maxChi2 " << selectionParameter_.maxChi2 << "\t maxDpOverP "
      << selectionParameter_.maxDpOverP << "\t minOuterHit " << selectionParameter_.minOuterHit << "\t minLayerCrossed "
      << selectionParameter_.minLayerCrossed << "\t maxInMiss " << selectionParameter_.maxInMiss << "\t maxOutMiss "
      << selectionParameter_.maxOutMiss << "\t a_coneR " << a_coneR_ << ":" << a_coneR1_ << ":" << a_coneR2_
      << "\t a_charIsoR " << a_charIsoR_ << "\t a_mipR " << a_mipR_ << "\n pTrackMin_ " << pTrackMin_ << "\t eEcalMax_ "
      << eEcalMax_ << "\t maxRestrictionP_ " << maxRestrictionP_ << "\t slopeRestrictionP_ " << slopeRestrictionP_
      << "\t eIsolateStrong_ " << eIsolate1_ << "\t eIsolateSoft_ " << eIsolate2_ << "\t hcalScale_ " << hcalScale_
      << "\n\t momentumLow_ " << pTrackLow_ << "\t prescaleLow_ " << prescaleLow_ << "\t momentumHigh_ " << pTrackHigh_
      << "\t prescaleHigh_ " << prescaleHigh_ << "\n\t useRaw_ " << useRaw_ << "\t ignoreTrigger_ " << ignoreTrigger_
      << "\n\t useL1Trigegr_ " << useL1Trigger_ << "\t dataType_      " << dataType_ << "\t unCorrect_     "
      << unCorrect_ << "\t collapseDepth_ " << collapseDepth_ << "\t L1TrigName_    " << l1TrigName_
      << "\nThreshold for EB " << hitEthrEB_ << " EE " << hitEthrEE0_ << ":" << hitEthrEE1_ << ":" << hitEthrEE2_ << ":"
      << hitEthrEE3_ << ":" << hitEthrEELo_ << ":" << hitEthrEEHi_;
  edm::LogVerbatim("HcalIsoTrack") << "Process " << processName_ << " L1Filter:" << l1Filter_
                                   << " L2Filter:" << l2Filter_ << " L3Filter:" << l3Filter_;
  for (unsigned int k = 0; k < trigNames_.size(); ++k) {
    edm::LogVerbatim("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames_[k];
  }

  for (int i = 0; i < 10; i++)
    phibins_.push_back(-M_PI + 0.1 * (2 * i + 1) * M_PI);
  for (int i = 0; i < 8; ++i)
    etabins_.push_back(-2.1 + 0.6 * i);
  etadist_ = etabins_[1] - etabins_[0];
  phidist_ = phibins_[1] - phibins_[0];
  etahalfdist_ = 0.5 * etadist_;
  phihalfdist_ = 0.5 * phidist_;
  edm::LogVerbatim("HcalIsoTrack") << "EtaDist " << etadist_ << " " << etahalfdist_ << " PhiDist " << phidist_ << " "
                                   << phihalfdist_;
  unsigned int k1(0), k2(0);
  for (auto phi : phibins_) {
    edm::LogVerbatim("HcalIsoTrack") << "phibin_[" << k1 << "] " << phi;
    ++k1;
  }
  for (auto eta : etabins_) {
    edm::LogVerbatim("HcalIsoTrack") << "etabin_[" << k2 << "] " << eta;
    ++k2;
  }
}

void HcalIsoTrackStudy::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  t_Run = iEvent.id().run();
  t_Event = iEvent.id().event();
  t_DataType = dataType_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Run " << t_Run << " Event " << t_Event << " type " << t_DataType
                                   << " Luminosity " << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing();
#endif
  //Get magnetic field and ECAL channel status
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField* bField = bFieldH.product();

  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
  const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* theEcalSevlv = sevlv.product();

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  const CaloTopology* caloTopology = theCaloTopology.product();

  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();

  edm::ESHandle<HcalRespCorrs> resp;
  iSetup.get<HcalRespCorrsRcd>().get(resp);
  HcalRespCorrs* respCorrs = new HcalRespCorrs(*resp.product());
  respCorrs->setTopo(theHBHETopology);

  //=== genParticle information
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(tok_parts_, genParticles);

  bool okC(true);
  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  if (!trkCollection.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
    okC = false;
  }

  //event weight for FLAT sample
  t_EventWeight = 1.0;
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid())
    t_EventWeight = genEventInfo->weight();

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0, 0, 0);
  t_goodPV = t_nVtx = 0;
  if (recVtxs.isValid() && !(recVtxs->empty())) {
    t_nVtx = recVtxs->size();
    for (unsigned int k = 0; k < recVtxs->size(); ++k) {
      if (!((*recVtxs)[k].isFake()) && ((*recVtxs)[k].ndof() > 4)) {
        if (t_goodPV == 0)
          leadPV = math::XYZPoint((*recVtxs)[k].x(), (*recVtxs)[k].y(), (*recVtxs)[k].z());
        t_goodPV++;
      }
    }
  }
  if (t_goodPV == 0 && beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
  t_allvertex = t_goodPV;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Primary Vertex " << leadPV << " out of " << t_goodPV << " vertex";
  if (beamSpotH.isValid()) {
    edm::LogVerbatim("HcalIsoTrack") << " Beam Spot " << beamSpotH->position();
  }
#endif
  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEB_;
    okC = false;
  }
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEE_;
    okC = false;
  }
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  if (!hbhe.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelHBHE_;
    okC = false;
  }
  edm::Handle<CaloTowerCollection> caloTower;
  iEvent.getByToken(tok_cala_, caloTower);

  //Propagate tracks to calorimeter surface)
  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDirections, false);
  std::vector<spr::propagatedTrackID> trkCaloDets;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDets, false);
  std::vector<math::XYZTLorentzVector> vecL1, vecL3;
  t_RunNo = iEvent.id().run();
  t_EventNo = iEvent.id().event();
  t_Tracks = trkCollection->size();
  t_TracksProp = trkCaloDirections.size();
  t_ietaAll->clear();
  t_ietaGood->clear();
  t_trackType->clear();
  t_trgbits->clear();
  t_hltbits->clear();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "# of propagated tracks " << t_TracksProp << " out of " << t_Tracks
                                   << " with Trigger " << ignoreTrigger_;
#endif

  //Trigger
  t_trgbits->assign(trigNames_.size(), false);
  t_hltbits->assign(trigNames_.size(), false);
  t_TracksSaved = t_TracksLoose = t_TracksTight = 0;
  t_L1Bit = true;
  t_TrigPass = false;

  //L1
  l1GtUtils_->retrieveL1(iEvent, iSetup, tok_alg_);
  const std::vector<std::pair<std::string, bool>>& finalDecisions = l1GtUtils_->decisionsFinal();
  for (const auto& decision : finalDecisions) {
    if (decision.first.find(l1TrigName_) != std::string::npos) {
      t_L1Bit = decision.second;
      break;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Trigger Information for " << l1TrigName_ << " is " << t_L1Bit
                                   << " from a list of " << finalDecisions.size() << " decisions";
#endif

  //HLT
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(tok_trigRes_, triggerResults);
  if (triggerResults.isValid()) {
    const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
    const std::vector<std::string>& names = triggerNames.triggerNames();
    if (!trigNames_.empty()) {
      for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
        int hlt = triggerResults->accept(iHLT);
        for (unsigned int i = 0; i < trigNames_.size(); ++i) {
          if (names[iHLT].find(trigNames_[i]) != std::string::npos) {
            t_trgbits->at(i) = (hlt > 0);
            t_hltbits->at(i) = (hlt > 0);
            if (hlt > 0)
              t_TrigPass = true;
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HcalIsoTrack")
                << "This trigger " << names[iHLT] << " Flag " << hlt << ":" << t_trgbits->at(i);
#endif
          }
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "HLT Information shows " << t_TrigPass << ":" << trigNames_.empty() << ":" << okC;
#endif

  std::array<int, 3> ntksave{{0, 0, 0}};
  if (ignoreTrigger_ || useL1Trigger_) {
    t_l1pt = t_l1eta = t_l1phi = 0;
    t_l3pt = t_l3eta = t_l3phi = 0;
    if (ignoreTrigger_ || t_L1Bit)
      ntksave = fillTree(vecL1,
                         vecL3,
                         leadPV,
                         trkCaloDirections,
                         trkCaloDets,
                         geo,
                         caloTopology,
                         theHBHETopology,
                         theEcalChStatus,
                         theEcalSevlv,
                         barrelRecHitsHandle,
                         endcapRecHitsHandle,
                         hbhe,
                         caloTower,
                         genParticles,
                         respCorrs);
    t_TracksSaved = ntksave[0];
    t_TracksLoose = ntksave[1];
    t_TracksTight = ntksave[2];
  } else {
    trigger::TriggerEvent triggerEvent;
    edm::Handle<trigger::TriggerEvent> triggerEventHandle;
    iEvent.getByToken(tok_trigEvt_, triggerEventHandle);
    if (!triggerEventHandle.isValid()) {
      edm::LogWarning("HcalIsoTrack") << "Error! Can't get the product " << triggerEvent_.label();
    } else if (okC) {
      triggerEvent = *(triggerEventHandle.product());
      const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
      bool done(false);
      if (triggerResults.isValid()) {
        std::vector<std::string> modules;
        const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
        const std::vector<std::string>& names = triggerNames.triggerNames();
        for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
          bool ok = (t_TrigPass) || (trigNames_.empty());
          if (ok) {
            unsigned int triggerindx = hltConfig_.triggerIndex(names[iHLT]);
            const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
            std::vector<math::XYZTLorentzVector> vecL2;
            vecL1.clear();
            vecL3.clear();
            //loop over all trigger filters in event (i.e. filters passed)
            for (unsigned int ifilter = 0; ifilter < triggerEvent.sizeFilters(); ++ifilter) {
              std::vector<int> Keys;
              std::string label = triggerEvent.filterTag(ifilter).label();
              //loop over keys to objects passing this filter
              for (unsigned int imodule = 0; imodule < moduleLabels.size(); imodule++) {
                if (label.find(moduleLabels[imodule]) != std::string::npos) {
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HcalIsoTrack") << "FilterName " << label;
#endif
                  for (unsigned int ifiltrKey = 0; ifiltrKey < triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
                    Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
                    const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
                    math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
                    if (label.find(l2Filter_) != std::string::npos) {
                      vecL2.push_back(v4);
                    } else if (label.find(l3Filter_) != std::string::npos) {
                      vecL3.push_back(v4);
                    } else if ((label.find(l1Filter_) != std::string::npos) || (l1Filter_.empty())) {
                      vecL1.push_back(v4);
                    }
#ifdef EDM_ML_DEBUG
                    edm::LogVerbatim("HcalIsoTrack")
                        << "key " << ifiltrKey << " : pt " << TO.pt() << " eta " << TO.eta() << " phi " << TO.phi()
                        << " mass " << TO.mass() << " Id " << TO.id();
#endif
                  }
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HcalIsoTrack")
                      << "sizes " << vecL1.size() << ":" << vecL2.size() << ":" << vecL3.size();
#endif
                }
              }
            }
            //// deta, dphi and dR for leading L1 object with L2 objects
            math::XYZTLorentzVector mindRvec1;
            double mindR1(999);
            for (unsigned int i = 0; i < vecL2.size(); i++) {
              double dr = dR(vecL1[0], vecL2[i]);
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("HcalIsoTrack") << "lvl2[" << i << "] dR " << dr;
#endif
              if (dr < mindR1) {
                mindR1 = dr;
                mindRvec1 = vecL2[i];
              }
            }
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HcalIsoTrack") << "L2 object closest to L1 " << mindRvec1 << " at Dr " << mindR1;
#endif

            if (!vecL1.empty()) {
              t_l1pt = vecL1[0].pt();
              t_l1eta = vecL1[0].eta();
              t_l1phi = vecL1[0].phi();
            } else {
              t_l1pt = t_l1eta = t_l1phi = 0;
            }
            if (!vecL3.empty()) {
              t_l3pt = vecL3[0].pt();
              t_l3eta = vecL3[0].eta();
              t_l3phi = vecL3[0].phi();
            } else {
              t_l3pt = t_l3eta = t_l3phi = 0;
            }
            // Now fill in the tree for each selected track
            if (!done) {
              ntksave = fillTree(vecL1,
                                 vecL3,
                                 leadPV,
                                 trkCaloDirections,
                                 trkCaloDets,
                                 geo,
                                 caloTopology,
                                 theHBHETopology,
                                 theEcalChStatus,
                                 theEcalSevlv,
                                 barrelRecHitsHandle,
                                 endcapRecHitsHandle,
                                 hbhe,
                                 caloTower,
                                 genParticles,
                                 respCorrs);
              t_TracksSaved += ntksave[0];
              t_TracksLoose += ntksave[1];
              t_TracksTight += ntksave[2];
              done = true;
            }
          }
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Final results on selected tracks " << t_TracksSaved << ":" << t_TracksLoose
                                   << ":" << t_TracksTight;
#endif
  t_TrigPassSel = (t_TracksSaved > 0);
  tree2->Fill();
}

void HcalIsoTrackStudy::beginJob() {
  tree = fs->make<TTree>("CalibTreeExtended", "CalibTreeExtended");

  tree->Branch("t_Run", &t_Run, "t_Run/I");
  tree->Branch("t_Event", &t_Event, "t_Event/I");
  tree->Branch("t_DataType", &t_DataType, "t_DataType/I");
  tree->Branch("t_ieta", &t_ieta, "t_ieta/I");
  tree->Branch("t_iphi", &t_iphi, "t_iphi/I");
  tree->Branch("t_EventWeight", &t_EventWeight, "t_EventWeight/D");
  tree->Branch("t_nVtx", &t_nVtx, "t_nVtx/I");
  tree->Branch("t_nTrk", &t_nTrk, "t_nTrk/I");
  tree->Branch("t_goodPV", &t_goodPV, "t_goodPV/I");
  tree->Branch("t_l1pt", &t_l1pt, "t_l1pt/D");
  tree->Branch("t_l1eta", &t_l1eta, "t_l1eta/D");
  tree->Branch("t_l1phi", &t_l1phi, "t_l1phi/D");
  tree->Branch("t_l3pt", &t_l3pt, "t_l3pt/D");
  tree->Branch("t_l3eta", &t_l3eta, "t_l3eta/D");
  tree->Branch("t_l3phi", &t_l3phi, "t_l3phi/D");
  tree->Branch("t_p", &t_p, "t_p/D");
  tree->Branch("t_pt", &t_pt, "t_pt/D");
  tree->Branch("t_phi", &t_phi, "t_phi/D");

  t_mapP = new std::vector<double>();
  t_mapPt = new std::vector<double>();
  t_mapEta = new std::vector<double>();
  t_mapPhi = new std::vector<double>();
  tree->Branch("t_mapP", "std::vector<double>", &t_mapP);
  tree->Branch("t_mapPt", "std::vector<double>", &t_mapPt);
  tree->Branch("t_mapEta", "std::vector<double>", &t_mapEta);
  tree->Branch("t_mapPhi", "std::vector<double>", &t_mapPhi);

  tree->Branch("t_mindR1", &t_mindR1, "t_mindR1/D");
  tree->Branch("t_mindR2", &t_mindR2, "t_mindR2/D");
  tree->Branch("t_eMipDR", &t_eMipDR, "t_eMipDR/D");
  tree->Branch("t_eHcal", &t_eHcal, "t_eHcal/D");
  tree->Branch("t_eHcal10", &t_eHcal10, "t_eHcal10/D");
  tree->Branch("t_eHcal30", &t_eHcal30, "t_eHcal30/D");
  tree->Branch("t_hmaxNearP", &t_hmaxNearP, "t_hmaxNearP/D");
  tree->Branch("t_emaxNearP", &t_emaxNearP, "t_emaxNearP/D");
  tree->Branch("t_eAnnular", &t_eAnnular, "t_eAnnular/D");
  tree->Branch("t_hAnnular", &t_hAnnular, "t_hAnnular/D");
  tree->Branch("t_rhoh", &t_rhoh, "t_rhoh/D");
  tree->Branch("t_selectTk", &t_selectTk, "t_selectTk/O");
  tree->Branch("t_qltyFlag", &t_qltyFlag, "t_qltyFlag/O");
  tree->Branch("t_qltyMissFlag", &t_qltyMissFlag, "t_qltyMissFlag/O");
  tree->Branch("t_qltyPVFlag", &t_qltyPVFlag, "t_qltyPVFlag/O");
  tree->Branch("t_gentrackP", &t_gentrackP, "t_gentrackP/D");
  tree->Branch("t_gentrackE", &t_gentrackE, "t_gentrackE/D");

  t_DetIds = new std::vector<unsigned int>();
  t_DetIds1 = new std::vector<unsigned int>();
  t_DetIds3 = new std::vector<unsigned int>();
  t_DetIdEC = new std::vector<unsigned int>();
  t_DetIdHC = new std::vector<unsigned int>();
  t_HitEnergies = new std::vector<double>();
  t_HitEnergies1 = new std::vector<double>();
  t_HitEnergies3 = new std::vector<double>();
  t_HitEnergyEC = new std::vector<double>();
  t_HitDistEC = new std::vector<double>();
  t_HitEnergyHC = new std::vector<double>();
  t_HitDistHC = new std::vector<double>();
  t_trgbits = new std::vector<bool>();
  tree->Branch("t_DetIds", "std::vector<unsigned int>", &t_DetIds);
  tree->Branch("t_HitEnergies", "std::vector<double>", &t_HitEnergies);
  tree->Branch("t_trgbits", "std::vector<bool>", &t_trgbits);
  tree->Branch("t_DetIds1", "std::vector<unsigned int>", &t_DetIds1);
  tree->Branch("t_HitEnergies1", "std::vector<double>", &t_HitEnergies1);
  tree->Branch("t_DetIds3", "std::vector<unsigned int>", &t_DetIds3);
  tree->Branch("t_HitEnergies3", "std::vector<double>", &t_HitEnergies3);
  tree->Branch("t_DetIdEC", "std::vector<unsigned int>", &t_DetIdEC);
  tree->Branch("t_HitEnergyEC", "std::vector<double>", &t_HitEnergyEC);
  tree->Branch("t_HitDistEC", "std::vector<double>", &t_HitDistEC);
  tree->Branch("t_DetIdHC", "std::vector<unsigned int>", &t_DetIdHC);
  tree->Branch("t_HitEnergyHC", "std::vector<double>", &t_HitEnergyHC);
  tree->Branch("t_HitDistHC", "std::vector<double>", &t_HitDistHC);

  tree2 = fs->make<TTree>("EventInfo", "Event Information");
  tree2->Branch("t_RunNo", &t_RunNo, "t_RunNo/i");
  tree2->Branch("t_EventNo", &t_EventNo, "t_EventNo/i");
  tree2->Branch("t_Tracks", &t_Tracks, "t_Tracks/I");
  tree2->Branch("t_TracksProp", &t_TracksProp, "t_TracksProp/I");
  tree2->Branch("t_TracksSaved", &t_TracksSaved, "t_TracksSaved/I");
  tree2->Branch("t_TracksLoose", &t_TracksLoose, "t_TracksLoose/I");
  tree2->Branch("t_TracksTight", &t_TracksTight, "t_TracksTight/I");
  tree2->Branch("t_TrigPass", &t_TrigPass, "t_TrigPass/O");
  tree2->Branch("t_TrigPassSel", &t_TrigPassSel, "t_TrigPassSel/O");
  tree2->Branch("t_L1Bit", &t_L1Bit, "t_L1Bit/O");
  tree2->Branch("t_allvertex", &t_allvertex, "t_allvertex/I");
  t_hltbits = new std::vector<bool>();
  t_ietaAll = new std::vector<int>();
  t_ietaGood = new std::vector<int>();
  t_trackType = new std::vector<int>();
  tree2->Branch("t_ietaAll", "std::vector<int>", &t_ietaAll);
  tree2->Branch("t_ietaGood", "std::vector<int>", &t_ietaGood);
  tree2->Branch("t_trackType", "std::vector<int>", &t_trackType);
  tree2->Branch("t_hltbits", "std::vector<bool>", &t_hltbits);
}

// ------------ method called when starting to processes a run  ------------
void HcalIsoTrackStudy::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
  hdc_ = pHRNDC.product();

  bool changed_(true);
  bool flag = hltConfig_.init(iRun, iSetup, processName_, changed_);
  edm::LogVerbatim("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() << " process " << processName_
                                   << " init flag " << flag << " change flag " << changed_;
  // check if trigger names in (new) config
  if (changed_) {
    changed_ = false;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "New trigger menu found !!!";
#endif
    const unsigned int n(hltConfig_.size());
    for (unsigned itrig = 0; itrig < trigNames_.size(); itrig++) {
      unsigned int triggerindx = hltConfig_.triggerIndex(trigNames_[itrig]);
      if (triggerindx >= n) {
        edm::LogWarning("HcalIsoTrack") << trigNames_[itrig] << " " << triggerindx << " does not exist in "
                                        << "the current menu";
#ifdef EDM_ML_DEBUG
      } else {
        edm::LogVerbatim("HcalIsoTrack") << trigNames_[itrig] << " " << triggerindx << " exists";
#endif
      }
    }
  }
}

// ------------ method called when ending the processing of a run  ------------
void HcalIsoTrackStudy::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun_++;
  edm::LogVerbatim("HcalIsoTrack") << "endRun[" << nRun_ << "] " << iRun.run();
}

void HcalIsoTrackStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> trig = {"HLT_PFJet40",
                                   "HLT_PFJet60",
                                   "HLT_PFJet80",
                                   "HLT_PFJet140",
                                   "HLT_PFJet200",
                                   "HLT_PFJet260",
                                   "HLT_PFJet320",
                                   "HLT_PFJet400",
                                   "HLT_PFJet450",
                                   "HLT_PFJet500"};
  desc.add<std::vector<std::string>>("triggers", trig);
  desc.add<std::string>("processName", "HLT");
  desc.add<std::string>("l1Filter", "");
  desc.add<std::string>("l2Filter", "L2Filter");
  desc.add<std::string>("l3Filter", "Filter");
  // following 10 parameters are parameters to select good tracks
  desc.add<std::string>("trackQuality", "highPurity");
  desc.add<double>("minTrackPt", 1.0);
  desc.add<double>("maxDxyPV", 0.02);
  desc.add<double>("maxDzPV", 0.02);
  desc.add<double>("maxChi2", 5.0);
  desc.add<double>("maxDpOverP", 0.1);
  desc.add<int>("minOuterHit", 4);
  desc.add<int>("minLayerCrossed", 8);
  desc.add<int>("maxInMiss", 0);
  desc.add<int>("maxOutMiss", 0);
  // Minimum momentum of selected isolated track and signal zone
  desc.add<double>("minimumTrackP", 20.0);
  desc.add<double>("coneRadius", 34.98);
  // signal zone in ECAL and MIP energy cutoff
  desc.add<double>("coneRadiusMIP", 14.0);
  desc.add<double>("maximumEcalEnergy", 2.0);
  // following 4 parameters are for isolation cuts and described in the code
  desc.add<double>("maxTrackP", 8.0);
  desc.add<double>("slopeTrackP", 0.05090504066);
  desc.add<double>("isolationEnergyTight", 2.0);
  desc.add<double>("isolationEnergyLoose", 10.0);
  // energy thershold for ECAL (from Egamma group)
  desc.add<double>("EBHitEnergyThreshold", 0.08);
  desc.add<double>("EEHitEnergyThreshold0", 0.30);
  desc.add<double>("EEHitEnergyThreshold1", 0.00);
  desc.add<double>("EEHitEnergyThreshold2", 0.00);
  desc.add<double>("EEHitEnergyThreshold3", 0.00);
  desc.add<double>("EEHitEnergyThresholdLow", 0.30);
  desc.add<double>("EEHitEnergyThresholdHigh", 0.30);
  // prescale factors
  desc.add<double>("momentumLow", 40.0);
  desc.add<double>("momentumHigh", 60.0);
  desc.add<int>("prescaleLow", 1);
  desc.add<int>("prescaleHigh", 1);
  // various labels for collections used in the code
  desc.add<edm::InputTag>("labelTriggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<edm::InputTag>("labelTriggerResult", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<std::string>("labelTrack", "generalTracks");
  desc.add<std::string>("labelVertex", "offlinePrimaryVertices");
  desc.add<std::string>("labelEBRecHit", "EcalRecHitsEB");
  desc.add<std::string>("labelEERecHit", "EcalRecHitsEE");
  desc.add<std::string>("labelHBHERecHit", "hbhereco");
  desc.add<std::string>("labelBeamSpot", "offlineBeamSpot");
  desc.add<std::string>("labelCaloTower", "towerMaker");
  desc.add<edm::InputTag>("algInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<edm::InputTag>("extInputTag", edm::InputTag("gtStage2Digis"));
  desc.addUntracked<std::string>("moduleName", "");
  desc.addUntracked<std::string>("producerName", "");
  //  Various flags used for selecting tracks, choice of energy Method2/0
  //  Data type 0/1 for single jet trigger or others
  desc.addUntracked<int>("useRaw", 0);
  desc.addUntracked<bool>("ignoreTriggers", false);
  desc.addUntracked<bool>("useL1Trigger", false);
  desc.addUntracked<double>("hcalScale", 1.0);
  desc.addUntracked<int>("dataType", 0);
  desc.addUntracked<bool>("unCorrect", false);
  desc.addUntracked<bool>("collapseDepth", false);
  desc.addUntracked<std::string>("l1TrigName", "L1_SingleJet60");
  desc.addUntracked<int>("matrixECAL", 5);
  desc.addUntracked<int>("matrixHCAL", 3);
  desc.addUntracked<double>("mapRadius", 34.98);
  desc.addUntracked<bool>("get2Ddist", false);
  descriptions.add("hcalIsoTrackStudy", desc);
}

std::array<int, 3> HcalIsoTrackStudy::fillTree(std::vector<math::XYZTLorentzVector>& vecL1,
                                               std::vector<math::XYZTLorentzVector>& vecL3,
                                               math::XYZPoint& leadPV,
                                               std::vector<spr::propagatedTrackDirection>& trkCaloDirections,
                                               std::vector<spr::propagatedTrackID>& trkCaloDets,
                                               const CaloGeometry* geo,
                                               const CaloTopology* caloTopology,
                                               const HcalTopology* theHBHETopology,
                                               const EcalChannelStatus* theEcalChStatus,
                                               const EcalSeverityLevelAlgo* theEcalSevlv,
                                               edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle,
                                               edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle,
                                               edm::Handle<HBHERecHitCollection>& hbhe,
                                               edm::Handle<CaloTowerCollection>& tower,
                                               edm::Handle<reco::GenParticleCollection>& genParticles,
                                               const HcalRespCorrs* respCorrs) {
  int nSave(0), nLoose(0), nTight(0);
  //Loop over tracks
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  unsigned int nTracks(0), nselTracks(0);
  t_nTrk = trkCaloDirections.size();
  t_rhoh = (tower.isValid()) ? rhoh(tower) : 0;
  for (trkDetItr = trkCaloDirections.begin(), nTracks = 0; trkDetItr != trkCaloDirections.end();
       trkDetItr++, nTracks++) {
    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
    math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "This track : " << nTracks << " (pt|eta|phi|p) :" << pTrack->pt() << "|"
                                     << pTrack->eta() << "|" << pTrack->phi() << "|" << pTrack->p();
#endif
    t_mindR2 = 999;
    for (unsigned int k = 0; k < vecL3.size(); ++k) {
      double dr = dR(vecL3[k], v4);
      if (dr < t_mindR2) {
        t_mindR2 = dr;
      }
    }
    t_mindR1 = (!vecL1.empty()) ? dR(vecL1[0], v4) : 999;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "Closest L3 object at dr :" << t_mindR2 << " and from L1 " << t_mindR1;
#endif
    t_ieta = t_iphi = 0;
    if (trkDetItr->okHCAL) {
      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
      t_ieta = detId.ieta();
      t_iphi = detId.iphi();
      if (t_p > 40.0 && t_p <= 60.0)
        t_ietaAll->emplace_back(t_ieta);
    }
    //Selection of good track
    t_selectTk = spr::goodTrack(pTrack, leadPV, selectionParameter_, false);
    spr::trackSelectionParameters oneCutParameters = selectionParameter_;
    oneCutParameters.maxDxyPV = 10;
    oneCutParameters.maxDzPV = 100;
    oneCutParameters.maxInMiss = 2;
    oneCutParameters.maxOutMiss = 2;
    bool qltyFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, false);
    oneCutParameters = selectionParameter_;
    oneCutParameters.maxDxyPV = 10;
    oneCutParameters.maxDzPV = 100;
    t_qltyMissFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, false);
    oneCutParameters = selectionParameter_;
    oneCutParameters.maxInMiss = 2;
    oneCutParameters.maxOutMiss = 2;
    t_qltyPVFlag = spr::goodTrack(pTrack, leadPV, oneCutParameters, false);
    double eIsolation = maxRestrictionP_ * exp(slopeRestrictionP_ * std::abs((double)t_ieta));
    if (eIsolation < eIsolate1_)
      eIsolation = eIsolate1_;
    if (eIsolation < eIsolate2_)
      eIsolation = eIsolate2_;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "qltyFlag|okECAL|okHCAL : " << qltyFlag << "|" << trkDetItr->okECAL << "|"
                                     << trkDetItr->okHCAL << " eIsolation " << eIsolation;
#endif
    t_qltyFlag = (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL);
    if (t_qltyFlag) {
      nselTracks++;
      int nNearTRKs(0);
      std::vector<DetId> eIds;
      std::vector<double> eHit;
      t_eMipDR = spr::eCone_ecal(geo,
                                 barrelRecHitsHandle,
                                 endcapRecHitsHandle,
                                 trkDetItr->pointHCAL,
                                 trkDetItr->pointECAL,
                                 a_mipR_,
                                 trkDetItr->directionECAL,
                                 eIds,
                                 eHit);
      double eEcal(0);
      for (unsigned int k = 0; k < eIds.size(); ++k) {
        const GlobalPoint& pos = geo->getPosition(eIds[k]);
        double eta = std::abs(pos.eta());
        double eThr(hitEthrEB_);
        if (eIds[k].subdetId() != EcalBarrel) {
          eThr = (((eta * hitEthrEE3_ + hitEthrEE2_) * eta + hitEthrEE1_) * eta + hitEthrEE0_);
          if (eThr < hitEthrEELo_)
            eThr = hitEthrEELo_;
          else if (eThr > hitEthrEEHi_)
            eThr = hitEthrEEHi_;
        }
        if (eHit[k] > eThr)
          eEcal += eHit[k];
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalIsoTrack") << "eMIP before and after: " << t_eMipDR << ":" << eEcal;
#endif
      t_eMipDR = eEcal;
      t_emaxNearP = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 15, 15);
      const DetId cellE(trkDetItr->detIdECAL);
      std::pair<double, bool> e11x11P = spr::eECALmatrix(cellE,
                                                         barrelRecHitsHandle,
                                                         endcapRecHitsHandle,
                                                         *theEcalChStatus,
                                                         geo,
                                                         caloTopology,
                                                         theEcalSevlv,
                                                         5,
                                                         5,
                                                         -100.0,
                                                         -100.0,
                                                         -100.0,
                                                         100.0);
      std::pair<double, bool> e15x15P = spr::eECALmatrix(cellE,
                                                         barrelRecHitsHandle,
                                                         endcapRecHitsHandle,
                                                         *theEcalChStatus,
                                                         geo,
                                                         caloTopology,
                                                         theEcalSevlv,
                                                         7,
                                                         7,
                                                         -100.0,
                                                         -100.0,
                                                         -100.0,
                                                         100.0);
      if (e11x11P.second && e15x15P.second) {
        t_eAnnular = (e15x15P.first - e11x11P.first);
      } else {
        t_eAnnular = -(e15x15P.first - e11x11P.first);
      }
      t_hmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, false);
      const DetId cellH(trkDetItr->detIdHCAL);
      double h5x5 = spr::eHCALmatrix(
          theHBHETopology, cellH, hbhe, 2, 2, false, true, -100.0, -100.0, -100.0, -100.0, -100.0, 100.0);
      double h7x7 = spr::eHCALmatrix(
          theHBHETopology, cellH, hbhe, 3, 3, false, true, -100.0, -100.0, -100.0, -100.0, -100.0, 100.0);
      t_hAnnular = h7x7 - h5x5;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalIsoTrack") << "max p Near (Ecal) " << t_emaxNearP << " (Hcal) " << t_hmaxNearP
                                       << " Annular E (Ecal) " << e11x11P.first << ":" << e15x15P.first << ":"
                                       << t_eAnnular << " (Hcal) " << h5x5 << ":" << h7x7 << ":" << t_hAnnular;
#endif
      t_gentrackP = trackP(pTrack, genParticles);
      t_gentrackE = trackE(pTrack, genParticles);
      if (t_eMipDR < eEcalMax_ && t_hmaxNearP < eIsolation) {
        t_DetIds->clear();
        t_HitEnergies->clear();
        t_DetIds1->clear();
        t_HitEnergies1->clear();
        t_DetIds3->clear();
        t_HitEnergies3->clear();
        t_DetIdEC->clear();
        t_HitEnergyEC->clear();
        t_HitDistEC->clear();
        t_DetIdHC->clear();
        t_HitEnergyHC->clear();
        t_HitDistHC->clear();
        t_mapP->clear();
        t_mapPt->clear();
        t_mapEta->clear();
        t_mapPhi->clear();

        int nRecHits(-999), nRecHits1(-999), nRecHits3(-999);
        std::vector<DetId> ids, ids1, ids3;
        std::vector<double> edet0, edet1, edet3;
        t_eHcal = spr::eCone_hcal(geo,
                                  hbhe,
                                  trkDetItr->pointHCAL,
                                  trkDetItr->pointECAL,
                                  a_coneR_,
                                  trkDetItr->directionHCAL,
                                  nRecHits,
                                  ids,
                                  edet0,
                                  useRaw_);
        storeEnergy(0, respCorrs, ids, edet0, t_eHcal, t_DetIds, t_HitEnergies);

        //----- hcal energy in the extended cone 1 (a_coneR+10) --------------
        t_eHcal10 = spr::eCone_hcal(geo,
                                    hbhe,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_coneR1_,
                                    trkDetItr->directionHCAL,
                                    nRecHits1,
                                    ids1,
                                    edet1,
                                    useRaw_);
        storeEnergy(1, respCorrs, ids1, edet1, t_eHcal10, t_DetIds1, t_HitEnergies1);

        //----- hcal energy in the extended cone 3 (a_coneR+30) --------------
        t_eHcal30 = spr::eCone_hcal(geo,
                                    hbhe,
                                    trkDetItr->pointHCAL,
                                    trkDetItr->pointECAL,
                                    a_coneR2_,
                                    trkDetItr->directionHCAL,
                                    nRecHits3,
                                    ids3,
                                    edet3,
                                    useRaw_);
        storeEnergy(3, respCorrs, ids3, edet3, t_eHcal30, t_DetIds3, t_HitEnergies3);

        fillECALmatrix(
            cellE, barrelRecHitsHandle, endcapRecHitsHandle, geo, caloTopology, pTrack, matrixECAL_, matrixECAL_);
        fillHCALmatrix(theHBHETopology, cellH, hbhe, geo, pTrack, matrixHCAL_, matrixHCAL_);

        TrackMap(nTracks, trkCaloDirections, mapR_);

        t_p = pTrack->p();
        t_pt = pTrack->pt();
        t_phi = pTrack->phi();

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HcalIsoTrack") << "This track : " << nTracks << " (pt|eta|phi|p) :" << t_pt << "|"
                                         << pTrack->eta() << "|" << t_phi << "|" << t_p << " Generator Level p "
                                         << t_gentrackP;
        edm::LogVerbatim("HcalIsoTrack") << "e_MIP " << t_eMipDR << " Chg Isolation " << t_hmaxNearP << " eHcal"
                                         << t_eHcal << " ieta " << t_ieta << " Quality " << t_qltyMissFlag << ":"
                                         << t_qltyPVFlag << ":" << t_selectTk;
        for (unsigned int ll = 0; ll < t_DetIds->size(); ll++) {
          edm::LogVerbatim("HcalIsoTrack")
              << "det id is = " << t_DetIds->at(ll) << "   hit enery is  = " << t_HitEnergies->at(ll);
        }
        for (unsigned int ll = 0; ll < t_DetIds1->size(); ll++) {
          edm::LogVerbatim("HcalIsoTrack")
              << "det id is = " << t_DetIds1->at(ll) << "   hit enery is  = " << t_HitEnergies1->at(ll);
        }
        for (unsigned int ll = 0; ll < t_DetIds3->size(); ll++)
          edm::LogVerbatim("HcalIsoTrack")
              << "det id is = " << t_DetIds3->at(ll) << "   hit enery is  = " << t_HitEnergies3->at(ll);
        edm::LogVerbatim("HcalIsoTrack") << t_DetIdEC->size() << ":" << t_DetIdHC->size() << " hits from "
                                         << "ECAL and HCAL";
        for (unsigned int ll = 0; ll < t_DetIdEC->size(); ++ll)
          edm::LogVerbatim("HcalIsoTrack") << "EC[" << ll << "] ID " << std::hex << t_DetIdEC->at(ll) << std::dec
                                           << " D " << t_HitDistEC->at(ll) << " E " << t_HitEnergyEC->at(ll);
        for (unsigned int ll = 0; ll < t_DetIdHC->size(); ++ll)
          edm::LogVerbatim("HcalIsoTrack") << "HC[" << ll << "] ID " << std::hex << t_DetIdHC->at(ll) << std::dec
                                           << " D " << t_HitDistHC->at(ll) << " E " << t_HitEnergyHC->at(ll);
#endif
        bool accept(false);
        if (t_p > pTrackMin_) {
          if (t_p < pTrackLow_) {
            ++nLow_;
            if (prescaleLow_ <= 1)
              accept = true;
            else if (nLow_ % prescaleLow_ == 1)
              accept = true;
          } else if (t_p > pTrackHigh_) {
            ++nHigh_;
            if (prescaleHigh_ <= 1)
              accept = true;
            else if (nHigh_ % prescaleHigh_ == 1)
              accept = true;
          } else {
            accept = true;
          }
        }
        if (accept) {
          tree->Fill();
          nSave++;
          int type(0);
          if (t_eMipDR < 1.0) {
            if (t_hmaxNearP < eIsolate2_) {
              ++nLoose;
              type = 1;
            }
            if (t_hmaxNearP < eIsolate1_) {
              ++nTight;
              type = 2;
            }
          }
          if (t_p > 40.0 && t_p <= 60.0 && t_selectTk) {
            t_ietaGood->emplace_back(t_ieta);
            t_trackType->emplace_back(type);
          }
#ifdef EDM_ML_DEBUG
          for (unsigned int k = 0; k < t_trgbits->size(); k++) {
            edm::LogVerbatim("HcalIsoTrack") << "trigger bit is  = " << t_trgbits->at(k);
          }
#endif
        }
      }
    }
  }
  std::array<int, 3> i3{{nSave, nLoose, nTight}};
  return i3;
}

double HcalIsoTrackStudy::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaR(vec1.eta(), vec1.phi(), vec2.eta(), vec2.phi());
}

double HcalIsoTrackStudy::trackP(const reco::Track* pTrack,
                                 const edm::Handle<reco::GenParticleCollection>& genParticles) {
  double pmom = -1.0;
  if (genParticles.isValid()) {
    double mindR(999.9);
    for (const auto& p : (*genParticles)) {
      double dR = reco::deltaR(pTrack->eta(), pTrack->phi(), p.momentum().Eta(), p.momentum().Phi());
      if (dR < mindR) {
        mindR = dR;
        pmom = p.momentum().R();
        //	std::cout<<"p.E() :"<<p.energy()<<"   p.p() :"<<p.momentum().R()<<" p.M  :"<<p.mass()<<std::endl;
      }
    }
  }
  return pmom;
}

double HcalIsoTrackStudy::trackE(const reco::Track* pTrack,
                                 const edm::Handle<reco::GenParticleCollection>& genParticles) {
  double pE = -1.0;
  if (genParticles.isValid()) {
    double mindR(999.9);
    for (const auto& p : (*genParticles)) {
      double dR = reco::deltaR(pTrack->eta(), pTrack->phi(), p.momentum().Eta(), p.momentum().Phi());
      if (dR < mindR) {
        mindR = dR;
        pE = p.energy();
        //	std::cout<<"p.E() :"<<p.energy()<<"   p.p() :"<<p.momentum().R()<<" p.M  :"<<p.mass()<<std::endl;
      }
    }
  }
  return pE;
}

double HcalIsoTrackStudy::rhoh(const edm::Handle<CaloTowerCollection>& tower) {
  std::vector<double> sumPFNallSMDQH2;
  sumPFNallSMDQH2.reserve(phibins_.size() * etabins_.size());

  for (auto eta : etabins_) {
    for (auto phi : phibins_) {
      double hadder = 0;
      for (const auto& pf_it : (*tower)) {
        if (fabs(eta - pf_it.eta()) > etahalfdist_)
          continue;
        if (fabs(reco::deltaPhi(phi, pf_it.phi())) > phihalfdist_)
          continue;
        hadder += pf_it.hadEt();
      }
      sumPFNallSMDQH2.emplace_back(hadder);
    }
  }

  double evt_smdq(0);
  std::sort(sumPFNallSMDQH2.begin(), sumPFNallSMDQH2.end());
  if (sumPFNallSMDQH2.size() % 2)
    evt_smdq = sumPFNallSMDQH2[(sumPFNallSMDQH2.size() - 1) / 2];
  else
    evt_smdq = (sumPFNallSMDQH2[sumPFNallSMDQH2.size() / 2] + sumPFNallSMDQH2[(sumPFNallSMDQH2.size() - 2) / 2]) / 2.;
  double rhoh = evt_smdq / (etadist_ * phidist_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Rho " << evt_smdq << ":" << rhoh;
#endif
  return rhoh;
}

void HcalIsoTrackStudy::storeEnergy(int indx,
                                    const HcalRespCorrs* respCorrs,
                                    const std::vector<DetId>& ids,
                                    std::vector<double>& edet,
                                    double& eHcal,
                                    std::vector<unsigned int>* detIds,
                                    std::vector<double>* hitEnergies) {
  double ehcal(0);
  if (unCorrect_) {
    for (unsigned int k = 0; k < ids.size(); ++k) {
      double corr = (respCorrs->getValues(ids[k]))->getValue();
      if (corr != 0)
        edet[k] /= corr;
      ehcal += edet[k];
    }
  } else {
    for (const auto& en : edet)
      ehcal += en;
  }
  if (std::abs(ehcal - eHcal) > 0.001)
    edm::LogWarning("HcalIsoTrack") << "Check inconsistent energies: " << indx << " " << eHcal << ":" << ehcal
                                    << " from " << ids.size() << " cells";
  eHcal = hcalScale_ * ehcal;

  if (collapseDepth_) {
    std::map<HcalDetId, double> hitMap;
    for (unsigned int k = 0; k < ids.size(); ++k) {
      HcalDetId id = hdc_->mergedDepthDetId(HcalDetId(ids[k]));
      auto itr = hitMap.find(id);
      if (itr == hitMap.end()) {
        hitMap[id] = edet[k];
      } else {
        (itr->second) += edet[k];
      }
    }
    detIds->reserve(hitMap.size());
    hitEnergies->reserve(hitMap.size());
    for (const auto& hit : hitMap) {
      detIds->emplace_back(hit.first.rawId());
      hitEnergies->emplace_back(hit.second);
    }
  } else {
    detIds->reserve(ids.size());
    hitEnergies->reserve(ids.size());
    for (unsigned int k = 0; k < ids.size(); ++k) {
      detIds->emplace_back(ids[k].rawId());
      hitEnergies->emplace_back(edet[k]);
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Input to storeEnergy with " << ids.size() << " cells";
  for (unsigned int k = 0; k < ids.size(); ++k)
    edm::LogVerbatim("HcalIsoTrack") << "Hit [" << k << "] " << HcalDetId(ids[k]) << " E " << edet[k];
  edm::LogVerbatim("HcalIsoTrack") << "Output of storeEnergy with " << detIds->size() << " cells and Etot " << eHcal;
  for (unsigned int k = 0; k < detIds->size(); ++k)
    edm::LogVerbatim("HcalIsoTrack") << "Hit [" << k << "] " << HcalDetId((*detIds)[k]) << " E " << (*hitEnergies)[k];
#endif
}

void HcalIsoTrackStudy::fillECALmatrix(const DetId& detId,
                                       edm::Handle<EcalRecHitCollection>& hitsEB,
                                       edm::Handle<EcalRecHitCollection>& hitsEE,
                                       const CaloGeometry* geo,
                                       const CaloTopology* caloTopology,
                                       const reco::Track* pTrack,
                                       int ieta,
                                       int iphi) {
  //Find the position of the central hit
  std::vector<EcalRecHitCollection::const_iterator> hits;

  GlobalPoint center;
  if (detId.subdetId() == EcalEndcap) {
    EEDetId EEid = EEDetId(detId);
    center = geo->getPosition(EEid);
  } else if (detId.subdetId() == EcalBarrel) {
    EBDetId EBid = EBDetId(detId);
    center = geo->getPosition(EBid);
  }
  std::vector<DetId> vdets;
  spr::matrixECALIds(detId, ieta, iphi, geo, caloTopology, vdets, false);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "Inside the fill func |||||||||||||||||"
                                   << "center.x(): " << center.x() << "| center.y(): " << center.y()
                                   << "| center.z(): " << center.z() << "\n"
                                   << "Size of detid vector: " << vdets.size();
#endif
  int i = 0;
  for (auto const& id : vdets) {
    i++;
    //    std::cout<<"counter inside vdet:"<<i<<std::endl;
    hits.clear();
    if (id.subdetId() == EcalBarrel) {
      spr::findHit(hitsEB, id, hits, false);
    } else if (id.subdetId() == EcalEndcap) {
      spr::findHit(hitsEE, id, hits, false);
    }
    if (i == 1 && hits.empty()) {
      t_DetIdEC->emplace_back(id);
      t_HitEnergyEC->emplace_back(0.0);
      t_HitDistEC->emplace_back(0.0);
    }
    if (!hits.empty()) {
      for (auto const& hit : hits) {
        GlobalPoint pos = spr::getGpos(geo, hit);
        GlobalVector v1(pos.x() - center.x(), pos.y() - center.y(), pos.z() - center.z());
        double dist = v1.mag();
        if (get2Ddist_ && (dist > 0)) {
          double cth = ((v1.x() * pTrack->px() + v1.y() * pTrack->py() + v1.z() * pTrack->pz()) / (dist * pTrack->p()));
          dist *= std::sqrt(1.0 - cth * cth);
        }
        double ener = hit->energy();
        t_DetIdEC->emplace_back(hit->id());
        t_HitEnergyEC->emplace_back(ener);
        t_HitDistEC->emplace_back(dist);
      }
    }
  }
}

void HcalIsoTrackStudy::fillHCALmatrix(const HcalTopology* topology,
                                       const DetId& detId,
                                       edm::Handle<HBHERecHitCollection>& hbhe,
                                       const CaloGeometry* geo,
                                       const reco::Track* pTrack,
                                       int ieta,
                                       int iphi) {
  //Find the position of the central hit
  std::vector<HBHERecHitCollection::const_iterator> hits;
  spr::hitHCALmatrix(topology, detId, hbhe, ieta, iphi, hits, false, false);
  GlobalPoint center = (static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(detId)))->getPosition(detId);
  for (auto const& hit : hits) {
    GlobalPoint pos = spr::getGpos(geo, hit);
    GlobalVector v1(pos.x() - center.x(), pos.y() - center.y(), pos.z() - center.z());
    double dist = v1.mag();
    if (get2Ddist_ && (dist > 0)) {
      double cth = ((v1.x() * pTrack->px() + v1.y() * pTrack->py() + v1.z() * pTrack->pz()) / (dist * pTrack->p()));
      dist *= std::sqrt(1.0 - cth * cth);
    }
    double ener = spr::getEnergy(hit, useRaw_);
    t_DetIdHC->emplace_back(hit->id());
    t_HitEnergyHC->emplace_back(ener);
    t_HitDistHC->emplace_back(dist);
  }
}

void HcalIsoTrackStudy::TrackMap(unsigned int trkIndex,
                                 std::vector<spr::propagatedTrackDirection>& trkDirs,
                                 double dR) {
  if (trkDirs[trkIndex].okHCAL) {
    for (unsigned int indx = 0; indx < trkDirs.size(); ++indx) {
      if (trkDirs[indx].ok && trkDirs[indx].okHCAL) {
        int isConeChargedIso = spr::coneChargeIsolation(
            trkDirs[trkIndex].pointHCAL, trkDirs[indx].pointHCAL, trkDirs[trkIndex].directionHCAL, dR);
        if (isConeChargedIso == 0) {
          const reco::Track* pTrack = &(*(trkDirs[indx].trkItr));
          TVector3 point(trkDirs[indx].pointHCAL.x(), trkDirs[indx].pointHCAL.y(), trkDirs[indx].pointHCAL.z());

          t_mapP->emplace_back(pTrack->p());
          t_mapPt->emplace_back(pTrack->pt());
          t_mapEta->emplace_back(point.Eta());
          t_mapPhi->emplace_back(point.Phi());
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrackStudy);
