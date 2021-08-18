/** \class PhotonIsolationCalculator
 *  Determine and Set quality information on Photon Objects
 *
 *  \author A. Askew, N. Marinelli, M.B. Anderson
 */

#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaEcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

void PhotonIsolationCalculator::setup(const edm::ParameterSet& conf,
                                      std::vector<int> const& flagsEB,
                                      std::vector<int> const& flagsEE,
                                      std::vector<int> const& severitiesEB,
                                      std::vector<int> const& severitiesEE,
                                      edm::ConsumesCollector&& iC) {
  trackInputTag_ = iC.consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("trackProducer"));
  beamSpotProducerTag_ = iC.consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotProducer"));
  barrelecalCollection_ =
      iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("barrelEcalRecHitCollection"));
  endcapecalCollection_ =
      iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("endcapEcalRecHitCollection"));

  auto hbhetag = conf.getParameter<edm::InputTag>("HBHERecHitCollection");
  if (not hbhetag.label().empty())
    hbheRecHitsTag_ = iC.consumes<HBHERecHitCollection>(hbhetag);

  caloGeometryToken_ = decltype(caloGeometryToken_){iC.esConsumes()};
  hcalTopologyToken_ = decltype(hcalTopologyToken_){iC.esConsumes()};
  hcalChannelQualityToken_ = decltype(hcalChannelQualityToken_){iC.esConsumes(edm::ESInputTag("", "withTopo"))};
  hcalSevLvlComputerToken_ = decltype(hcalSevLvlComputerToken_){iC.esConsumes()};
  towerMapToken_ = decltype(towerMapToken_){iC.esConsumes()};

  //  gsfRecoInputTag_ = conf.getParameter<edm::InputTag>("GsfRecoCollection");
  modulePhiBoundary_ = conf.getParameter<double>("modulePhiBoundary");
  moduleEtaBoundary_ = conf.getParameter<std::vector<double>>("moduleEtaBoundary");
  //
  vetoClusteredEcalHits_ = conf.getParameter<bool>("vetoClustered");
  useNumCrystals_ = conf.getParameter<bool>("useNumCrystals");

  /// Isolation parameters for barrel and for two different cone sizes
  int i = 0;
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("isolationtrackThresholdA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("longImpactParameterA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("transImpactParameterA_Barrel"));
  trkIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceA_Barrel"));

  i = 0;
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEA_Barrel"));
  ecalIsoBarrelRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtA_Barrel"));

  hcalIsoInnerRadAEB_ = conf.getParameter<std::array<double, 7>>("HcalRecHitInnerRadiusA_Barrel");
  hcalIsoOuterRadAEB_ = conf.getParameter<std::array<double, 7>>("HcalRecHitOuterRadiusA_Barrel");

  i = 0;
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("isolationtrackThresholdB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("longImpactParameterB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("transImpactParameterB_Barrel"));
  trkIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceB_Barrel"));

  i = 0;
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEB_Barrel"));
  ecalIsoBarrelRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtB_Barrel"));

  hcalIsoInnerRadBEB_ = conf.getParameter<std::array<double, 7>>("HcalRecHitInnerRadiusB_Barrel");
  hcalIsoOuterRadBEB_ = conf.getParameter<std::array<double, 7>>("HcalRecHitOuterRadiusB_Barrel");

  /// Isolation parameters for Endcap and for two different cone sizes
  i = 0;
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("isolationtrackThresholdA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("longImpactParameterA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("transImpactParameterA_Endcap"));
  trkIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceA_Endcap"));

  i = 0;
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEA_Endcap"));
  ecalIsoEndcapRadiusA_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtA_Endcap"));

  hcalIsoInnerRadAEE_ = conf.getParameter<std::array<double, 7>>("HcalRecHitInnerRadiusA_Endcap");
  hcalIsoOuterRadAEE_ = conf.getParameter<std::array<double, 7>>("HcalRecHitOuterRadiusA_Endcap");

  i = 0;
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("TrackConeOuterRadiusB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("TrackConeInnerRadiusB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("isolationtrackThresholdB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("longImpactParameterB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("transImpactParameterB_Endcap"));
  trkIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("isolationtrackEtaSliceB_Endcap"));

  i = 0;
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitInnerRadiusB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitOuterRadiusB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitEtaSliceB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEB_Endcap"));
  ecalIsoEndcapRadiusB_[i++] = (conf.getParameter<double>("EcalRecHitThreshEtB_Endcap"));

  hcalIsoInnerRadBEE_ = conf.getParameter<std::array<double, 7>>("HcalRecHitInnerRadiusB_Endcap");
  hcalIsoOuterRadBEE_ = conf.getParameter<std::array<double, 7>>("HcalRecHitOuterRadiusB_Endcap");

  //Pick up the variables for the spike removal
  flagsEB_ = flagsEB;
  flagsEE_ = flagsEE;
  severityExclEB_ = severitiesEB;
  severityExclEE_ = severitiesEE;

  hcalIsoEThresHB_ = conf.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
  hcalIsoEThresHE_ = conf.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");
  maxHcalSeverity_ = conf.getParameter<int>("maxHcalRecHitSeverity");
}

void PhotonIsolationCalculator::calculate(const reco::Photon* pho,
                                          const edm::Event& e,
                                          const edm::EventSetup& es,
                                          reco::Photon::FiducialFlags& phofid,
                                          reco::Photon::IsolationVariables& phoisolR1,
                                          reco::Photon::IsolationVariables& phoisolR2) const {
  //Get fiducial flags. This does not really belong here
  bool isEBPho = false;
  bool isEEPho = false;
  bool isEBEtaGap = false;
  bool isEBPhiGap = false;
  bool isEERingGap = false;
  bool isEEDeeGap = false;
  bool isEBEEGap = false;
  classify(pho, isEBPho, isEEPho, isEBEtaGap, isEBPhiGap, isEERingGap, isEEDeeGap, isEBEEGap);
  phofid.isEB = isEBPho;
  phofid.isEE = isEEPho;
  phofid.isEBEtaGap = isEBEtaGap;
  phofid.isEBPhiGap = isEBPhiGap;
  phofid.isEERingGap = isEERingGap;
  phofid.isEEDeeGap = isEEDeeGap;
  phofid.isEBEEGap = isEBEEGap;

  auto const& caloGeometry = es.getData(caloGeometryToken_);
  auto const& hcalTopology = &es.getData(hcalTopologyToken_);
  auto const& hcalChannelQuality = &es.getData(hcalChannelQualityToken_);
  auto const& hcalSevLvlComputer = &es.getData(hcalSevLvlComputerToken_);
  auto const& towerMap = es.getData(towerMapToken_);

  // Calculate isolation variables. cone sizes and thresholds
  // are set for Barrel and endcap separately

  reco::SuperClusterRef scRef = pho->superCluster();
  const reco::BasicCluster& seedCluster = *(scRef->seed());
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first;
  int detector = seedXtalId.subdetId();

  //Isolation parameters variables
  double photonEcalRecHitConeInnerRadiusA_;
  double photonEcalRecHitConeOuterRadiusA_;
  double photonEcalRecHitEtaSliceA_;
  double photonEcalRecHitThreshEA_;
  double photonEcalRecHitThreshEtA_;
  double trackConeOuterRadiusA_;
  double trackConeInnerRadiusA_;
  double isolationtrackThresholdA_;
  double isolationtrackEtaSliceA_;
  double trackLipRadiusA_;
  double trackD0RadiusA_;
  double photonEcalRecHitConeInnerRadiusB_;
  double photonEcalRecHitConeOuterRadiusB_;
  double photonEcalRecHitEtaSliceB_;
  double photonEcalRecHitThreshEB_;
  double photonEcalRecHitThreshEtB_;
  double trackConeOuterRadiusB_;
  double trackConeInnerRadiusB_;
  double isolationtrackThresholdB_;
  double isolationtrackEtaSliceB_;
  double trackLipRadiusB_;
  double trackD0RadiusB_;

  if (detector == EcalBarrel) {
    trackConeOuterRadiusA_ = trkIsoBarrelRadiusA_[0];
    trackConeInnerRadiusA_ = trkIsoBarrelRadiusA_[1];
    isolationtrackThresholdA_ = trkIsoBarrelRadiusA_[2];
    trackLipRadiusA_ = trkIsoBarrelRadiusA_[3];
    trackD0RadiusA_ = trkIsoBarrelRadiusA_[4];
    isolationtrackEtaSliceA_ = trkIsoBarrelRadiusA_[5];

    photonEcalRecHitConeInnerRadiusA_ = ecalIsoBarrelRadiusA_[0];
    photonEcalRecHitConeOuterRadiusA_ = ecalIsoBarrelRadiusA_[1];
    photonEcalRecHitEtaSliceA_ = ecalIsoBarrelRadiusA_[2];
    photonEcalRecHitThreshEA_ = ecalIsoBarrelRadiusA_[3];
    photonEcalRecHitThreshEtA_ = ecalIsoBarrelRadiusA_[4];

    trackConeOuterRadiusB_ = trkIsoBarrelRadiusB_[0];
    trackConeInnerRadiusB_ = trkIsoBarrelRadiusB_[1];
    isolationtrackThresholdB_ = trkIsoBarrelRadiusB_[2];
    trackLipRadiusB_ = trkIsoBarrelRadiusB_[3];
    trackD0RadiusB_ = trkIsoBarrelRadiusB_[4];
    isolationtrackEtaSliceB_ = trkIsoBarrelRadiusB_[5];

    photonEcalRecHitConeInnerRadiusB_ = ecalIsoBarrelRadiusB_[0];
    photonEcalRecHitConeOuterRadiusB_ = ecalIsoBarrelRadiusB_[1];
    photonEcalRecHitEtaSliceB_ = ecalIsoBarrelRadiusB_[2];
    photonEcalRecHitThreshEB_ = ecalIsoBarrelRadiusB_[3];
    photonEcalRecHitThreshEtB_ = ecalIsoBarrelRadiusB_[4];
  } else {
    // detector==EcalEndcap

    trackConeOuterRadiusA_ = trkIsoEndcapRadiusA_[0];
    trackConeInnerRadiusA_ = trkIsoEndcapRadiusA_[1];
    isolationtrackThresholdA_ = trkIsoEndcapRadiusA_[2];
    trackLipRadiusA_ = trkIsoEndcapRadiusA_[3];
    trackD0RadiusA_ = trkIsoEndcapRadiusA_[4];
    isolationtrackEtaSliceA_ = trkIsoEndcapRadiusA_[5];

    photonEcalRecHitConeInnerRadiusA_ = ecalIsoEndcapRadiusA_[0];
    photonEcalRecHitConeOuterRadiusA_ = ecalIsoEndcapRadiusA_[1];
    photonEcalRecHitEtaSliceA_ = ecalIsoEndcapRadiusA_[2];
    photonEcalRecHitThreshEA_ = ecalIsoEndcapRadiusA_[3];
    photonEcalRecHitThreshEtA_ = ecalIsoEndcapRadiusA_[4];

    trackConeOuterRadiusB_ = trkIsoEndcapRadiusB_[0];
    trackConeInnerRadiusB_ = trkIsoEndcapRadiusB_[1];
    isolationtrackThresholdB_ = trkIsoEndcapRadiusB_[2];
    trackLipRadiusB_ = trkIsoEndcapRadiusB_[3];
    trackD0RadiusB_ = trkIsoEndcapRadiusB_[4];
    isolationtrackEtaSliceB_ = trkIsoEndcapRadiusB_[5];

    photonEcalRecHitConeInnerRadiusB_ = ecalIsoEndcapRadiusB_[0];
    photonEcalRecHitConeOuterRadiusB_ = ecalIsoEndcapRadiusB_[1];
    photonEcalRecHitEtaSliceB_ = ecalIsoEndcapRadiusB_[2];
    photonEcalRecHitThreshEB_ = ecalIsoEndcapRadiusB_[3];
    photonEcalRecHitThreshEtB_ = ecalIsoEndcapRadiusB_[4];
  }

  //Calculate hollow cone track isolation, CONE A
  int ntrkA = 0;
  double trkisoA = 0;
  calculateTrackIso(pho,
                    e,
                    trkisoA,
                    ntrkA,
                    isolationtrackThresholdA_,
                    trackConeOuterRadiusA_,
                    trackConeInnerRadiusA_,
                    isolationtrackEtaSliceA_,
                    trackLipRadiusA_,
                    trackD0RadiusA_);

  //Calculate solid cone track isolation, CONE A
  int sntrkA = 0;
  double strkisoA = 0;
  calculateTrackIso(pho,
                    e,
                    strkisoA,
                    sntrkA,
                    isolationtrackThresholdA_,
                    trackConeOuterRadiusA_,
                    0.,
                    isolationtrackEtaSliceA_,
                    trackLipRadiusA_,
                    trackD0RadiusA_);

  phoisolR1.nTrkHollowCone = ntrkA;
  phoisolR1.trkSumPtHollowCone = trkisoA;
  phoisolR1.nTrkSolidCone = sntrkA;
  phoisolR1.trkSumPtSolidCone = strkisoA;

  //Calculate hollow cone track isolation, CONE B
  int ntrkB = 0;
  double trkisoB = 0;
  calculateTrackIso(pho,
                    e,
                    trkisoB,
                    ntrkB,
                    isolationtrackThresholdB_,
                    trackConeOuterRadiusB_,
                    trackConeInnerRadiusB_,
                    isolationtrackEtaSliceB_,
                    trackLipRadiusB_,
                    trackD0RadiusB_);

  //Calculate solid cone track isolation, CONE B
  int sntrkB = 0;
  double strkisoB = 0;
  calculateTrackIso(pho,
                    e,
                    strkisoB,
                    sntrkB,
                    isolationtrackThresholdB_,
                    trackConeOuterRadiusB_,
                    0.,
                    isolationtrackEtaSliceB_,
                    trackLipRadiusB_,
                    trackD0RadiusB_);

  phoisolR2.nTrkHollowCone = ntrkB;
  phoisolR2.trkSumPtHollowCone = trkisoB;
  phoisolR2.nTrkSolidCone = sntrkB;
  phoisolR2.trkSumPtSolidCone = strkisoB;

  //   std::cout << "Output from solid cone track isolation: ";
  //   std::cout << " Sum pT: " << strkiso << " ntrk: " << sntrk << std::endl;

  double EcalRecHitIsoA = calculateEcalRecHitIso(pho,
                                                 e,
                                                 es,
                                                 photonEcalRecHitConeOuterRadiusA_,
                                                 photonEcalRecHitConeInnerRadiusA_,
                                                 photonEcalRecHitEtaSliceA_,
                                                 photonEcalRecHitThreshEA_,
                                                 photonEcalRecHitThreshEtA_,
                                                 vetoClusteredEcalHits_,
                                                 useNumCrystals_);
  phoisolR1.ecalRecHitSumEt = EcalRecHitIsoA;

  double EcalRecHitIsoB = calculateEcalRecHitIso(pho,
                                                 e,
                                                 es,
                                                 photonEcalRecHitConeOuterRadiusB_,
                                                 photonEcalRecHitConeInnerRadiusB_,
                                                 photonEcalRecHitEtaSliceB_,
                                                 photonEcalRecHitThreshEB_,
                                                 photonEcalRecHitThreshEtB_,
                                                 vetoClusteredEcalHits_,
                                                 useNumCrystals_);
  phoisolR2.ecalRecHitSumEt = EcalRecHitIsoB;

  if (not hbheRecHitsTag_.isUninitialized()) {
    auto const& hbheRecHits = e.get(hbheRecHitsTag_);

    auto fcone = [this,
                  pho,
                  &caloGeometry,
                  &hcalTopo = *hcalTopology,
                  &hcalQual = *hcalChannelQuality,
                  &hcalSev = *hcalSevLvlComputer,
                  &towerMap,
                  &hbheRecHits](double outer, double inner, int depth) {
      return calculateHcalRecHitIso<false>(
          pho, caloGeometry, hcalTopo, hcalQual, hcalSev, towerMap, hbheRecHits, outer, inner, depth);
    };

    auto fbc = [this,
                pho,
                &caloGeometry,
                &hcalTopo = *hcalTopology,
                &hcalQual = *hcalChannelQuality,
                &hcalSev = *hcalSevLvlComputer,
                &towerMap,
                &hbheRecHits](double outer, int depth) {
      return calculateHcalRecHitIso<true>(
          pho, caloGeometry, hcalTopo, hcalQual, hcalSev, towerMap, hbheRecHits, outer, 0., depth);
    };

    for (size_t id = 0; id < phoisolR1.hcalRecHitSumEt.size(); ++id) {
      const auto& outerA = detector == EcalBarrel ? hcalIsoOuterRadAEB_[id] : hcalIsoOuterRadAEE_[id];
      const auto& outerB = detector == EcalBarrel ? hcalIsoOuterRadBEB_[id] : hcalIsoOuterRadBEE_[id];
      const auto& innerA = detector == EcalBarrel ? hcalIsoInnerRadAEB_[id] : hcalIsoInnerRadAEE_[id];
      const auto& innerB = detector == EcalBarrel ? hcalIsoInnerRadBEB_[id] : hcalIsoInnerRadBEE_[id];

      phoisolR1.hcalRecHitSumEt[id] = fcone(outerA, innerA, id + 1);
      phoisolR2.hcalRecHitSumEt[id] = fcone(outerB, innerB, id + 1);

      phoisolR1.hcalRecHitSumEtBc[id] = fbc(outerA, id + 1);
      phoisolR2.hcalRecHitSumEtBc[id] = fbc(outerB, id + 1);
    }
  }

  phoisolR1.pre7DepthHcal = false;
  phoisolR2.pre7DepthHcal = false;
}

void PhotonIsolationCalculator::classify(const reco::Photon* photon,
                                         bool& isEBPho,
                                         bool& isEEPho,
                                         bool& isEBEtaGap,
                                         bool& isEBPhiGap,
                                         bool& isEERingGap,
                                         bool& isEEDeeGap,
                                         bool& isEBEEGap) {
  const reco::CaloCluster& seedCluster = *(photon->superCluster()->seed());
  // following line is temporary until the D. Evans or F. Ferri submit the new tag for ClusterAlgos
  // DEfinitive will be
  // DetId seedXtalId = scRef->seed()->seed();
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first;
  int detector = seedXtalId.subdetId();

  //Set fiducial flags for this photon.
  double eta = photon->superCluster()->position().eta();
  double feta = fabs(eta);

  if (detector == EcalBarrel) {
    isEBPho = true;
    if (EBDetId::isNextToEtaBoundary(EBDetId(seedXtalId))) {
      if (fabs(feta - 1.479) < .1)
        isEBEEGap = true;
      else
        isEBEtaGap = true;
    }

    if (EBDetId::isNextToPhiBoundary(EBDetId(seedXtalId)))
      isEBPhiGap = true;

  } else if (detector == EcalEndcap) {
    isEEPho = true;
    if (EEDetId::isNextToRingBoundary(EEDetId(seedXtalId))) {
      if (fabs(feta - 1.479) < .1)
        isEBEEGap = true;
      else
        isEERingGap = true;
    }

    if (EEDetId::isNextToDBoundary(EEDetId(seedXtalId)))
      isEEDeeGap = true;
  }
}

void PhotonIsolationCalculator::calculateTrackIso(const reco::Photon* photon,
                                                  const edm::Event& e,
                                                  double& trkCone,
                                                  int& ntrkCone,
                                                  double pTThresh,
                                                  double RCone,
                                                  double RinnerCone,
                                                  double etaSlice,
                                                  double lip,
                                                  double d0) const {
  ntrkCone = 0;
  trkCone = 0;
  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  e.getByToken(trackInputTag_, tracks);
  if (!tracks.isValid()) {
    return;
  }
  const reco::TrackCollection* trackCollection = tracks.product();
  //Photon Eta and Phi.  Hope these are correct.
  reco::BeamSpot vertexBeamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByToken(beamSpotProducerTag_, recoBeamSpotHandle);
  vertexBeamSpot = *recoBeamSpotHandle;

  PhotonTkIsolation phoIso(RCone,
                           RinnerCone,
                           etaSlice,
                           pTThresh,
                           lip,
                           d0,
                           trackCollection,
                           math::XYZPoint(vertexBeamSpot.x0(), vertexBeamSpot.y0(), vertexBeamSpot.z0()));

  std::pair<int, double> res = phoIso.getIso(photon);
  ntrkCone = res.first;
  trkCone = res.second;
}

double PhotonIsolationCalculator::calculateEcalRecHitIso(const reco::Photon* photon,
                                                         const edm::Event& iEvent,
                                                         const edm::EventSetup& iSetup,
                                                         double RCone,
                                                         double RConeInner,
                                                         double etaSlice,
                                                         double eMin,
                                                         double etMin,
                                                         bool vetoClusteredHits,
                                                         bool useNumXtals) const {
  edm::Handle<EcalRecHitCollection> ecalhitsCollEB;
  edm::Handle<EcalRecHitCollection> ecalhitsCollEE;
  iEvent.getByToken(endcapecalCollection_, ecalhitsCollEE);

  iEvent.getByToken(barrelecalCollection_, ecalhitsCollEB);

  const EcalRecHitCollection* rechitsCollectionEE_ = ecalhitsCollEE.product();
  const EcalRecHitCollection* rechitsCollectionEB_ = ecalhitsCollEB.product();

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);

  EgammaRecHitIsolation phoIsoEB(
      RCone, RConeInner, etaSlice, etMin, eMin, geoHandle, *rechitsCollectionEB_, sevLevel, DetId::Ecal);

  phoIsoEB.setVetoClustered(vetoClusteredHits);
  phoIsoEB.setUseNumCrystals(useNumXtals);
  phoIsoEB.doSeverityChecks(ecalhitsCollEB.product(), severityExclEB_);
  phoIsoEB.doFlagChecks(flagsEB_);
  double ecalIsolEB = phoIsoEB.getEtSum(photon);

  EgammaRecHitIsolation phoIsoEE(
      RCone, RConeInner, etaSlice, etMin, eMin, geoHandle, *rechitsCollectionEE_, sevLevel, DetId::Ecal);

  phoIsoEE.setVetoClustered(vetoClusteredHits);
  phoIsoEE.setUseNumCrystals(useNumXtals);
  phoIsoEE.doSeverityChecks(ecalhitsCollEE.product(), severityExclEE_);
  phoIsoEE.doFlagChecks(flagsEE_);

  double ecalIsolEE = phoIsoEE.getEtSum(photon);
  //  delete phoIso;
  double ecalIsol = ecalIsolEB + ecalIsolEE;

  return ecalIsol;
}

template <bool isoBC>
double PhotonIsolationCalculator::calculateHcalRecHitIso(const reco::Photon* photon,
                                                         const CaloGeometry& geometry,
                                                         const HcalTopology& hcalTopology,
                                                         const HcalChannelQuality& hcalChStatus,
                                                         const HcalSeverityLevelComputer& hcalSevLvlComputer,
                                                         const CaloTowerConstituentsMap& towerMap,
                                                         const HBHERecHitCollection& hbheRecHits,
                                                         double RCone,
                                                         double RConeInner,
                                                         int depth) const {
  const EgammaHcalIsolation::arrayHB e04{{0., 0., 0., 0.}};
  const EgammaHcalIsolation::arrayHE e07{{0., 0., 0., 0., 0., 0., 0.}};

  if constexpr (isoBC) {
    auto hcaliso = EgammaHcalIsolation(EgammaHcalIsolation::InclusionRule::withinConeAroundCluster,
                                       RCone,
                                       EgammaHcalIsolation::InclusionRule::isBehindClusterSeed,
                                       RConeInner,
                                       hcalIsoEThresHB_,
                                       e04,
                                       maxHcalSeverity_,
                                       hcalIsoEThresHE_,
                                       e07,
                                       maxHcalSeverity_,
                                       hbheRecHits,
                                       geometry,
                                       hcalTopology,
                                       hcalChStatus,
                                       hcalSevLvlComputer,
                                       towerMap);

    return hcaliso.getHcalEtSumBc(photon, depth);
  } else {
    auto hcaliso = EgammaHcalIsolation(EgammaHcalIsolation::InclusionRule::withinConeAroundCluster,
                                       RCone,
                                       EgammaHcalIsolation::InclusionRule::withinConeAroundCluster,
                                       RConeInner,
                                       hcalIsoEThresHB_,
                                       e04,
                                       maxHcalSeverity_,
                                       hcalIsoEThresHE_,
                                       e07,
                                       maxHcalSeverity_,
                                       hbheRecHits,
                                       geometry,
                                       hcalTopology,
                                       hcalChStatus,
                                       hcalSevLvlComputer,
                                       towerMap);

    return hcaliso.getHcalEtSum(photon, depth);
  }
}
