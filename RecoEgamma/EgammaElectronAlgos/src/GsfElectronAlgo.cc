#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/EgAmbiguityTools.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronTools.h"
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"


#include <Math/Point3D.h>
#include <sstream>
#include <algorithm>


using namespace edm ;
using namespace reco ;


GsfElectronAlgo::EventSetupData::EventSetupData()
 : cacheIDGeom(0), cacheIDTopo(0), cacheIDTDGeom(0), cacheIDMagField(0),
   cacheSevLevel(0), mtsTransform(nullptr), constraintAtVtx(nullptr)
 {}

void GsfElectronAlgo::EventData::retreiveOriginalTrackCollections
 ( const reco::TrackRef & ctfTrack, const reco::GsfTrackRef & gsfTrack )
 {
  if ((!originalCtfTrackCollectionRetreived)&&(ctfTrack.isNonnull()))
   {
    event->get(ctfTrack.id(),originalCtfTracks) ;
    originalCtfTrackCollectionRetreived = true ;
   }
  if ((!originalGsfTrackCollectionRetreived)&&(gsfTrack.isNonnull()))
   {
    event->get(gsfTrack.id(),originalGsfTracks) ;
    originalGsfTrackCollectionRetreived = true ;
   }
 }


GsfElectronAlgo::ElectronData::ElectronData
 ( const reco::GsfElectronCoreRef & core,
   const reco::BeamSpot & bs )
 : coreRef(core),
   gsfTrackRef(coreRef->gsfTrack()),
   superClusterRef(coreRef->superCluster()),
   ctfTrackRef(coreRef->ctfTrack()), shFracInnerHits(coreRef->ctfGsfOverlap()),
   beamSpot(bs)
 {}


void GsfElectronAlgo::ElectronData::computeCharge
 ( int & charge, GsfElectron::ChargeInfo & info )
 {
  // determine charge from SC
  GlobalPoint orig, scpos ;
  ele_convert(beamSpot.position(),orig) ;
  ele_convert(superClusterRef->position(),scpos) ;
  GlobalVector scvect(scpos-orig) ;
  GlobalPoint inntkpos = innTSOS.globalPosition() ;
  GlobalVector inntkvect = GlobalVector(inntkpos-orig) ;
  float dPhiInnEle=normalizedPhi(scvect.barePhi()-inntkvect.barePhi()) ;
  if(dPhiInnEle>0) info.scPixCharge = -1 ;
  else info.scPixCharge = 1 ;

  // flags
  int chargeGsf = gsfTrackRef->charge() ;
  info.isGsfScPixConsistent = ((chargeGsf*info.scPixCharge)>0) ;
  info.isGsfCtfConsistent = (ctfTrackRef.isNonnull()&&((chargeGsf*ctfTrackRef->charge())>0)) ;
  info.isGsfCtfScPixConsistent = (info.isGsfScPixConsistent&&info.isGsfCtfConsistent) ;

  // default charge
  if (info.isGsfScPixConsistent||ctfTrackRef.isNull())
   { charge = info.scPixCharge ; }
  else
   { charge = ctfTrackRef->charge() ; }
 }

CaloClusterPtr GsfElectronAlgo::ElectronData::getEleBasicCluster( MultiTrajectoryStateTransform const& mtsTransform )
 {
  CaloClusterPtr eleRef ;
  TrajectoryStateOnSurface tempTSOS ;
  TrajectoryStateOnSurface outTSOS = mtsTransform.outerStateOnSurface(*gsfTrackRef) ;
  float dphimin = 1.e30 ;
  for(auto const& bc : superClusterRef->clusters())
   {
    GlobalPoint posclu(bc->position().x(),bc->position().y(),bc->position().z()) ;
    tempTSOS = mtsTransform.extrapolatedState(outTSOS,posclu) ;
    if (!tempTSOS.isValid()) tempTSOS=outTSOS ;
    GlobalPoint extrap = tempTSOS.globalPosition() ;
    float dphi = EleRelPointPair(posclu,extrap,beamSpot.position()).dPhi() ;
    if (std::abs(dphi)<dphimin)
     {
      dphimin = std::abs(dphi) ;
      eleRef = bc;
      eleTSOS = tempTSOS ;
     }
   }
  return eleRef ;
 }

bool GsfElectronAlgo::ElectronData::calculateTSOS
 ( MultiTrajectoryStateTransform const& mtsTransform, GsfConstraintAtVertex const& constraintAtVtx )
 {
  //at innermost point
  innTSOS = mtsTransform.innerStateOnSurface(*gsfTrackRef);
  if (!innTSOS.isValid()) return false;

  //at vertex
  // innermost state propagation to the beam spot position
  GlobalPoint bsPos ;
  ele_convert(beamSpot.position(),bsPos) ;
  vtxTSOS = mtsTransform.extrapolatedState(innTSOS,bsPos) ;
  if (!vtxTSOS.isValid()) vtxTSOS=innTSOS;

  //at seed
  outTSOS = mtsTransform.outerStateOnSurface(*gsfTrackRef);
  if (!outTSOS.isValid()) return false;

  //    TrajectoryStateOnSurface seedTSOS
  seedTSOS = mtsTransform.extrapolatedState(outTSOS,
           GlobalPoint(superClusterRef->seed()->position().x(),
               superClusterRef->seed()->position().y(),
                 superClusterRef->seed()->position().z()));
  if (!seedTSOS.isValid()) seedTSOS=outTSOS;

  // at scl
  sclTSOS = mtsTransform.extrapolatedState(innTSOS,GlobalPoint(superClusterRef->x(),superClusterRef->y(),superClusterRef->z()));
  if (!sclTSOS.isValid()) sclTSOS=outTSOS;

  // constrained momentum
  constrainedVtxTSOS = constraintAtVtx.constrainAtBeamSpot(*gsfTrackRef,beamSpot);

  return true ;
 }

void GsfElectronAlgo::ElectronData::calculateMode()
 {
  multiTrajectoryStateMode::momentumFromModeCartesian(innTSOS,innMom) ;
  multiTrajectoryStateMode::positionFromModeCartesian(innTSOS,innPos) ;
  multiTrajectoryStateMode::momentumFromModeCartesian(seedTSOS,seedMom) ;
  multiTrajectoryStateMode::positionFromModeCartesian(seedTSOS,seedPos) ;
  multiTrajectoryStateMode::momentumFromModeCartesian(eleTSOS,eleMom) ;
  multiTrajectoryStateMode::positionFromModeCartesian(eleTSOS,elePos) ;
  multiTrajectoryStateMode::momentumFromModeCartesian(sclTSOS,sclMom) ;
  multiTrajectoryStateMode::positionFromModeCartesian(sclTSOS,sclPos) ;
  multiTrajectoryStateMode::momentumFromModeCartesian(vtxTSOS,vtxMom) ;
  multiTrajectoryStateMode::positionFromModeCartesian(vtxTSOS,vtxPos) ;
  multiTrajectoryStateMode::momentumFromModeCartesian(outTSOS,outMom);
  multiTrajectoryStateMode::positionFromModeCartesian(outTSOS,outPos) ;
  multiTrajectoryStateMode::momentumFromModeCartesian(constrainedVtxTSOS,vtxMomWithConstraint);
 }

Candidate::LorentzVector GsfElectronAlgo::ElectronData::calculateMomentum()
 {
  double scale = superClusterRef->energy()/vtxMom.mag() ;
  return Candidate::LorentzVector
   ( vtxMom.x()*scale,vtxMom.y()*scale,vtxMom.z()*scale,
     superClusterRef->energy() ) ;
 }

void GsfElectronAlgo::calculateSaturationInfo(const reco::SuperClusterRef& theClus,
                                              reco::GsfElectron::SaturationInfo& si,
                                              EventData const& eventData)
{
  const reco::CaloCluster & seedCluster = *(theClus->seed()) ;
  DetId seedXtalId = seedCluster.seed();
  int detector = seedXtalId.subdetId();
  const EcalRecHitCollection* ecalRecHits = nullptr ;
  if (detector==EcalBarrel)
    ecalRecHits = eventData.barrelRecHits.product() ;
  else
    ecalRecHits = eventData.endcapRecHits.product() ;
  
  int nSaturatedXtals = 0;
  bool isSeedSaturated = false;
  for (auto&& hitFractionPair : theClus->hitsAndFractions()) {    
    auto&& ecalRecHit = ecalRecHits->find(hitFractionPair.first);
    if (ecalRecHit == ecalRecHits->end()) continue;
    if (ecalRecHit->checkFlag(EcalRecHit::Flags::kSaturated)) {
      nSaturatedXtals++;
      if (seedXtalId == ecalRecHit->detid())
	isSeedSaturated = true;
    }
  }
  si.nSaturatedXtals = nSaturatedXtals;
  si.isSeedSaturated = isSeedSaturated;

}

template<bool full5x5>
void GsfElectronAlgo::calculateShowerShape( const reco::SuperClusterRef & theClus,
                                            ElectronHcalHelper const& hcalHelper,
                                            reco::GsfElectron::ShowerShape & showerShape,
                                            EventData const& eventData)
 {

  using ClusterTools = EcalClusterToolsT<full5x5>;

  const reco::CaloCluster & seedCluster = *(theClus->seed()) ;
  // temporary, till CaloCluster->seed() is made available
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;
  int detector = seedXtalId.subdetId() ;

  const CaloTopology * topology = eventSetupData_.caloTopo.product() ;
  const CaloGeometry * geometry = eventSetupData_.caloGeom.product() ;
  const EcalRecHitCollection * recHits = nullptr ;
  std::vector<int> recHitFlagsToBeExcluded ;
  std::vector<int> recHitSeverityToBeExcluded ;
  if (detector==EcalBarrel)
   {
    recHits = eventData.barrelRecHits.product() ;
    recHitFlagsToBeExcluded = generalData_.recHitsCfg.recHitFlagsToBeExcludedBarrel ;
    recHitSeverityToBeExcluded = generalData_.recHitsCfg.recHitSeverityToBeExcludedBarrel ;
   }
  else
   {
    recHits = eventData.endcapRecHits.product() ;
    recHitFlagsToBeExcluded = generalData_.recHitsCfg.recHitFlagsToBeExcludedEndcaps ;
    recHitSeverityToBeExcluded = generalData_.recHitsCfg.recHitSeverityToBeExcludedEndcaps ;
   }

  std::vector<float> covariances = ClusterTools::covariances(seedCluster,recHits,topology,geometry) ;
  std::vector<float> localCovariances = ClusterTools::localCovariances(seedCluster,recHits,topology) ;
  showerShape.sigmaEtaEta = sqrt(covariances[0]) ;
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]) ;
  if (!edm::isNotFinite(localCovariances[2])) showerShape.sigmaIphiIphi = sqrt(localCovariances[2]) ;
  showerShape.e1x5 = ClusterTools::e1x5(seedCluster,recHits,topology)  ;
  showerShape.e2x5Max = ClusterTools::e2x5Max(seedCluster,recHits,topology)  ;
  showerShape.e5x5 = ClusterTools::e5x5(seedCluster,recHits,topology) ;
  showerShape.r9 = ClusterTools::e3x3(seedCluster,recHits,topology)/theClus->rawEnergy() ;

  const float scale = full5x5 ? showerShape.e5x5 : theClus->energy();

  showerShape.hcalDepth1OverEcal = hcalHelper.hcalESumDepth1(*theClus)/theClus->energy() ;
  showerShape.hcalDepth2OverEcal = hcalHelper.hcalESumDepth2(*theClus)/theClus->energy() ;
  showerShape.hcalTowersBehindClusters = hcalHelper.hcalTowersBehindClusters(*theClus) ;
  showerShape.hcalDepth1OverEcalBc = hcalHelper.hcalESumDepth1BehindClusters(showerShape.hcalTowersBehindClusters)/scale ;
  showerShape.hcalDepth2OverEcalBc = hcalHelper.hcalESumDepth2BehindClusters(showerShape.hcalTowersBehindClusters)/scale ;
  showerShape.invalidHcal = (showerShape.hcalDepth1OverEcalBc == 0 && 
                             showerShape.hcalDepth2OverEcalBc == 0 &&
                             !hcalHelper.hasActiveHcal(*theClus));
  
  // extra shower shapes
  const float see_by_spp = showerShape.sigmaIetaIeta*showerShape.sigmaIphiIphi;
  if(  see_by_spp > 0 ) {
    showerShape.sigmaIetaIphi = localCovariances[1] / see_by_spp;
  } else if ( localCovariances[1] > 0 ) {
    showerShape.sigmaIetaIphi = 1.f;
  } else {
    showerShape.sigmaIetaIphi = -1.f;
  }
  showerShape.eMax          = ClusterTools::eMax(seedCluster,recHits);
  showerShape.e2nd          = ClusterTools::e2nd(seedCluster,recHits);
  showerShape.eTop          = ClusterTools::eTop(seedCluster,recHits,topology);
  showerShape.eLeft         = ClusterTools::eLeft(seedCluster,recHits,topology);
  showerShape.eRight        = ClusterTools::eRight(seedCluster,recHits,topology);
  showerShape.eBottom       = ClusterTools::eBottom(seedCluster,recHits,topology);

  showerShape.e2x5Left = ClusterTools::e2x5Left(seedCluster,recHits,topology);
  showerShape.e2x5Right = ClusterTools::e2x5Right(seedCluster,recHits,topology);
  showerShape.e2x5Top = ClusterTools::e2x5Top(seedCluster,recHits,topology);
  showerShape.e2x5Bottom = ClusterTools::e2x5Bottom(seedCluster,recHits,topology);
 }


//===================================================================
// GsfElectronAlgo
//===================================================================

GsfElectronAlgo::GsfElectronAlgo
 ( const InputTagsConfiguration & inputCfg,
   const StrategyConfiguration & strategyCfg,
   const CutsConfiguration & cutsCfg,
   const CutsConfiguration & cutsCfgPflow,
   const ElectronHcalHelper::Configuration & hcalCfg,
   const ElectronHcalHelper::Configuration & hcalCfgPflow,
   const IsolationConfiguration & isoCfg,
   const EcalRecHitsConfiguration & recHitsCfg,
   EcalClusterFunctionBaseClass * superClusterErrorFunction,
   EcalClusterFunctionBaseClass * crackCorrectionFunction,
   const RegressionHelper::Configuration & regCfg,
   const edm::ParameterSet& tkIsol03Cfg,
   const edm::ParameterSet& tkIsol04Cfg,
   const edm::ParameterSet& tkIsolHEEP03Cfg,
   const edm::ParameterSet& tkIsolHEEP04Cfg

 )
: generalData_{inputCfg,strategyCfg,cutsCfg,cutsCfgPflow,isoCfg,recHitsCfg,hcalCfg,hcalCfgPflow,superClusterErrorFunction,crackCorrectionFunction,regCfg},
   eventSetupData_{},
   tkIsol03Calc_(tkIsol03Cfg),tkIsol04Calc_(tkIsol04Cfg),
   tkIsolHEEP03Calc_(tkIsolHEEP03Cfg),tkIsolHEEP04Calc_(tkIsolHEEP04Cfg)
  
 {}

void GsfElectronAlgo::checkSetup( const edm::EventSetup & es )
 {
  // get EventSetupRecords if needed
  const bool updateField = eventSetupData_.cacheIDMagField!=es.get<IdealMagneticFieldRecord>().cacheIdentifier();
  if (updateField){
    eventSetupData_.cacheIDMagField=es.get<IdealMagneticFieldRecord>().cacheIdentifier();
    es.get<IdealMagneticFieldRecord>().get(eventSetupData_.magField);
  }

  const bool updateGeometry = eventSetupData_.cacheIDTDGeom!=es.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  if (updateGeometry){
    eventSetupData_.cacheIDTDGeom=es.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    es.get<TrackerDigiGeometryRecord>().get(eventSetupData_.trackerHandle);
  }

  if ( updateField || updateGeometry ) {
    eventSetupData_.mtsTransform = std::make_unique<MultiTrajectoryStateTransform>(eventSetupData_.trackerHandle.product(),eventSetupData_.magField.product());
    eventSetupData_.constraintAtVtx = std::make_unique<GsfConstraintAtVertex>(es) ;
  }

  if (eventSetupData_.cacheIDGeom!=es.get<CaloGeometryRecord>().cacheIdentifier()){
    eventSetupData_.cacheIDGeom=es.get<CaloGeometryRecord>().cacheIdentifier();
    es.get<CaloGeometryRecord>().get(eventSetupData_.caloGeom);
  }

  if (eventSetupData_.cacheIDTopo!=es.get<CaloTopologyRecord>().cacheIdentifier()){
    eventSetupData_.cacheIDTopo=es.get<CaloTopologyRecord>().cacheIdentifier();
    es.get<CaloTopologyRecord>().get(eventSetupData_.caloTopo);
  }

  generalData_.hcalHelper.checkSetup(es) ;
  generalData_.hcalHelperPflow.checkSetup(es) ;
  if(generalData_.strategyCfg.useEcalRegression || generalData_.strategyCfg.useCombinationRegression)
    generalData_.regHelper.checkSetup(es);


  if (generalData_.superClusterErrorFunction)
   { generalData_.superClusterErrorFunction->init(es) ; }
  if (generalData_.crackCorrectionFunction)
   { generalData_.crackCorrectionFunction->init(es) ; }

  if(eventSetupData_.cacheSevLevel != es.get<EcalSeverityLevelAlgoRcd>().cacheIdentifier()){
    eventSetupData_.cacheSevLevel = es.get<EcalSeverityLevelAlgoRcd>().cacheIdentifier();
    es.get<EcalSeverityLevelAlgoRcd>().get(eventSetupData_.sevLevel);
  }
 }


GsfElectronAlgo::EventData GsfElectronAlgo::beginEvent( edm::Event const& event )
 {
  // prepare access to hcal data
  generalData_.hcalHelper.readEvent(event) ;
  generalData_.hcalHelperPflow.readEvent(event) ;

  auto const& towers = event.get(generalData_.inputCfg.hcalTowersTag) ;

  // Isolation algos
  float egHcalIsoConeSizeOutSmall=0.3, egHcalIsoConeSizeOutLarge=0.4;
  float egHcalIsoConeSizeIn=generalData_.isoCfg.intRadiusHcal,egHcalIsoPtMin=generalData_.isoCfg.etMinHcal;
  int egHcalDepth1=1, egHcalDepth2=2;

  float egIsoConeSizeOutSmall=0.3, egIsoConeSizeOutLarge=0.4, egIsoJurassicWidth=generalData_.isoCfg.jurassicWidth;
  float egIsoPtMinBarrel=generalData_.isoCfg.etMinBarrel,egIsoEMinBarrel=generalData_.isoCfg.eMinBarrel, egIsoConeSizeInBarrel=generalData_.isoCfg.intRadiusEcalBarrel;
  float egIsoPtMinEndcap=generalData_.isoCfg.etMinEndcaps,egIsoEMinEndcap=generalData_.isoCfg.eMinEndcaps, egIsoConeSizeInEndcap=generalData_.isoCfg.intRadiusEcalEndcaps;

  auto barrelRecHits = event.getHandle(generalData_.inputCfg.barrelRecHitCollection);
  auto endcapRecHits = event.getHandle(generalData_.inputCfg.endcapRecHitCollection);

  EventData eventData{
      .event             = &event,
      .beamspot          = &event.get(generalData_.inputCfg.beamSpotTag),
      .previousElectrons = event.getHandle(generalData_.inputCfg.previousGsfElectrons),
      .pflowElectrons    = event.getHandle(generalData_.inputCfg.pflowGsfElectronsTag),
      .coreElectrons     = event.getHandle(generalData_.inputCfg.gsfElectronCores),
      .barrelRecHits     = barrelRecHits,
      .endcapRecHits     = endcapRecHits,
      .currentCtfTracks  = event.getHandle(generalData_.inputCfg.ctfTracks),
      .seeds             = event.getHandle(generalData_.inputCfg.seedsTag),
      .gsfPfRecTracks    = generalData_.strategyCfg.useGsfPfRecTracks ? event.getHandle(generalData_.inputCfg.gsfPfRecTracksTag) : edm::Handle<reco::GsfPFRecTrackCollection>{},
      .vertices          = event.getHandle(generalData_.inputCfg.vtxCollectionTag),
      .conversions       = generalData_.strategyCfg.fillConvVtxFitProb ? event.getHandle(generalData_.inputCfg.conversions) : edm::Handle<reco::ConversionCollection>(),
      .hadDepth1Isolation03 = EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,&towers),
      .hadDepth1Isolation04 = EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,&towers),
      .hadDepth2Isolation03 = EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,&towers),
      .hadDepth2Isolation04 = EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,&towers),
      .hadDepth1Isolation03Bc = EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,0.,egHcalIsoPtMin,egHcalDepth1,&towers),
      .hadDepth1Isolation04Bc = EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,0.,egHcalIsoPtMin,egHcalDepth1,&towers),
      .hadDepth2Isolation03Bc = EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,0.,egHcalIsoPtMin,egHcalDepth2,&towers),
      .hadDepth2Isolation04Bc = EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,0.,egHcalIsoPtMin,egHcalDepth2,&towers),
      .ecalBarrelIsol03 = EgammaRecHitIsolation(egIsoConeSizeOutSmall,egIsoConeSizeInBarrel,egIsoJurassicWidth,egIsoPtMinBarrel,egIsoEMinBarrel,eventSetupData_.caloGeom,*barrelRecHits,eventSetupData_.sevLevel.product(),DetId::Ecal),
      .ecalBarrelIsol04 = EgammaRecHitIsolation(egIsoConeSizeOutLarge,egIsoConeSizeInBarrel,egIsoJurassicWidth,egIsoPtMinBarrel,egIsoEMinBarrel,eventSetupData_.caloGeom,*barrelRecHits,eventSetupData_.sevLevel.product(),DetId::Ecal),
      .ecalEndcapIsol03 = EgammaRecHitIsolation(egIsoConeSizeOutSmall,egIsoConeSizeInEndcap,egIsoJurassicWidth,egIsoPtMinEndcap,egIsoEMinEndcap,eventSetupData_.caloGeom,*endcapRecHits,eventSetupData_.sevLevel.product(),DetId::Ecal),
      .ecalEndcapIsol04 = EgammaRecHitIsolation(egIsoConeSizeOutLarge,egIsoConeSizeInEndcap,egIsoJurassicWidth,egIsoPtMinEndcap,egIsoEMinEndcap,eventSetupData_.caloGeom,*endcapRecHits,eventSetupData_.sevLevel.product(),DetId::Ecal)
  } ;

  eventData.ecalBarrelIsol03.setUseNumCrystals(generalData_.isoCfg.useNumCrystals);
  eventData.ecalBarrelIsol03.setVetoClustered(generalData_.isoCfg.vetoClustered);
  eventData.ecalBarrelIsol03.doSeverityChecks(eventData.barrelRecHits.product(),generalData_.recHitsCfg.recHitSeverityToBeExcludedBarrel);
  eventData.ecalBarrelIsol03.doFlagChecks(generalData_.recHitsCfg.recHitFlagsToBeExcludedBarrel);
  eventData.ecalBarrelIsol04.setUseNumCrystals(generalData_.isoCfg.useNumCrystals);
  eventData.ecalBarrelIsol04.setVetoClustered(generalData_.isoCfg.vetoClustered);
  eventData.ecalBarrelIsol04.doSeverityChecks(eventData.barrelRecHits.product(),generalData_.recHitsCfg.recHitSeverityToBeExcludedBarrel);
  eventData.ecalBarrelIsol04.doFlagChecks(generalData_.recHitsCfg.recHitFlagsToBeExcludedBarrel);
  eventData.ecalEndcapIsol03.setUseNumCrystals(generalData_.isoCfg.useNumCrystals);
  eventData.ecalEndcapIsol03.setVetoClustered(generalData_.isoCfg.vetoClustered);
  eventData.ecalEndcapIsol03.doSeverityChecks(eventData.endcapRecHits.product(),generalData_.recHitsCfg.recHitSeverityToBeExcludedEndcaps);
  eventData.ecalEndcapIsol03.doFlagChecks(generalData_.recHitsCfg.recHitFlagsToBeExcludedEndcaps);
  eventData.ecalEndcapIsol04.setUseNumCrystals(generalData_.isoCfg.useNumCrystals);
  eventData.ecalEndcapIsol04.setVetoClustered(generalData_.isoCfg.vetoClustered);
  eventData.ecalEndcapIsol04.doSeverityChecks(eventData.endcapRecHits.product(),generalData_.recHitsCfg.recHitSeverityToBeExcludedEndcaps);
  eventData.ecalEndcapIsol04.doFlagChecks(generalData_.recHitsCfg.recHitFlagsToBeExcludedEndcaps);
  
  return eventData;
 }


void GsfElectronAlgo::completeElectrons( reco::GsfElectronCollection & electrons,
                                         edm::Event const& event,
                                         edm::EventSetup const& eventSetup,
                                         const gsfAlgoHelpers::HeavyObjectCache* hoc )
 {
  checkSetup(eventSetup);
  auto eventData = beginEvent(event);

  const GsfElectronCoreCollection * coreCollection = eventData.coreElectrons.product() ;
  for ( unsigned int i=0 ; i<coreCollection->size() ; ++i )
   {
    // check there is no existing electron with this core
    const GsfElectronCoreRef coreRef = edm::Ref<GsfElectronCoreCollection>(eventData.coreElectrons,i) ;
    bool coreFound = false ;
    for(auto const& ele : electrons)
     {
      if (ele.core()==coreRef)
       {
        coreFound = true ;
        break ;
       }
     }
    if (coreFound) continue ;

    // check there is a super-cluster
    if (coreRef->superCluster().isNull()) continue ;

    // prepare internal structure for electron specific data
    ElectronData electronData(coreRef,*eventData.beamspot) ;

    // calculate and check Trajectory StatesOnSurface....
    if ( !electronData.calculateTSOS( *eventSetupData_.mtsTransform, *eventSetupData_.constraintAtVtx ) ) continue ;

    createElectron(electrons, electronData, eventData, hoc) ;

   } // loop over tracks

 }


void GsfElectronAlgo::setCutBasedPreselectionFlag( GsfElectron & ele, const reco::BeamSpot & bs )
 {
  // default value
  ele.setPassCutBasedPreselection(false) ;

  // kind of seeding
  bool eg = ele.core()->ecalDrivenSeed() ;
  bool pf = ele.core()->trackerDrivenSeed() && !ele.core()->ecalDrivenSeed() ;
  bool gedMode = generalData_.strategyCfg.gedElectronMode;
  if (eg&&pf) { throw cms::Exception("GsfElectronAlgo|BothEcalAndPureTrackerDriven")<<"An electron cannot be both egamma and purely pflow" ; }
  if ((!eg)&&(!pf)) { throw cms::Exception("GsfElectronAlgo|NeitherEcalNorPureTrackerDriven")<<"An electron cannot be neither egamma nor purely pflow" ; }

  const CutsConfiguration * cfg = ((eg||gedMode)?&generalData_.cutsCfg:&generalData_.cutsCfgPflow);

  // Et cut
  double etaValue = EleRelPoint(ele.superCluster()->position(),bs.position()).eta() ;
  double etValue = ele.superCluster()->energy()/cosh(etaValue) ;
  LogTrace("GsfElectronAlgo") << "Et : " << etValue ;
  if (ele.isEB() && (etValue < cfg->minSCEtBarrel)) return ;
  if (ele.isEE() && (etValue < cfg->minSCEtEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Et criteria are satisfied";

  // E/p cut
  double eopValue = ele.eSuperClusterOverP() ;
  LogTrace("GsfElectronAlgo") << "E/p : " << eopValue ;
  if (ele.isEB() && (eopValue > cfg->maxEOverPBarrel)) return ;
  if (ele.isEE() && (eopValue > cfg->maxEOverPEndcaps)) return ;
  if (ele.isEB() && (eopValue < cfg->minEOverPBarrel)) return ;
  if (ele.isEE() && (eopValue < cfg->minEOverPEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "E/p criteria are satisfied";

  // HoE cuts
  LogTrace("GsfElectronAlgo") << "HoE1 : " << ele.hcalDepth1OverEcal() << ", HoE2 : " << ele.hcalDepth2OverEcal();
  double hoeCone = ele.hcalOverEcal();
  double hoeTower = ele.hcalOverEcalBc();
  const reco::CaloCluster & seedCluster = *(ele.superCluster()->seed()) ;
  int detector = seedCluster.hitsAndFractions()[0].first.subdetId() ;
  bool HoEveto = false ;
  double scle = ele.superCluster()->energy();

  if (detector==EcalBarrel) HoEveto =
      hoeCone*scle<cfg->maxHBarrelCone || hoeTower*scle<cfg->maxHBarrelTower ||
     hoeCone<cfg->maxHOverEBarrelCone || hoeTower<cfg->maxHOverEBarrelTower;
  else if (detector==EcalEndcap) HoEveto =
      hoeCone*scle<cfg->maxHEndcapsCone || hoeTower*scle<cfg->maxHEndcapsTower ||
     hoeCone<cfg->maxHOverEEndcapsCone || hoeTower<cfg->maxHOverEEndcapsTower;

  if ( !HoEveto ) return ;
  LogTrace("GsfElectronAlgo") << "H/E criteria are satisfied";

  // delta eta criteria
  double deta = ele.deltaEtaSuperClusterTrackAtVtx() ;
  LogTrace("GsfElectronAlgo") << "delta eta : " << deta ;
  if (ele.isEB() && (std::abs(deta) > cfg->maxDeltaEtaBarrel)) return ;
  if (ele.isEE() && (std::abs(deta) > cfg->maxDeltaEtaEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Delta eta criteria are satisfied";

  // delta phi criteria
  double dphi = ele.deltaPhiSuperClusterTrackAtVtx();
  LogTrace("GsfElectronAlgo") << "delta phi : " << dphi;
  if (ele.isEB() && (std::abs(dphi) > cfg->maxDeltaPhiBarrel)) return ;
  if (ele.isEE() && (std::abs(dphi) > cfg->maxDeltaPhiEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Delta phi criteria are satisfied";

  // sigma ieta ieta
  LogTrace("GsfElectronAlgo") << "sigma ieta ieta : " << ele.sigmaIetaIeta();
  if (ele.isEB() && (ele.sigmaIetaIeta() > cfg->maxSigmaIetaIetaBarrel)) return ;
  if (ele.isEE() && (ele.sigmaIetaIeta() > cfg->maxSigmaIetaIetaEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Sigma ieta ieta criteria are satisfied";

  // fiducial
  if (!ele.isEB() && cfg->isBarrel) return ;
  if (!ele.isEE() && cfg->isEndcaps) return ;
  if (cfg->isFiducial && (ele.isEBEEGap()||ele.isEBEtaGap()||ele.isEBPhiGap()||ele.isEERingGap()||ele.isEEDeeGap())) return ;
  LogTrace("GsfElectronAlgo") << "Fiducial flags criteria are satisfied";

  // seed in TEC
  edm::RefToBase<TrajectorySeed> seed = ele.gsfTrack()->extra()->seedRef() ;
  ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>() ;
  if (eg && !generalData_.cutsCfg.seedFromTEC)
   {
    if (elseed.isNull())
     { throw cms::Exception("GsfElectronAlgo|NotElectronSeed")<<"The GsfTrack seed is not an ElectronSeed ?!" ; }
    else
     { if (elseed->subDet2()==6) return ; }
   }

  // transverse impact parameter
  if (std::abs(ele.gsfTrack()->dxy(bs.position()))>cfg->maxTIP) return ;
  LogTrace("GsfElectronAlgo") << "TIP criterion is satisfied" ;

  LogTrace("GsfElectronAlgo") << "All cut based criteria are satisfied" ;
  ele.setPassCutBasedPreselection(true) ;
 }


void GsfElectronAlgo::createElectron(reco::GsfElectronCollection & electrons, ElectronData & electronData, EventData & eventData, const gsfAlgoHelpers::HeavyObjectCache* hoc)
 {
  // eventually check ctf track
  if (generalData_.strategyCfg.ctfTracksCheck && electronData.ctfTrackRef.isNull()) {
    electronData.ctfTrackRef = gsfElectronTools::getClosestCtfToGsf( electronData.gsfTrackRef,
                                                                       eventData.currentCtfTracks ).first;
  }

  // charge ID
  int eleCharge ;
  GsfElectron::ChargeInfo eleChargeInfo ;
  electronData.computeCharge(eleCharge,eleChargeInfo) ;

  // electron basic cluster
  CaloClusterPtr elbcRef = electronData.getEleBasicCluster(*eventSetupData_.mtsTransform) ;

  // Seed cluster
  const reco::CaloCluster & seedCluster = *(electronData.superClusterRef->seed()) ;

  // seed Xtal
  // temporary, till CaloCluster->seed() is made available
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;

  electronData.calculateMode() ;


  //====================================================
  // Candidate attributes
  //====================================================

  Candidate::LorentzVector momentum = electronData.calculateMomentum() ;


  //====================================================
  // Track-Cluster Matching
  //====================================================

  reco::GsfElectron::TrackClusterMatching tcMatching ;
  tcMatching.electronCluster = elbcRef ;
  tcMatching.eSuperClusterOverP = (electronData.vtxMom.mag()>0)?(electronData.superClusterRef->energy()/electronData.vtxMom.mag()):(-1.) ;
  tcMatching.eSeedClusterOverP = (electronData.vtxMom.mag()>0.)?(seedCluster.energy()/electronData.vtxMom.mag()):(-1) ;
  tcMatching.eSeedClusterOverPout = (electronData.seedMom.mag()>0.)?(seedCluster.energy()/electronData.seedMom.mag()):(-1.) ;
  tcMatching.eEleClusterOverPout = (electronData.eleMom.mag()>0.)?(elbcRef->energy()/electronData.eleMom.mag()):(-1.) ;

  EleRelPointPair scAtVtx(electronData.superClusterRef->position(),electronData.sclPos,eventData.beamspot->position()) ;
  tcMatching.deltaEtaSuperClusterAtVtx = scAtVtx.dEta() ;
  tcMatching.deltaPhiSuperClusterAtVtx = scAtVtx.dPhi() ;

  EleRelPointPair seedAtCalo(seedCluster.position(),electronData.seedPos,eventData.beamspot->position()) ;
  tcMatching.deltaEtaSeedClusterAtCalo = seedAtCalo.dEta() ;
  tcMatching.deltaPhiSeedClusterAtCalo = seedAtCalo.dPhi() ;

  EleRelPointPair ecAtCalo(elbcRef->position(),electronData.elePos,eventData.beamspot->position()) ;
  tcMatching.deltaEtaEleClusterAtCalo = ecAtCalo.dEta() ;
  tcMatching.deltaPhiEleClusterAtCalo = ecAtCalo.dPhi() ;


  //=======================================================
  // Track extrapolations
  //=======================================================

  reco::GsfElectron::TrackExtrapolations tkExtra ;
  ele_convert(electronData.vtxPos,tkExtra.positionAtVtx) ;
  ele_convert(electronData.sclPos,tkExtra.positionAtCalo) ;
  ele_convert(electronData.vtxMom,tkExtra.momentumAtVtx) ;
  ele_convert(electronData.sclMom,tkExtra.momentumAtCalo) ;
  ele_convert(electronData.seedMom,tkExtra.momentumOut) ;
  ele_convert(electronData.eleMom,tkExtra.momentumAtEleClus) ;
  ele_convert(electronData.vtxMomWithConstraint,tkExtra.momentumAtVtxWithConstraint) ;


  //=======================================================
  // Closest Ctf Track
  //=======================================================

  reco::GsfElectron::ClosestCtfTrack ctfInfo ;
  ctfInfo.ctfTrack = electronData.ctfTrackRef  ;
  ctfInfo.shFracInnerHits = electronData.shFracInnerHits ;


  //====================================================
  // FiducialFlags, using nextToBoundary definition of gaps
  //====================================================

  reco::GsfElectron::FiducialFlags fiducialFlags ;
  int region   = seedXtalId.det();
  int detector = seedXtalId.subdetId() ;
  double feta=std::abs(electronData.superClusterRef->position().eta()) ;
  if (detector==EcalBarrel)
   {
    fiducialFlags.isEB = true ;
    EBDetId ebdetid(seedXtalId);
    if (EBDetId::isNextToEtaBoundary(ebdetid))
     {
      if (ebdetid.ietaAbs()==85)
       { fiducialFlags.isEBEEGap = true ; }
      else
       { fiducialFlags.isEBEtaGap = true ; }
     }
    if (EBDetId::isNextToPhiBoundary(ebdetid))
     { fiducialFlags.isEBPhiGap = true ; }
   }
  else if (detector==EcalEndcap)
   {
    fiducialFlags.isEE = true ;
    EEDetId eedetid(seedXtalId);
    if (EEDetId::isNextToRingBoundary(eedetid))
     {
      if (std::abs(feta)<2.)
       { fiducialFlags.isEBEEGap = true ; }
      else
       { fiducialFlags.isEERingGap = true ; }
     }
    if (EEDetId::isNextToDBoundary(eedetid))
     { fiducialFlags.isEEDeeGap = true ; }
   }
  else if ( EcalTools::isHGCalDet((DetId::Detector)region) )
   {
    fiducialFlags.isEE = true ;
    //HGCalDetId eeDetid(seedXtalId);    
    // fill in fiducial information when we know how to use it...
   }
  else
   { throw cms::Exception("GsfElectronAlgo|UnknownXtalRegion")<<"createElectron(): do not know if it is a barrel or endcap seed cluster !!!!" ; }


  //====================================================
  // SaturationInfo
  //====================================================

  reco::GsfElectron::SaturationInfo saturationInfo;
  calculateSaturationInfo(electronData.superClusterRef, saturationInfo, eventData);

  //====================================================
  // ShowerShape
  //====================================================

  reco::GsfElectron::ShowerShape showerShape;
  reco::GsfElectron::ShowerShape full5x5_showerShape;
  if( !EcalTools::isHGCalDet((DetId::Detector)region) ) {
    const bool pflow = !(electronData.coreRef->ecalDrivenSeed());
    auto const& hcalHelper = pflow ? generalData_.hcalHelperPflow : generalData_.hcalHelper;
    calculateShowerShape<false>(electronData.superClusterRef,hcalHelper,showerShape,eventData) ;    
    calculateShowerShape<true>(electronData.superClusterRef,hcalHelper,full5x5_showerShape,eventData) ;
  }

  //====================================================
  // ConversionRejection
  //====================================================

  eventData.retreiveOriginalTrackCollections(electronData.ctfTrackRef,electronData.coreRef->gsfTrack()) ;

  double BInTesla = eventSetupData_.magField->inTesla(GlobalPoint(0.,0.,0.)).z() ;
  edm::Handle<reco::TrackCollection> ctfTracks = eventData.originalCtfTracks ;
  if (!ctfTracks.isValid()) { ctfTracks = eventData.currentCtfTracks ; }

  // values of conversionInfo.flag
  // -9999 : Partner track was not found
  // 0     : Partner track found in the CTF collection using
  // 1     : Partner track found in the CTF collection using
  // 2     : Partner track found in the GSF collection using
  // 3     : Partner track found in the GSF collection using the electron's GSF track
  ConversionInfo conversionInfo = egammaTools::getConversionInfo
   (*electronData.coreRef,ctfTracks,eventData.originalGsfTracks,BInTesla) ;

  reco::GsfElectron::ConversionRejection conversionVars ;
  conversionVars.flags = conversionInfo.flag  ;
  conversionVars.dist = conversionInfo.dist  ;
  conversionVars.dcot = conversionInfo.dcot  ;
  conversionVars.radius = conversionInfo.radiusOfConversion  ;
  if(generalData_.strategyCfg.fillConvVtxFitProb){
    //this is an intentionally bugged version which ignores the GsfTrack
    //this is a bug which was introduced in reduced e/gamma where the GsfTrack gets 
    //relinked to a new collection which means it can no longer match the conversion
    //as it matches based on product/id
    //we keep this defination for the MVAs
    const auto matchedConv =  ConversionTools::matchedConversion(electronData.coreRef->ctfTrack(),
								 *eventData.conversions,
								 eventData.beamspot->position(),
								 2.0,1e-6,0);
    conversionVars.vtxFitProb = ConversionTools::getVtxFitProb(matchedConv);
  }
  if ((conversionVars.flags==0)or(conversionVars.flags==1))
    conversionVars.partner = TrackBaseRef(conversionInfo.conversionPartnerCtfTk)  ;
  else if ((conversionVars.flags==2)or(conversionVars.flags==3))
    conversionVars.partner = TrackBaseRef(conversionInfo.conversionPartnerGsfTk)  ;


  //====================================================
  // Go !
  //====================================================

  electrons.emplace_back( eleCharge,eleChargeInfo,electronData.coreRef,
       tcMatching, tkExtra, ctfInfo,
       fiducialFlags,showerShape, full5x5_showerShape,
       conversionVars, saturationInfo ) ;
  auto & ele = electrons.back();
  // Will be overwritten later in the case of the regression
  ele.setCorrectedEcalEnergyError(generalData_.superClusterErrorFunction->getValue(*(ele.superCluster()),0)) ;
  ele.setP4(GsfElectron::P4_FROM_SUPER_CLUSTER,momentum,0,true) ;
  
  //====================================================
  // brems fractions
  //====================================================

  if (electronData.innMom.mag()>0.)
   { ele.setTrackFbrem((electronData.innMom.mag()-electronData.outMom.mag())/electronData.innMom.mag()) ; }

  // the supercluster is the refined one The seed is not necessarily the first cluster
  // hence the use of the electronCluster
  SuperClusterRef sc = ele.superCluster() ;
  if (!(sc.isNull()))
   {
    CaloClusterPtr cl = ele.electronCluster() ;
    if (sc->clustersSize()>1)
     { 
       float pf_fbrem =( sc->energy() - cl->energy() ) / sc->energy();
       ele.setSuperClusterFbrem( pf_fbrem ) ;
     }
    else
      { 
	ele.setSuperClusterFbrem(0) ; 
      }
   }

  //====================================================
  // classification and corrections
  //====================================================
  // classification
  ElectronClassification theClassifier ;
  theClassifier.classify(ele) ;
  theClassifier.refineWithPflow(ele) ;
  // ecal energy
  ElectronEnergyCorrector theEnCorrector(generalData_.crackCorrectionFunction) ;
  if (generalData_.strategyCfg.useEcalRegression) // new 
    { 
      generalData_.regHelper.applyEcalRegression(ele,
						   eventData.vertices,
						   eventData.barrelRecHits,
						   eventData.endcapRecHits);
    }
  else  // original implementation
    {
      if( !EcalTools::isHGCalDet((DetId::Detector)region) ) {
        if (ele.core()->ecalDrivenSeed())
         {
          if (generalData_.strategyCfg.ecalDrivenEcalEnergyFromClassBasedParameterization)
            { theEnCorrector.classBasedParameterizationEnergy(ele,*eventData.beamspot) ; }
          if (generalData_.strategyCfg.ecalDrivenEcalErrorFromClassBasedParameterization)
            { theEnCorrector.classBasedParameterizationUncertainty(ele) ; }
         }
        else
         {
          if (generalData_.strategyCfg.pureTrackerDrivenEcalErrorFromSimpleParameterization)
            { theEnCorrector.simpleParameterizationUncertainty(ele) ; }
         }
      }
    }
  
  // momentum
  // Keep the default correction running first. The track momentum error is computed in there
  if (ele.core()->ecalDrivenSeed())
    {
      ElectronMomentumCorrector theMomCorrector;
      theMomCorrector.correct(ele,electronData.vtxTSOS);
    }
  if(generalData_.strategyCfg.useCombinationRegression)  // new 
    {
      generalData_.regHelper.applyCombinationRegression(ele);
    }

  //====================================================
  // now isolation variables
  //====================================================

  reco::GsfElectron::IsolationVariables dr03, dr04 ;
  dr03.tkSumPt = tkIsol03Calc_.calIsolPt(*ele.gsfTrack(),*eventData.currentCtfTracks);
  dr04.tkSumPt = tkIsol04Calc_.calIsolPt(*ele.gsfTrack(),*eventData.currentCtfTracks);
  dr03.tkSumPtHEEP = tkIsolHEEP03Calc_.calIsolPt(*ele.gsfTrack(),*eventData.currentCtfTracks);
  dr04.tkSumPtHEEP = tkIsolHEEP04Calc_.calIsolPt(*ele.gsfTrack(),*eventData.currentCtfTracks);

  if( !EcalTools::isHGCalDet((DetId::Detector)region) ) {
    dr03.hcalDepth1TowerSumEt = eventData.hadDepth1Isolation03.getTowerEtSum(&ele) ;
    dr03.hcalDepth2TowerSumEt = eventData.hadDepth2Isolation03.getTowerEtSum(&ele) ;
    dr03.hcalDepth1TowerSumEtBc = eventData.hadDepth1Isolation03Bc.getTowerEtSum(&ele,&(showerShape.hcalTowersBehindClusters)) ;
    dr03.hcalDepth2TowerSumEtBc = eventData.hadDepth2Isolation03Bc.getTowerEtSum(&ele,&(showerShape.hcalTowersBehindClusters)) ;
    dr03.ecalRecHitSumEt = eventData.ecalBarrelIsol03.getEtSum(&ele);
    dr03.ecalRecHitSumEt += eventData.ecalEndcapIsol03.getEtSum(&ele);    
    dr04.hcalDepth1TowerSumEt = eventData.hadDepth1Isolation04.getTowerEtSum(&ele);
    dr04.hcalDepth2TowerSumEt = eventData.hadDepth2Isolation04.getTowerEtSum(&ele);
    dr04.hcalDepth1TowerSumEtBc = eventData.hadDepth1Isolation04Bc.getTowerEtSum(&ele,&(showerShape.hcalTowersBehindClusters)) ;
    dr04.hcalDepth2TowerSumEtBc = eventData.hadDepth2Isolation04Bc.getTowerEtSum(&ele,&(showerShape.hcalTowersBehindClusters)) ;
    dr04.ecalRecHitSumEt = eventData.ecalBarrelIsol04.getEtSum(&ele);
    dr04.ecalRecHitSumEt += eventData.ecalEndcapIsol04.getEtSum(&ele);
  }
  ele.setIsolation03(dr03);
  ele.setIsolation04(dr04);


  //====================================================
  // preselection flag
  //====================================================

  setCutBasedPreselectionFlag(ele,*eventData.beamspot) ;
  //setting mva flag, currently GedGsfElectron and GsfElectron pre-selection flags have desynced
  //this is for GedGsfElectrons, GsfElectrons (ie old pre 7X std reco) resets this later on
  //in the function "addPfInfo"
  //yes this is awful, we'll fix it once we work out how to...
  float mvaValue = hoc->sElectronMVAEstimator->mva( ele,*(eventData.vertices));
  ele.setPassMvaPreselection(mvaValue>generalData_.strategyCfg.PreSelectMVA);

  //====================================================
  // Pixel match variables
  //====================================================
  setPixelMatchInfomation(ele) ;

  LogTrace("GsfElectronAlgo")<<"Constructed new electron with energy  "<< ele.p4().e() ;
 }


// Pixel match variables
void GsfElectronAlgo::setPixelMatchInfomation(reco::GsfElectron & ele){
  int sd1     = 0 ;
  int sd2     = 0 ;
  float dPhi1 = 0 ;
  float dPhi2 = 0 ;
  float dRz1  = 0 ;
  float dRz2  = 0 ;
  edm::RefToBase<TrajectorySeed> seed = ele.gsfTrack()->extra()->seedRef();
  ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>();
  if(seed.isNull()){}
  else{
    if(elseed.isNull()){}
    else{
      sd1     = elseed->subDet1() ;
      sd2     = elseed->subDet2() ;
      dPhi1 = (ele.charge()>0) ? elseed->dPhi1Pos() : elseed->dPhi1() ;
      dPhi2 = (ele.charge()>0) ? elseed->dPhi2Pos() : elseed->dPhi2() ;
      dRz1  = (ele.charge()>0) ? elseed->dRz1Pos () : elseed->dRz1 () ;
      dRz2  = (ele.charge()>0) ? elseed->dRz2Pos () : elseed->dRz2 () ;
    }
  }
  ele.setPixelMatchSubdetectors(sd1,sd2) ;
  ele.setPixelMatchDPhi1(dPhi1) ;
  ele.setPixelMatchDPhi2(dPhi2) ;
  ele.setPixelMatchDRz1 (dRz1 ) ;
  ele.setPixelMatchDRz2 (dRz2 ) ;
}
