
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/EgAmbiguityTools.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"


#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"


#include <Math/Point3D.h>
#include <sstream>
#include <algorithm>


using namespace edm ;
using namespace std ;
using namespace reco ;


//===================================================================
// GsfElectronAlgo::GeneralData
//===================================================================

// general data and helpers
struct GsfElectronAlgo::GeneralData
 {
  // constructors
  GeneralData
   ( const InputTagsConfiguration &,
     const StrategyConfiguration &,
     const CutsConfiguration & cutsCfg,
     const CutsConfiguration & cutsCfgPflow,
     const ElectronHcalHelper::Configuration & hcalCfg,
     const ElectronHcalHelper::Configuration & hcalCfgPflow,
     const IsolationConfiguration &,
     const EcalRecHitsConfiguration &,
     EcalClusterFunctionBaseClass * superClusterErrorFunction,
     EcalClusterFunctionBaseClass * crackCorrectionFunction,
     const SoftElectronMVAEstimator::Configuration & mva_NIso_Cfg ,
     const ElectronMVAEstimator::Configuration & mva_Iso_Cfg ,
     const RegressionHelper::Configuration &) ;
  ~GeneralData() ;

  // configurables
  const InputTagsConfiguration inputCfg ;
  const StrategyConfiguration strategyCfg ;
  const CutsConfiguration cutsCfg ;
  const CutsConfiguration cutsCfgPflow ;
  const IsolationConfiguration isoCfg ;
  const EcalRecHitsConfiguration recHitsCfg ;

  // additional configuration and helpers
  ElectronHcalHelper * hcalHelper, * hcalHelperPflow ;
  EcalClusterFunctionBaseClass * superClusterErrorFunction ;
  EcalClusterFunctionBaseClass * crackCorrectionFunction ;
  //SoftElectronMVAEstimator *sElectronMVAEstimator;
  //ElectronMVAEstimator *iElectronMVAEstimator;
  const RegressionHelper::Configuration regCfg;
  RegressionHelper * regHelper;
 } ;

 GsfElectronAlgo::GeneralData::GeneralData
 ( const InputTagsConfiguration & inputConfig,
   const StrategyConfiguration & strategyConfig,
   const CutsConfiguration & cutsConfig,
   const CutsConfiguration & cutsConfigPflow,
   const ElectronHcalHelper::Configuration & hcalConfig,
   const ElectronHcalHelper::Configuration & hcalConfigPflow,
   const IsolationConfiguration & isoConfig,
   const EcalRecHitsConfiguration & recHitsConfig,
   EcalClusterFunctionBaseClass * superClusterErrorFunc,
   EcalClusterFunctionBaseClass * crackCorrectionFunc,
   const SoftElectronMVAEstimator::Configuration & /*mva_NIso_Config*/,
   const ElectronMVAEstimator::Configuration & /*mva_Iso_Config*/,
   const RegressionHelper::Configuration & regConfig
   )
 : inputCfg(inputConfig),
   strategyCfg(strategyConfig),
   cutsCfg(cutsConfig),
   cutsCfgPflow(cutsConfigPflow),
   isoCfg(isoConfig),
   recHitsCfg(recHitsConfig),
   hcalHelper(new ElectronHcalHelper(hcalConfig)),
   hcalHelperPflow(new ElectronHcalHelper(hcalConfigPflow)),
   superClusterErrorFunction(superClusterErrorFunc),
   crackCorrectionFunction(crackCorrectionFunc),
   //sElectronMVAEstimator(new SoftElectronMVAEstimator(mva_NIso_Config)),
   //iElectronMVAEstimator(new ElectronMVAEstimator(mva_Iso_Config)),
   regCfg(regConfig),
   regHelper(new RegressionHelper(regConfig))
  {}

GsfElectronAlgo::GeneralData::~GeneralData()
 {
  delete hcalHelper ;
  delete hcalHelperPflow ;
  //delete sElectronMVAEstimator;
  //delete iElectronMVAEstimator;
  delete regHelper;
 }

//===================================================================
// GsfElectronAlgo::EventSetupData
//===================================================================

struct GsfElectronAlgo::EventSetupData
 {
   EventSetupData() ;
   ~EventSetupData() ;

   unsigned long long cacheIDGeom ;
   unsigned long long cacheIDTopo ;
   unsigned long long cacheIDTDGeom ;
   unsigned long long cacheIDMagField ;
   //unsigned long long cacheChStatus ;
   unsigned long long cacheSevLevel ;

   edm::ESHandle<MagneticField> magField ;
   edm::ESHandle<CaloGeometry> caloGeom ;
   edm::ESHandle<CaloTopology> caloTopo ;
   edm::ESHandle<TrackerGeometry> trackerHandle ;
   //edm::ESHandle<EcalChannelStatus> chStatus ;
   edm::ESHandle<EcalSeverityLevelAlgo> sevLevel;

   const MultiTrajectoryStateTransform * mtsTransform ;
   GsfConstraintAtVertex * constraintAtVtx ;
   const MultiTrajectoryStateMode * mtsMode ;
} ;

GsfElectronAlgo::EventSetupData::EventSetupData()
 : cacheIDGeom(0), cacheIDTopo(0), cacheIDTDGeom(0), cacheIDMagField(0),/*cacheChStatus(0),*/
   cacheSevLevel(0), mtsTransform(0), constraintAtVtx(0), mtsMode(new MultiTrajectoryStateMode)
 {}

GsfElectronAlgo::EventSetupData::~EventSetupData()
 {
  delete mtsMode ;
  delete constraintAtVtx ;
  delete mtsTransform ;
 }


//===================================================================
// GsfElectronAlgo::EventData
//===================================================================

struct GsfElectronAlgo::EventData
 {
  // general
  edm::Event * event ;
  const reco::BeamSpot * beamspot ;
  GsfElectronPtrCollection * electrons ;

  EventData() ;
  ~EventData() ;

  // utilities
  void retreiveOriginalTrackCollections
   ( const reco::TrackRef &, const reco::GsfTrackRef & ) ;

  // input collections
  edm::Handle<reco::GsfElectronCollection> previousElectrons ;
  edm::Handle<reco::GsfElectronCollection> pflowElectrons ;
  edm::Handle<reco::GsfElectronCoreCollection> coreElectrons ;
  edm::Handle<EcalRecHitCollection> barrelRecHits ;
  edm::Handle<EcalRecHitCollection> endcapRecHits ;
  edm::Handle<reco::TrackCollection> currentCtfTracks ;
  edm::Handle<CaloTowerCollection> towers ;
  edm::Handle<edm::ValueMap<float> > pfMva ;
  edm::Handle<reco::ElectronSeedCollection> seeds ;
  edm::Handle<reco::GsfPFRecTrackCollection> gsfPfRecTracks ;
  bool originalCtfTrackCollectionRetreived ;
  bool originalGsfTrackCollectionRetreived ;
  edm::Handle<reco::TrackCollection> originalCtfTracks ;
  edm::Handle<reco::GsfTrackCollection> originalGsfTracks ;
  edm::Handle<reco::VertexCollection> vertices;

  // isolation helpers
  ElectronTkIsolation * tkIsolation03, * tkIsolation04 ;
  EgammaTowerIsolation * hadDepth1Isolation03, * hadDepth1Isolation04 ;
  EgammaTowerIsolation * hadDepth2Isolation03, * hadDepth2Isolation04 ;
  EgammaTowerIsolation * hadDepth1Isolation03Bc, * hadDepth1Isolation04Bc ;
  EgammaTowerIsolation * hadDepth2Isolation03Bc, * hadDepth2Isolation04Bc ;
  EgammaRecHitIsolation * ecalBarrelIsol03, * ecalBarrelIsol04 ;
  EgammaRecHitIsolation * ecalEndcapIsol03, * ecalEndcapIsol04 ;

  //Isolation Value Maps for PF and EcalDriven electrons
  typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;
  IsolationValueMaps pfIsolationValues;
  IsolationValueMaps edIsolationValues;
 } ;

GsfElectronAlgo::EventData::EventData()
 : event(0), beamspot(0),
   originalCtfTrackCollectionRetreived(false),
   originalGsfTrackCollectionRetreived(false),
   tkIsolation03(0), tkIsolation04(0),
   hadDepth1Isolation03(0), hadDepth1Isolation04(0),
   hadDepth2Isolation03(0), hadDepth2Isolation04(0),
   hadDepth1Isolation03Bc(0), hadDepth1Isolation04Bc(0),
   hadDepth2Isolation03Bc(0), hadDepth2Isolation04Bc(0),
   ecalBarrelIsol03(0), ecalBarrelIsol04(0),
   ecalEndcapIsol03(0), ecalEndcapIsol04(0)
 {
  electrons = new GsfElectronPtrCollection ;
 }

GsfElectronAlgo::EventData::~EventData()
 {
  delete tkIsolation03 ;
  delete tkIsolation04 ;
  delete hadDepth1Isolation03 ;
  delete hadDepth1Isolation04 ;
  delete hadDepth2Isolation03 ;
  delete hadDepth2Isolation04 ; 
  delete hadDepth1Isolation03Bc ;
  delete hadDepth1Isolation04Bc ;
  delete hadDepth2Isolation03Bc ;
  delete hadDepth2Isolation04Bc ;
  delete ecalBarrelIsol03 ;
  delete ecalBarrelIsol04 ;
  delete ecalEndcapIsol03 ;
  delete ecalEndcapIsol04 ;

  GsfElectronPtrCollection::const_iterator it ;
  for ( it = electrons->begin() ; it != electrons->end() ; it++ )
   { delete (*it) ; }
  delete electrons ;
 }

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


//===================================================================
// GsfElectronAlgo::ElectronData
//===================================================================

struct GsfElectronAlgo::ElectronData
 {
  // Refs to subproducts
  const reco::GsfElectronCoreRef coreRef ;
  const reco::GsfTrackRef gsfTrackRef ;
  const reco::SuperClusterRef superClusterRef ;
  reco::TrackRef ctfTrackRef ;
  float shFracInnerHits ;
  const reco::BeamSpot beamSpot ;

  // constructors
  ElectronData
   ( const reco::GsfElectronCoreRef & core,
     const reco::BeamSpot & bs ) ;
  ~ElectronData() ;

  // utilities
  void checkCtfTrack( edm::Handle<reco::TrackCollection> currentCtfTracks ) ;
  void computeCharge( int & charge, reco::GsfElectron::ChargeInfo & info ) ;
  CaloClusterPtr getEleBasicCluster( const MultiTrajectoryStateTransform * ) ;
  bool calculateTSOS( const MultiTrajectoryStateTransform *, GsfConstraintAtVertex * ) ;
  void calculateMode( const MultiTrajectoryStateMode * mtsMode ) ;
  Candidate::LorentzVector calculateMomentum() ;

  // TSOS
  TrajectoryStateOnSurface innTSOS ;
  TrajectoryStateOnSurface outTSOS ;
  TrajectoryStateOnSurface vtxTSOS ;
  TrajectoryStateOnSurface sclTSOS ;
  TrajectoryStateOnSurface seedTSOS ;
  TrajectoryStateOnSurface eleTSOS ;
  TrajectoryStateOnSurface constrainedVtxTSOS ;

  // mode
  GlobalVector innMom, seedMom, eleMom, sclMom, vtxMom, outMom ;
  GlobalPoint innPos, seedPos, elePos, sclPos, vtxPos, outPos ;
  GlobalVector vtxMomWithConstraint ;
 } ;

GsfElectronAlgo::ElectronData::ElectronData
 ( const reco::GsfElectronCoreRef & core,
   const reco::BeamSpot & bs )
 : coreRef(core),
   gsfTrackRef(coreRef->gsfTrack()),
   superClusterRef(coreRef->superCluster()),
   ctfTrackRef(coreRef->ctfTrack()), shFracInnerHits(coreRef->ctfGsfOverlap()),
   beamSpot(bs)
 {}

GsfElectronAlgo::ElectronData::~ElectronData()
 {}

void GsfElectronAlgo::ElectronData::checkCtfTrack( edm::Handle<reco::TrackCollection> currentCtfTracks )
{
    if (!ctfTrackRef.isNull()) return ;

    // Code below from Puneeth Kalavase

    shFracInnerHits = 0 ;
    const TrackCollection * ctfTrackCollection = currentCtfTracks.product() ;

    // get the Hit Pattern for the gsfTrack
    const HitPattern &gsfHitPattern = gsfTrackRef->hitPattern() ;

    unsigned int counter ;
    TrackCollection::const_iterator ctfTkIter ;
    for (ctfTkIter = ctfTrackCollection->begin(), counter = 0;
            ctfTkIter != ctfTrackCollection->end(); ctfTkIter++, counter++)
    {
        double dEta = gsfTrackRef->eta() - ctfTkIter->eta() ;
        double dPhi = gsfTrackRef->phi() - ctfTkIter->phi() ;
        double pi = acos(-1.);
        if(std::abs(dPhi) > pi) dPhi = 2*pi - std::abs(dPhi) ;

        // dont want to look at every single track in the event!
        if (sqrt(dEta*dEta + dPhi*dPhi) > 0.3) continue ;

        unsigned int shared = 0 ;
        int gsfHitCounter = 0 ;
        int numGsfInnerHits = 0 ;
        int numCtfInnerHits = 0 ;

        // get the CTF Track Hit Pattern
        const HitPattern &ctfHitPattern = ctfTkIter->hitPattern() ;

        trackingRecHit_iterator elHitsIt;
        for (elHitsIt = gsfTrackRef->recHitsBegin();
                elHitsIt != gsfTrackRef->recHitsEnd();
                elHitsIt++, gsfHitCounter++)
        {
            if (!((**elHitsIt).isValid()))  //count only valid Hits
            { continue ; }

            // look only in the pixels/TIB/TID
            uint32_t gsfHit = gsfHitPattern.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter) ;
            if (!(HitPattern::pixelHitFilter(gsfHit)
                        || HitPattern::stripTIBHitFilter(gsfHit)
                        || HitPattern::stripTIDHitFilter(gsfHit))){
                continue;
            }

            numGsfInnerHits++ ;

            int ctfHitsCounter = 0 ;
            numCtfInnerHits = 0 ;
            trackingRecHit_iterator ctfHitsIt ;
            for (ctfHitsIt = ctfTkIter->recHitsBegin();
                    ctfHitsIt != ctfTkIter->recHitsEnd();
                    ctfHitsIt++, ctfHitsCounter++ )
            {
                if(!((**ctfHitsIt).isValid())) //count only valid Hits!
                { continue; }

                uint32_t ctfHit = ctfHitPattern.getHitPattern(HitPattern::TRACK_HITS, ctfHitsCounter);
                if(!(HitPattern::pixelHitFilter(ctfHit)
                            || HitPattern::stripTIBHitFilter(ctfHit)
                            || HitPattern::stripTIDHitFilter(ctfHit)))
                {
                    continue;
                }

                numCtfInnerHits++ ;

                if((**elHitsIt).sharesInput(&(**ctfHitsIt), TrackingRecHit::all))
                {
                    shared++ ;
                    break ;
                }

            } //ctfHits iterator

        } //gsfHits iterator

        if ((numGsfInnerHits==0)||(numCtfInnerHits==0))
        { continue ; }

        if ( static_cast<float>(shared)/min(numGsfInnerHits,numCtfInnerHits) > shFracInnerHits )
        {
            shFracInnerHits = static_cast<float>(shared)/min(numGsfInnerHits, numCtfInnerHits);
            ctfTrackRef = TrackRef(currentCtfTracks,counter);
        }
    } //ctfTrack iterator
}

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
  float dPhiInnEle=normalized_phi(scvect.phi()-inntkvect.phi()) ;
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

CaloClusterPtr GsfElectronAlgo::ElectronData::getEleBasicCluster
 ( const MultiTrajectoryStateTransform * mtsTransform )
 {
  CaloClusterPtr eleRef ;
  TrajectoryStateOnSurface tempTSOS ;
  TrajectoryStateOnSurface outTSOS = mtsTransform->outerStateOnSurface(*gsfTrackRef) ;
  float dphimin = 1.e30 ;
  for (CaloCluster_iterator bc=superClusterRef->clustersBegin(); bc!=superClusterRef->clustersEnd(); bc++)
   {
    GlobalPoint posclu((*bc)->position().x(),(*bc)->position().y(),(*bc)->position().z()) ;
    tempTSOS = mtsTransform->extrapolatedState(outTSOS,posclu) ;
    if (!tempTSOS.isValid()) tempTSOS=outTSOS ;
    GlobalPoint extrap = tempTSOS.globalPosition() ;
    float dphi = EleRelPointPair(posclu,extrap,beamSpot.position()).dPhi() ;
    if (std::abs(dphi)<dphimin)
     {
      dphimin = std::abs(dphi) ;
      eleRef = (*bc);
      eleTSOS = tempTSOS ;
     }
   }
  return eleRef ;
 }

bool GsfElectronAlgo::ElectronData::calculateTSOS
 ( const MultiTrajectoryStateTransform * mtsTransform, GsfConstraintAtVertex * constraintAtVtx )
 {
  //at innermost point
  innTSOS = mtsTransform->innerStateOnSurface(*gsfTrackRef);
  if (!innTSOS.isValid()) return false;

  //at vertex
  // innermost state propagation to the beam spot position
  GlobalPoint bsPos ;
  ele_convert(beamSpot.position(),bsPos) ;
  vtxTSOS = mtsTransform->extrapolatedState(innTSOS,bsPos) ;
  if (!vtxTSOS.isValid()) vtxTSOS=innTSOS;

  //at seed
  outTSOS = mtsTransform->outerStateOnSurface(*gsfTrackRef);
  if (!outTSOS.isValid()) return false;

  //    TrajectoryStateOnSurface seedTSOS
  seedTSOS = mtsTransform->extrapolatedState(outTSOS,
           GlobalPoint(superClusterRef->seed()->position().x(),
               superClusterRef->seed()->position().y(),
                 superClusterRef->seed()->position().z()));
  if (!seedTSOS.isValid()) seedTSOS=outTSOS;

  // at scl
  sclTSOS = mtsTransform->extrapolatedState(innTSOS,GlobalPoint(superClusterRef->x(),superClusterRef->y(),superClusterRef->z()));
  if (!sclTSOS.isValid()) sclTSOS=outTSOS;

  // constrained momentum
  constrainedVtxTSOS = constraintAtVtx->constrainAtBeamSpot(*gsfTrackRef,beamSpot);

  return true ;
 }

void GsfElectronAlgo::ElectronData::calculateMode( const MultiTrajectoryStateMode * mtsMode )
 {
  mtsMode->momentumFromModeCartesian(innTSOS,innMom) ;
  mtsMode->positionFromModeCartesian(innTSOS,innPos) ;
  mtsMode->momentumFromModeCartesian(seedTSOS,seedMom) ;
  mtsMode->positionFromModeCartesian(seedTSOS,seedPos) ;
  mtsMode->momentumFromModeCartesian(eleTSOS,eleMom) ;
  mtsMode->positionFromModeCartesian(eleTSOS,elePos) ;
  mtsMode->momentumFromModeCartesian(sclTSOS,sclMom) ;
  mtsMode->positionFromModeCartesian(sclTSOS,sclPos) ;
  mtsMode->momentumFromModeCartesian(vtxTSOS,vtxMom) ;
  mtsMode->positionFromModeCartesian(vtxTSOS,vtxPos) ;
  mtsMode->momentumFromModeCartesian(outTSOS,outMom);
  mtsMode->positionFromModeCartesian(outTSOS,outPos) ;
  mtsMode->momentumFromModeCartesian(constrainedVtxTSOS,vtxMomWithConstraint);
 }

Candidate::LorentzVector GsfElectronAlgo::ElectronData::calculateMomentum()
 {
  double scale = superClusterRef->energy()/vtxMom.mag() ;
  return Candidate::LorentzVector
   ( vtxMom.x()*scale,vtxMom.y()*scale,vtxMom.z()*scale,
     superClusterRef->energy() ) ;
 }

void GsfElectronAlgo::calculateShowerShape( const reco::SuperClusterRef & theClus, bool pflow, reco::GsfElectron::ShowerShape & showerShape )
 {
  const reco::CaloCluster & seedCluster = *(theClus->seed()) ;
  // temporary, till CaloCluster->seed() is made available
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;
  int detector = seedXtalId.subdetId() ;

  const CaloTopology * topology = eventSetupData_->caloTopo.product() ;
  const CaloGeometry * geometry = eventSetupData_->caloGeom.product() ;
  const EcalRecHitCollection * recHits = 0 ;
  std::vector<int> recHitFlagsToBeExcluded ;
  std::vector<int> recHitSeverityToBeExcluded ;
  if (detector==EcalBarrel)
   {
    recHits = eventData_->barrelRecHits.product() ;
    recHitFlagsToBeExcluded = generalData_->recHitsCfg.recHitFlagsToBeExcludedBarrel ;
    recHitSeverityToBeExcluded = generalData_->recHitsCfg.recHitSeverityToBeExcludedBarrel ;
   }
  else
   {
    recHits = eventData_->endcapRecHits.product() ;
    recHitFlagsToBeExcluded = generalData_->recHitsCfg.recHitFlagsToBeExcludedEndcaps ;
    recHitSeverityToBeExcluded = generalData_->recHitsCfg.recHitSeverityToBeExcludedEndcaps ;
   }

  std::vector<float> covariances = EcalClusterTools::covariances(seedCluster,recHits,topology,geometry) ;
  std::vector<float> localCovariances = EcalClusterTools::localCovariances(seedCluster,recHits,topology) ;
  showerShape.sigmaEtaEta = sqrt(covariances[0]) ;
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]) ;
  if (!edm::isNotFinite(localCovariances[2])) showerShape.sigmaIphiIphi = sqrt(localCovariances[2]) ;
  showerShape.e1x5 = EcalClusterTools::e1x5(seedCluster,recHits,topology)  ;
  showerShape.e2x5Max = EcalClusterTools::e2x5Max(seedCluster,recHits,topology)  ;
  showerShape.e5x5 = EcalClusterTools::e5x5(seedCluster,recHits,topology) ;
  showerShape.r9 = EcalClusterTools::e3x3(seedCluster,recHits,topology)/theClus->rawEnergy() ;

  if (pflow)
   {
    showerShape.hcalDepth1OverEcal = generalData_->hcalHelperPflow->hcalESumDepth1(*theClus)/theClus->energy() ;
    showerShape.hcalDepth2OverEcal = generalData_->hcalHelperPflow->hcalESumDepth2(*theClus)/theClus->energy() ;
    showerShape.hcalTowersBehindClusters = generalData_->hcalHelperPflow->hcalTowersBehindClusters(*theClus) ;
    showerShape.hcalDepth1OverEcalBc = generalData_->hcalHelperPflow->hcalESumDepth1BehindClusters(showerShape.hcalTowersBehindClusters)/theClus->energy() ;
    showerShape.hcalDepth2OverEcalBc = generalData_->hcalHelperPflow->hcalESumDepth2BehindClusters(showerShape.hcalTowersBehindClusters)/theClus->energy() ;
   }
  else
   {
    showerShape.hcalDepth1OverEcal = generalData_->hcalHelper->hcalESumDepth1(*theClus)/theClus->energy() ;
    showerShape.hcalDepth2OverEcal = generalData_->hcalHelper->hcalESumDepth2(*theClus)/theClus->energy() ;
    showerShape.hcalTowersBehindClusters = generalData_->hcalHelper->hcalTowersBehindClusters(*theClus) ;
    showerShape.hcalDepth1OverEcalBc = generalData_->hcalHelper->hcalESumDepth1BehindClusters(showerShape.hcalTowersBehindClusters)/theClus->energy() ;
    showerShape.hcalDepth2OverEcalBc = generalData_->hcalHelper->hcalESumDepth2BehindClusters(showerShape.hcalTowersBehindClusters)/theClus->energy() ;
   }
 }

void GsfElectronAlgo::calculateShowerShape_full5x5( const reco::SuperClusterRef & theClus, bool pflow, reco::GsfElectron::ShowerShape & showerShape )
 {
  const reco::CaloCluster & seedCluster = *(theClus->seed()) ;
  // temporary, till CaloCluster->seed() is made available
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;
  int detector = seedXtalId.subdetId() ;

  const CaloTopology * topology = eventSetupData_->caloTopo.product() ;
  const CaloGeometry * geometry = eventSetupData_->caloGeom.product() ;
  const EcalRecHitCollection * recHits = 0 ;
  std::vector<int> recHitFlagsToBeExcluded ;
  std::vector<int> recHitSeverityToBeExcluded ;
  if (detector==EcalBarrel)
   {
    recHits = eventData_->barrelRecHits.product() ;
    recHitFlagsToBeExcluded = generalData_->recHitsCfg.recHitFlagsToBeExcludedBarrel ;
    recHitSeverityToBeExcluded = generalData_->recHitsCfg.recHitSeverityToBeExcludedBarrel ;
   }
  else
   {
    recHits = eventData_->endcapRecHits.product() ;
    recHitFlagsToBeExcluded = generalData_->recHitsCfg.recHitFlagsToBeExcludedEndcaps ;
    recHitSeverityToBeExcluded = generalData_->recHitsCfg.recHitSeverityToBeExcludedEndcaps ;
   }

  std::vector<float> covariances = noZS::EcalClusterTools::covariances(seedCluster,recHits,topology,geometry) ;
  std::vector<float> localCovariances = noZS::EcalClusterTools::localCovariances(seedCluster,recHits,topology) ;
  showerShape.sigmaEtaEta = sqrt(covariances[0]) ;
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]) ;
  if (!edm::isNotFinite(localCovariances[2])) showerShape.sigmaIphiIphi = sqrt(localCovariances[2]) ;
  showerShape.e1x5 = noZS::EcalClusterTools::e1x5(seedCluster,recHits,topology)  ;
  showerShape.e2x5Max = noZS::EcalClusterTools::e2x5Max(seedCluster,recHits,topology)  ;
  showerShape.e5x5 = noZS::EcalClusterTools::e5x5(seedCluster,recHits,topology) ;
  showerShape.r9 = noZS::EcalClusterTools::e3x3(seedCluster,recHits,topology)/theClus->rawEnergy() ;

  if (pflow)
   {
    showerShape.hcalDepth1OverEcal = generalData_->hcalHelperPflow->hcalESumDepth1(*theClus)/theClus->energy() ;
    showerShape.hcalDepth2OverEcal = generalData_->hcalHelperPflow->hcalESumDepth2(*theClus)/theClus->energy() ;
    showerShape.hcalTowersBehindClusters = generalData_->hcalHelperPflow->hcalTowersBehindClusters(*theClus) ;
    showerShape.hcalDepth1OverEcalBc = generalData_->hcalHelperPflow->hcalESumDepth1BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
    showerShape.hcalDepth2OverEcalBc = generalData_->hcalHelperPflow->hcalESumDepth2BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
   }
  else
   {
    showerShape.hcalDepth1OverEcal = generalData_->hcalHelper->hcalESumDepth1(*theClus)/theClus->energy() ;
    showerShape.hcalDepth2OverEcal = generalData_->hcalHelper->hcalESumDepth2(*theClus)/theClus->energy() ;
    showerShape.hcalTowersBehindClusters = generalData_->hcalHelper->hcalTowersBehindClusters(*theClus) ;
    showerShape.hcalDepth1OverEcalBc = generalData_->hcalHelper->hcalESumDepth1BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
    showerShape.hcalDepth2OverEcalBc = generalData_->hcalHelper->hcalESumDepth2BehindClusters(showerShape.hcalTowersBehindClusters)/showerShape.e5x5 ;
   }
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
   const SoftElectronMVAEstimator::Configuration & mva_NIso_Cfg,
   const ElectronMVAEstimator::Configuration & mva_Iso_Cfg,
   const RegressionHelper::Configuration & regCfg
 )
   : generalData_(new GeneralData(inputCfg,strategyCfg,cutsCfg,cutsCfgPflow,hcalCfg,hcalCfgPflow,isoCfg,recHitsCfg,superClusterErrorFunction,crackCorrectionFunction,mva_NIso_Cfg,mva_Iso_Cfg,regCfg)),
   eventSetupData_(new EventSetupData),
   eventData_(0), electronData_(0)
 {}

GsfElectronAlgo::~GsfElectronAlgo()
 {
  delete generalData_ ;
  delete eventSetupData_ ;
  delete eventData_ ;
  delete electronData_ ;
 }

void GsfElectronAlgo::checkSetup( const edm::EventSetup & es )
 {
  // get EventSetupRecords if needed
  bool updateField(false);
  if (eventSetupData_->cacheIDMagField!=es.get<IdealMagneticFieldRecord>().cacheIdentifier()){
    updateField = true;
    eventSetupData_->cacheIDMagField=es.get<IdealMagneticFieldRecord>().cacheIdentifier();
    es.get<IdealMagneticFieldRecord>().get(eventSetupData_->magField);
  }

  bool updateGeometry(false);
  if (eventSetupData_->cacheIDTDGeom!=es.get<TrackerDigiGeometryRecord>().cacheIdentifier()){
    updateGeometry = true;
    eventSetupData_->cacheIDTDGeom=es.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    es.get<TrackerDigiGeometryRecord>().get(eventSetupData_->trackerHandle);
  }

  if ( updateField || updateGeometry ) {
    delete eventSetupData_->mtsTransform ;
    eventSetupData_->mtsTransform = new MultiTrajectoryStateTransform(eventSetupData_->trackerHandle.product(),eventSetupData_->magField.product());
    delete eventSetupData_->constraintAtVtx ;
    eventSetupData_->constraintAtVtx = new GsfConstraintAtVertex(es) ;
  }

  if (eventSetupData_->cacheIDGeom!=es.get<CaloGeometryRecord>().cacheIdentifier()){
    eventSetupData_->cacheIDGeom=es.get<CaloGeometryRecord>().cacheIdentifier();
    es.get<CaloGeometryRecord>().get(eventSetupData_->caloGeom);
  }

  if (eventSetupData_->cacheIDTopo!=es.get<CaloTopologyRecord>().cacheIdentifier()){
    eventSetupData_->cacheIDTopo=es.get<CaloTopologyRecord>().cacheIdentifier();
    es.get<CaloTopologyRecord>().get(eventSetupData_->caloTopo);
  }

  generalData_->hcalHelper->checkSetup(es) ;
  generalData_->hcalHelperPflow->checkSetup(es) ;
  if(generalData_->strategyCfg.useEcalRegression || generalData_->strategyCfg.useCombinationRegression)
    generalData_->regHelper->checkSetup(es);


  if (generalData_->superClusterErrorFunction)
   { generalData_->superClusterErrorFunction->init(es) ; }
  if (generalData_->crackCorrectionFunction)
   { generalData_->crackCorrectionFunction->init(es) ; }

  //if(eventSetupData_->cacheChStatus!=es.get<EcalChannelStatusRcd>().cacheIdentifier()){
  //  eventSetupData_->cacheChStatus=es.get<EcalChannelStatusRcd>().cacheIdentifier();
  //  es.get<EcalChannelStatusRcd>().get(eventSetupData_->chStatus);
  //}

  if(eventSetupData_->cacheSevLevel != es.get<EcalSeverityLevelAlgoRcd>().cacheIdentifier()){
    eventSetupData_->cacheSevLevel = es.get<EcalSeverityLevelAlgoRcd>().cacheIdentifier();
    es.get<EcalSeverityLevelAlgoRcd>().get(eventSetupData_->sevLevel);
  }
 }


void GsfElectronAlgo::copyElectrons( GsfElectronCollection & outEle )
 {
  GsfElectronPtrCollection::const_iterator it ;
  for
   ( it = eventData_->electrons->begin() ;
     it != eventData_->electrons->end() ;
     it++ )
   { outEle.push_back(**it) ; }
 }

void GsfElectronAlgo::beginEvent( edm::Event & event )
 {
  if (eventData_!=0)
   { throw cms::Exception("GsfElectronAlgo|InternalError")<<"unexpected event data" ; }
  eventData_ = new EventData ;

  // init the handles linked to the current event
  eventData_->event = &event ;
  event.getByToken(generalData_->inputCfg.previousGsfElectrons,eventData_->previousElectrons) ;
  event.getByToken(generalData_->inputCfg.pflowGsfElectronsTag,eventData_->pflowElectrons) ;
  event.getByToken(generalData_->inputCfg.gsfElectronCores,eventData_->coreElectrons) ;
  event.getByToken(generalData_->inputCfg.ctfTracks,eventData_->currentCtfTracks) ;
  event.getByToken(generalData_->inputCfg.barrelRecHitCollection,eventData_->barrelRecHits) ;
  event.getByToken(generalData_->inputCfg.endcapRecHitCollection,eventData_->endcapRecHits) ;
  event.getByToken(generalData_->inputCfg.hcalTowersTag,eventData_->towers) ;
  event.getByToken(generalData_->inputCfg.pfMVA,eventData_->pfMva) ;
  event.getByToken(generalData_->inputCfg.seedsTag,eventData_->seeds) ;
  event.getByToken(generalData_->inputCfg.vtxCollectionTag,eventData_->vertices);
  if (generalData_->strategyCfg.useGsfPfRecTracks)
   { event.getByToken(generalData_->inputCfg.gsfPfRecTracksTag,eventData_->gsfPfRecTracks) ; }

  // get the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle ;
  event.getByToken(generalData_->inputCfg.beamSpotTag,recoBeamSpotHandle) ;
  eventData_->beamspot = recoBeamSpotHandle.product() ;

  // prepare access to hcal data
  generalData_->hcalHelper->readEvent(event) ;
  generalData_->hcalHelperPflow->readEvent(event) ;

  // Isolation algos
  float extRadiusSmall=0.3, extRadiusLarge=0.4 ;
  float intRadiusBarrel=generalData_->isoCfg.intRadiusBarrelTk, intRadiusEndcap=generalData_->isoCfg.intRadiusEndcapTk, stripBarrel=generalData_->isoCfg.stripBarrelTk, stripEndcap=generalData_->isoCfg.stripEndcapTk ;
  float ptMin=generalData_->isoCfg.ptMinTk, maxVtxDist=generalData_->isoCfg.maxVtxDistTk, drb=generalData_->isoCfg.maxDrbTk;
  eventData_->tkIsolation03 = new ElectronTkIsolation(extRadiusSmall,intRadiusBarrel,intRadiusEndcap,stripBarrel,stripEndcap,ptMin,maxVtxDist,drb,eventData_->currentCtfTracks.product(),eventData_->beamspot->position()) ;
  eventData_->tkIsolation04 = new ElectronTkIsolation(extRadiusLarge,intRadiusBarrel,intRadiusEndcap,stripBarrel,stripEndcap,ptMin,maxVtxDist,drb,eventData_->currentCtfTracks.product(),eventData_->beamspot->position()) ;

  float egHcalIsoConeSizeOutSmall=0.3, egHcalIsoConeSizeOutLarge=0.4;
  float egHcalIsoConeSizeIn=generalData_->isoCfg.intRadiusHcal,egHcalIsoPtMin=generalData_->isoCfg.etMinHcal;
  int egHcalDepth1=1, egHcalDepth2=2;
  eventData_->hadDepth1Isolation03 = new EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,eventData_->towers.product()) ;
  eventData_->hadDepth2Isolation03 = new EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,eventData_->towers.product()) ;
  eventData_->hadDepth1Isolation04 = new EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,eventData_->towers.product()) ;
  eventData_->hadDepth2Isolation04 = new EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,eventData_->towers.product()) ;
  eventData_->hadDepth1Isolation03Bc = new EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,0.,egHcalIsoPtMin,egHcalDepth1,eventData_->towers.product()) ;
  eventData_->hadDepth2Isolation03Bc = new EgammaTowerIsolation(egHcalIsoConeSizeOutSmall,0.,egHcalIsoPtMin,egHcalDepth2,eventData_->towers.product()) ;
  eventData_->hadDepth1Isolation04Bc = new EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,0.,egHcalIsoPtMin,egHcalDepth1,eventData_->towers.product()) ;
  eventData_->hadDepth2Isolation04Bc = new EgammaTowerIsolation(egHcalIsoConeSizeOutLarge,0.,egHcalIsoPtMin,egHcalDepth2,eventData_->towers.product()) ;

  float egIsoConeSizeOutSmall=0.3, egIsoConeSizeOutLarge=0.4, egIsoJurassicWidth=generalData_->isoCfg.jurassicWidth;
  float egIsoPtMinBarrel=generalData_->isoCfg.etMinBarrel,egIsoEMinBarrel=generalData_->isoCfg.eMinBarrel, egIsoConeSizeInBarrel=generalData_->isoCfg.intRadiusEcalBarrel;
  float egIsoPtMinEndcap=generalData_->isoCfg.etMinEndcaps,egIsoEMinEndcap=generalData_->isoCfg.eMinEndcaps, egIsoConeSizeInEndcap=generalData_->isoCfg.intRadiusEcalEndcaps;
  eventData_->ecalBarrelIsol03 = new EgammaRecHitIsolation(egIsoConeSizeOutSmall,egIsoConeSizeInBarrel,egIsoJurassicWidth,egIsoPtMinBarrel,egIsoEMinBarrel,eventSetupData_->caloGeom,*(eventData_->barrelRecHits),eventSetupData_->sevLevel.product(),DetId::Ecal);
  eventData_->ecalBarrelIsol04 = new EgammaRecHitIsolation(egIsoConeSizeOutLarge,egIsoConeSizeInBarrel,egIsoJurassicWidth,egIsoPtMinBarrel,egIsoEMinBarrel,eventSetupData_->caloGeom,*(eventData_->barrelRecHits),eventSetupData_->sevLevel.product(),DetId::Ecal);
  eventData_->ecalEndcapIsol03 = new EgammaRecHitIsolation(egIsoConeSizeOutSmall,egIsoConeSizeInEndcap,egIsoJurassicWidth,egIsoPtMinEndcap,egIsoEMinEndcap,eventSetupData_->caloGeom,*(eventData_->endcapRecHits),eventSetupData_->sevLevel.product(),DetId::Ecal);
  eventData_->ecalEndcapIsol04 = new EgammaRecHitIsolation(egIsoConeSizeOutLarge,egIsoConeSizeInEndcap,egIsoJurassicWidth,egIsoPtMinEndcap,egIsoEMinEndcap,eventSetupData_->caloGeom,*(eventData_->endcapRecHits),eventSetupData_->sevLevel.product(),DetId::Ecal);
  eventData_->ecalBarrelIsol03->setUseNumCrystals(generalData_->isoCfg.useNumCrystals);
  eventData_->ecalBarrelIsol03->setVetoClustered(generalData_->isoCfg.vetoClustered);
  eventData_->ecalBarrelIsol03->doSeverityChecks(eventData_->barrelRecHits.product(),generalData_->recHitsCfg.recHitSeverityToBeExcludedBarrel);
  eventData_->ecalBarrelIsol03->doFlagChecks(generalData_->recHitsCfg.recHitFlagsToBeExcludedBarrel);
  eventData_->ecalBarrelIsol04->setUseNumCrystals(generalData_->isoCfg.useNumCrystals);
  eventData_->ecalBarrelIsol04->setVetoClustered(generalData_->isoCfg.vetoClustered);
  eventData_->ecalBarrelIsol04->doSeverityChecks(eventData_->barrelRecHits.product(),generalData_->recHitsCfg.recHitSeverityToBeExcludedBarrel);
  eventData_->ecalBarrelIsol04->doFlagChecks(generalData_->recHitsCfg.recHitFlagsToBeExcludedBarrel);
  eventData_->ecalEndcapIsol03->setUseNumCrystals(generalData_->isoCfg.useNumCrystals);
  eventData_->ecalEndcapIsol03->setVetoClustered(generalData_->isoCfg.vetoClustered);
  eventData_->ecalEndcapIsol03->doSeverityChecks(eventData_->endcapRecHits.product(),generalData_->recHitsCfg.recHitSeverityToBeExcludedEndcaps);
  eventData_->ecalEndcapIsol03->doFlagChecks(generalData_->recHitsCfg.recHitFlagsToBeExcludedEndcaps);
  eventData_->ecalEndcapIsol04->setUseNumCrystals(generalData_->isoCfg.useNumCrystals);
  eventData_->ecalEndcapIsol04->setVetoClustered(generalData_->isoCfg.vetoClustered);
  eventData_->ecalEndcapIsol04->doSeverityChecks(eventData_->endcapRecHits.product(),generalData_->recHitsCfg.recHitSeverityToBeExcludedEndcaps);
  eventData_->ecalEndcapIsol04->doFlagChecks(generalData_->recHitsCfg.recHitFlagsToBeExcludedEndcaps);

  //Fill in the Isolation Value Maps for PF and EcalDriven electrons
  std::vector<edm::InputTag> inputTagIsoVals;
  if(! generalData_->inputCfg.pfIsoVals.empty() ) {
    inputTagIsoVals.push_back(generalData_->inputCfg.pfIsoVals.getParameter<edm::InputTag>("pfSumChargedHadronPt"));
    inputTagIsoVals.push_back(generalData_->inputCfg.pfIsoVals.getParameter<edm::InputTag>("pfSumPhotonEt"));
    inputTagIsoVals.push_back(generalData_->inputCfg.pfIsoVals.getParameter<edm::InputTag>("pfSumNeutralHadronEt"));

    eventData_->pfIsolationValues.resize(inputTagIsoVals.size());

    for (size_t j = 0; j<inputTagIsoVals.size(); ++j) {
      event.getByLabel(inputTagIsoVals[j], eventData_->pfIsolationValues[j]);
    }

  }

  if(! generalData_->inputCfg.edIsoVals.empty() ) {
    inputTagIsoVals.clear();
    inputTagIsoVals.push_back(generalData_->inputCfg.edIsoVals.getParameter<edm::InputTag>("edSumChargedHadronPt"));
    inputTagIsoVals.push_back(generalData_->inputCfg.edIsoVals.getParameter<edm::InputTag>("edSumPhotonEt"));
    inputTagIsoVals.push_back(generalData_->inputCfg.edIsoVals.getParameter<edm::InputTag>("edSumNeutralHadronEt"));

    eventData_->edIsolationValues.resize(inputTagIsoVals.size());

    for (size_t j = 0; j<inputTagIsoVals.size(); ++j) {
      event.getByLabel(inputTagIsoVals[j], eventData_->edIsolationValues[j]);
    }
  }
 }

void GsfElectronAlgo::endEvent()
 {
  if (eventData_==0)
   { throw cms::Exception("GsfElectronAlgo|InternalError")<<"lacking event data" ; }
  delete eventData_ ;
  eventData_ = 0 ;
 }

void GsfElectronAlgo::displayInternalElectrons( const std::string & title ) const
 {
  LogTrace("GsfElectronAlgo") << "========== " << title << " ==========";
  LogTrace("GsfElectronAlgo") << "Event: " << eventData_->event->id();
  LogTrace("GsfElectronAlgo") << "Number of electrons: " << eventData_->electrons->size() ;
  GsfElectronPtrCollection::const_iterator it ;
  for ( it = eventData_->electrons->begin(); it != eventData_->electrons->end(); it++ )
   {
    LogTrace("GsfElectronAlgo") << "Electron with charge, pt, eta, phi: "  << (*it)->charge() << " , "
        << (*it)->pt() << " , " << (*it)->eta() << " , " << (*it)->phi();
   }
  LogTrace("GsfElectronAlgo") << "=================================================";
 }

void GsfElectronAlgo::completeElectrons(const gsfAlgoHelpers::HeavyObjectCache* hoc)
 {
  if (electronData_!=0)
   { throw cms::Exception("GsfElectronAlgo|InternalError")<<"unexpected electron data" ; }

  const GsfElectronCoreCollection * coreCollection = eventData_->coreElectrons.product() ;
  for ( unsigned int i=0 ; i<coreCollection->size() ; ++i )
   {
    // check there is no existing electron with this core
    const GsfElectronCoreRef coreRef = edm::Ref<GsfElectronCoreCollection>(eventData_->coreElectrons,i) ;
    bool coreFound = false ;
    GsfElectronPtrCollection::const_iterator itrEle ;
    for
     ( itrEle = eventData_->electrons->begin() ;
       itrEle != eventData_->electrons->end() ;
       itrEle++ )
     {
      if ((*itrEle)->core()==coreRef)
       {
        coreFound = true ;
        break ;
       }
     }
    if (coreFound) continue ;

    // check there is a super-cluster
    if (coreRef->superCluster().isNull()) continue ;

    // prepare internal structure for electron specific data
    delete electronData_ ;
    electronData_ = new ElectronData(coreRef,*eventData_->beamspot) ;

    // calculate and check Trajectory StatesOnSurface....
    if ( !electronData_->calculateTSOS( eventSetupData_->mtsTransform, eventSetupData_->constraintAtVtx ) ) continue ;

    createElectron(hoc) ;

   } // loop over tracks

  delete electronData_ ;
  electronData_ = 0 ;
 }

void GsfElectronAlgo::clonePreviousElectrons()
 {
  const GsfElectronCollection * oldElectrons = eventData_->previousElectrons.product() ;
  const GsfElectronCoreCollection * newCores = eventData_->coreElectrons.product() ;
  GsfElectronCollection::const_iterator oldElectron ;
  for
   ( oldElectron = oldElectrons->begin() ;
     oldElectron != oldElectrons->end() ;
     ++oldElectron )
   {
    const GsfElectronCoreRef oldCoreRef = oldElectron->core() ;
    const GsfTrackRef oldElectronGsfTrackRef = oldCoreRef->gsfTrack() ;
    unsigned int icore ;
    for ( icore=0 ; icore<newCores->size() ; ++icore )
     {
      if (oldElectronGsfTrackRef==(*newCores)[icore].gsfTrack())
       {
        const GsfElectronCoreRef coreRef = edm::Ref<GsfElectronCoreCollection>(eventData_->coreElectrons,icore) ;
        eventData_->electrons->push_back(new GsfElectron(*oldElectron,coreRef)) ;
        break ;
       }
     }
   }
 }


// now deprecated
void GsfElectronAlgo::addPflowInfo()
 {
  bool found ;
  const GsfElectronCollection * edElectrons = eventData_->previousElectrons.product() ;
  const GsfElectronCollection * pfElectrons = eventData_->pflowElectrons.product() ;
  GsfElectronCollection::const_iterator pfElectron, edElectron ;
  unsigned int edIndex, pfIndex ;

  GsfElectronPtrCollection::iterator el ;
  for
   ( el = eventData_->electrons->begin() ;
     el != eventData_->electrons->end() ;
     el++ )
   {

    // Retreive info from pflow electrons
    found = false ;
    for
     ( pfIndex = 0, pfElectron = pfElectrons->begin() ; pfElectron != pfElectrons->end() ; pfIndex++, pfElectron++ )
     {
      if (pfElectron->gsfTrack()==(*el)->gsfTrack())
       {
        if (found)
         {
          edm::LogWarning("GsfElectronProducer")<<"associated pfGsfElectron already found" ;
         }
        else
         {
          found = true ;

	  // Isolation Values
        if( (eventData_->pfIsolationValues).size() != 0 )
        {
	  reco::GsfElectronRef 
		pfElectronRef(eventData_->pflowElectrons, pfIndex);
	  reco::GsfElectron::PflowIsolationVariables isoVariables;
	  isoVariables.sumChargedHadronPt =(*(eventData_->pfIsolationValues)[0])[pfElectronRef];
	  isoVariables.sumPhotonEt        =(*(eventData_->pfIsolationValues)[1])[pfElectronRef];
	  isoVariables.sumNeutralHadronEt =(*(eventData_->pfIsolationValues)[2])[pfElectronRef];
	  (*el)->setPfIsolationVariables(isoVariables);
        }

//          (*el)->setPfIsolationVariables(pfElectron->pfIsolationVariables()) ;
          (*el)->setMvaInput(pfElectron->mvaInput()) ;
          (*el)->setMvaOutput(pfElectron->mvaOutput()) ;
          if ((*el)->ecalDrivenSeed())
           { (*el)->setP4(GsfElectron::P4_PFLOW_COMBINATION,pfElectron->p4(GsfElectron::P4_PFLOW_COMBINATION),pfElectron->p4Error(GsfElectron::P4_PFLOW_COMBINATION),false) ; }
          else
           { (*el)->setP4(GsfElectron::P4_PFLOW_COMBINATION,pfElectron->p4(GsfElectron::P4_PFLOW_COMBINATION),pfElectron->p4Error(GsfElectron::P4_PFLOW_COMBINATION),true) ; }
          double noCutMin = -999999999. ;
          if ((*el)->mva_e_pi()<noCutMin) { throw cms::Exception("GsfElectronAlgo|UnexpectedMvaValue")<<"unexpected MVA value: "<<(*el)->mva_e_pi() ; }
         }
       }
     }

     // Isolation Values
     // Retreive not found info from ed electrons
   if( (eventData_->edIsolationValues).size() != 0 )
   {
     edIndex = 0, edElectron = edElectrons->begin() ;
     while ((found == false)&&(edElectron != edElectrons->end()))
     {
        if (edElectron->gsfTrack()==(*el)->gsfTrack())
        {
          found = true ; 

          // CONSTRUCTION D UNE REF dans le handle eventData_->previousElectrons avec l'indice edIndex,
          // puis recuperation dans la ValueMap ED

	  reco::GsfElectronRef 
		edElectronRef(eventData_->previousElectrons, edIndex);
	  reco::GsfElectron::PflowIsolationVariables isoVariables;
	  isoVariables.sumChargedHadronPt =(*(eventData_->edIsolationValues)[0])[edElectronRef];
	  isoVariables.sumPhotonEt        =(*(eventData_->edIsolationValues)[1])[edElectronRef];
	  isoVariables.sumNeutralHadronEt =(*(eventData_->edIsolationValues)[2])[edElectronRef];
	  (*el)->setPfIsolationVariables(isoVariables);
        } 

        edIndex++ ; 
        edElectron++ ;
     }
   }

    // Preselection
    setPflowPreselectionFlag(*el) ;

   }
 }

bool GsfElectronAlgo::isPreselected( GsfElectron * ele )
 {
	bool passCutBased=ele->passingCutBasedPreselection();
	bool passPF=ele->passingPflowPreselection(); //it is worth nothing for gedGsfElectrons, this does nothing as its not set till GedGsfElectron finaliser, this is always false
	if(generalData_->strategyCfg.gedElectronMode){
         	bool passmva=ele->passingMvaPreselection();
		if(!ele->ecalDrivenSeed()){
		  if(ele->pt() > generalData_->strategyCfg.MaxElePtForOnlyMVA) 
		    return passmva && passCutBased;
		  else
		    return passmva;
		}	
		else{
		  return passCutBased || passPF || passmva;
		}
	}
	else{
		return passCutBased || passPF;
	}

	return true; 
 }

void GsfElectronAlgo::removeNotPreselectedElectrons()
 {
  GsfElectronPtrCollection::size_type ei = 1, emax = eventData_->electrons->size() ;
  GsfElectronPtrCollection::iterator eitr = eventData_->electrons->begin() ;
  while (eitr!=eventData_->electrons->end())
   {
    LogTrace("GsfElectronAlgo")<<"========== removed not preselected "<<ei<<"/"<<emax<<"==========" ;
    if (isPreselected(*eitr))
     { ++eitr ; ++ei ; }
    else
     { delete (*eitr) ; eitr = eventData_->electrons->erase(eitr) ; ++ei ; }
   }
 }


void GsfElectronAlgo::setCutBasedPreselectionFlag( GsfElectron * ele, const reco::BeamSpot & bs )
 {
  // default value
  ele->setPassCutBasedPreselection(false) ;

  // kind of seeding
  bool eg = ele->core()->ecalDrivenSeed() ;
  bool pf = ele->core()->trackerDrivenSeed() && !ele->core()->ecalDrivenSeed() ;
  bool gedMode = generalData_->strategyCfg.gedElectronMode;
  if (eg&&pf) { throw cms::Exception("GsfElectronAlgo|BothEcalAndPureTrackerDriven")<<"An electron cannot be both egamma and purely pflow" ; }
  if ((!eg)&&(!pf)) { throw cms::Exception("GsfElectronAlgo|NeitherEcalNorPureTrackerDriven")<<"An electron cannot be neither egamma nor purely pflow" ; }

  const CutsConfiguration * cfg = ((eg||gedMode)?&generalData_->cutsCfg:&generalData_->cutsCfgPflow);

  // Et cut
  double etaValue = EleRelPoint(ele->superCluster()->position(),bs.position()).eta() ;
  double etValue = ele->superCluster()->energy()/cosh(etaValue) ;
  LogTrace("GsfElectronAlgo") << "Et : " << etValue ;
  if (ele->isEB() && (etValue < cfg->minSCEtBarrel)) return ;
  if (ele->isEE() && (etValue < cfg->minSCEtEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Et criteria are satisfied";

  // E/p cut
  double eopValue = ele->eSuperClusterOverP() ;
  LogTrace("GsfElectronAlgo") << "E/p : " << eopValue ;
  if (ele->isEB() && (eopValue > cfg->maxEOverPBarrel)) return ;
  if (ele->isEE() && (eopValue > cfg->maxEOverPEndcaps)) return ;
  if (ele->isEB() && (eopValue < cfg->minEOverPBarrel)) return ;
  if (ele->isEE() && (eopValue < cfg->minEOverPEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "E/p criteria are satisfied";

  // HoE cuts
  LogTrace("GsfElectronAlgo") << "HoE1 : " << ele->hcalDepth1OverEcal() << ", HoE2 : " << ele->hcalDepth2OverEcal();
  double had = ele->hcalOverEcal()*ele->superCluster()->energy() ;
  const reco::CaloCluster & seedCluster = *(ele->superCluster()->seed()) ;
  int detector = seedCluster.hitsAndFractions()[0].first.subdetId() ;
  bool HoEveto = false ;
  if (detector==EcalBarrel && (had<cfg->maxHBarrel || (had/ele->superCluster()->energy())<cfg->maxHOverEBarrel)) HoEveto=true;
  else if (detector==EcalEndcap && (had<cfg->maxHEndcaps || (had/ele->superCluster()->energy())<cfg->maxHOverEEndcaps)) HoEveto=true;
  if ( !HoEveto ) return ;
  LogTrace("GsfElectronAlgo") << "H/E criteria are satisfied";

  // delta eta criteria
  double deta = ele->deltaEtaSuperClusterTrackAtVtx() ;
  LogTrace("GsfElectronAlgo") << "delta eta : " << deta ;
  if (ele->isEB() && (std::abs(deta) > cfg->maxDeltaEtaBarrel)) return ;
  if (ele->isEE() && (std::abs(deta) > cfg->maxDeltaEtaEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Delta eta criteria are satisfied";

  // delta phi criteria
  double dphi = ele->deltaPhiSuperClusterTrackAtVtx();
  LogTrace("GsfElectronAlgo") << "delta phi : " << dphi;
  if (ele->isEB() && (std::abs(dphi) > cfg->maxDeltaPhiBarrel)) return ;
  if (ele->isEE() && (std::abs(dphi) > cfg->maxDeltaPhiEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Delta phi criteria are satisfied";

  // sigma ieta ieta
  LogTrace("GsfElectronAlgo") << "sigma ieta ieta : " << ele->sigmaIetaIeta();
  if (ele->isEB() && (ele->sigmaIetaIeta() > cfg->maxSigmaIetaIetaBarrel)) return ;
  if (ele->isEE() && (ele->sigmaIetaIeta() > cfg->maxSigmaIetaIetaEndcaps)) return ;
  LogTrace("GsfElectronAlgo") << "Sigma ieta ieta criteria are satisfied";

  // fiducial
  if (!ele->isEB() && cfg->isBarrel) return ;
  if (!ele->isEE() && cfg->isEndcaps) return ;
  if (cfg->isFiducial && (ele->isEBEEGap()||ele->isEBEtaGap()||ele->isEBPhiGap()||ele->isEERingGap()||ele->isEEDeeGap())) return ;
  LogTrace("GsfElectronAlgo") << "Fiducial flags criteria are satisfied";

  // seed in TEC
  edm::RefToBase<TrajectorySeed> seed = ele->gsfTrack()->extra()->seedRef() ;
  ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>() ;
  if (eg && !generalData_->cutsCfg.seedFromTEC)
   {
    if (elseed.isNull())
     { throw cms::Exception("GsfElectronAlgo|NotElectronSeed")<<"The GsfTrack seed is not an ElectronSeed ?!" ; }
    else
     { if (elseed->subDet2()==6) return ; }
   }

  // transverse impact parameter
  if (std::abs(ele->gsfTrack()->dxy(bs.position()))>cfg->maxTIP) return ;
  LogTrace("GsfElectronAlgo") << "TIP criterion is satisfied" ;

  LogTrace("GsfElectronAlgo") << "All cut based criteria are satisfied" ;
  ele->setPassCutBasedPreselection(true) ;
 }

void GsfElectronAlgo::setPflowPreselectionFlag( GsfElectron * ele )
 {
  ele->setPassMvaPreselection(false) ;

  if (ele->core()->ecalDrivenSeed())
   { if (ele->mvaOutput().mva_e_pi>=generalData_->cutsCfg.minMVA) ele->setPassMvaPreselection(true) ; }
  else
   { if (ele->mvaOutput().mva_e_pi>=generalData_->cutsCfgPflow.minMVA) ele->setPassMvaPreselection(true) ; }

  if (ele->passingMvaPreselection())
   { LogTrace("GsfElectronAlgo") << "Main mva criterion is satisfied" ; }

  ele->setPassPflowPreselection(ele->passingMvaPreselection()) ;

 }

void GsfElectronAlgo::setMVAInputs(const std::map<reco::GsfTrackRef,reco::GsfElectron::MvaInput> & mvaInputs) 
{
  GsfElectronPtrCollection::iterator el ;
  for
    ( el = eventData_->electrons->begin() ;
      el != eventData_->electrons->end() ;
      el++ )
    {
      std::map<reco::GsfTrackRef,reco::GsfElectron::MvaInput>::const_iterator itcheck=mvaInputs.find((*el)->gsfTrack());
      (*el)->setMvaInput(itcheck->second);
    }
}

void GsfElectronAlgo::setMVAOutputs(const gsfAlgoHelpers::HeavyObjectCache* hoc,
                                    const std::map<reco::GsfTrackRef,reco::GsfElectron::MvaOutput> & mvaOutputs)
{
  GsfElectronPtrCollection::iterator el ;
  for
    ( el = eventData_->electrons->begin() ;
      el != eventData_->electrons->end() ;
      el++ )
    {
	if(generalData_->strategyCfg.gedElectronMode==true){
                float mva_NIso_Value=	hoc->sElectronMVAEstimator->mva( *(*el), *(eventData_->vertices));
		float mva_Iso_Value =   hoc->iElectronMVAEstimator->mva( *(*el), eventData_->vertices->size() );
	        GsfElectron::MvaOutput mvaOutput ;
	        mvaOutput.mva_e_pi = mva_NIso_Value ;
		mvaOutput.mva_Isolated = mva_Iso_Value ;
	        (*el)->setMvaOutput(mvaOutput);
	}
	else{
		std::map<reco::GsfTrackRef,reco::GsfElectron::MvaOutput>::const_iterator itcheck=mvaOutputs.find((*el)->gsfTrack());
                (*el)->setMvaOutput(itcheck->second);
	}
    }
}

void GsfElectronAlgo::createElectron(const gsfAlgoHelpers::HeavyObjectCache* hoc)
 {
  // eventually check ctf track
  if (generalData_->strategyCfg.ctfTracksCheck)
   { electronData_->checkCtfTrack(eventData_->currentCtfTracks) ; }

  // charge ID
  int eleCharge ;
  GsfElectron::ChargeInfo eleChargeInfo ;
  electronData_->computeCharge(eleCharge,eleChargeInfo) ;

  // electron basic cluster
  CaloClusterPtr elbcRef = electronData_->getEleBasicCluster(eventSetupData_->mtsTransform) ;

  // Seed cluster
  const reco::CaloCluster & seedCluster = *(electronData_->superClusterRef->seed()) ;

  // seed Xtal
  // temporary, till CaloCluster->seed() is made available
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;

  electronData_->calculateMode(eventSetupData_->mtsMode) ;


  //====================================================
  // Candidate attributes
  //====================================================

  Candidate::LorentzVector momentum = electronData_->calculateMomentum() ;


  //====================================================
  // Track-Cluster Matching
  //====================================================

  reco::GsfElectron::TrackClusterMatching tcMatching ;
  tcMatching.electronCluster = elbcRef ;
  tcMatching.eSuperClusterOverP = (electronData_->vtxMom.mag()>0)?(electronData_->superClusterRef->energy()/electronData_->vtxMom.mag()):(-1.) ;
  tcMatching.eSeedClusterOverP = (electronData_->vtxMom.mag()>0.)?(seedCluster.energy()/electronData_->vtxMom.mag()):(-1) ;
  tcMatching.eSeedClusterOverPout = (electronData_->seedMom.mag()>0.)?(seedCluster.energy()/electronData_->seedMom.mag()):(-1.) ;
  tcMatching.eEleClusterOverPout = (electronData_->eleMom.mag()>0.)?(elbcRef->energy()/electronData_->eleMom.mag()):(-1.) ;

  EleRelPointPair scAtVtx(electronData_->superClusterRef->position(),electronData_->sclPos,eventData_->beamspot->position()) ;
  tcMatching.deltaEtaSuperClusterAtVtx = scAtVtx.dEta() ;
  tcMatching.deltaPhiSuperClusterAtVtx = scAtVtx.dPhi() ;

  EleRelPointPair seedAtCalo(seedCluster.position(),electronData_->seedPos,eventData_->beamspot->position()) ;
  tcMatching.deltaEtaSeedClusterAtCalo = seedAtCalo.dEta() ;
  tcMatching.deltaPhiSeedClusterAtCalo = seedAtCalo.dPhi() ;

  EleRelPointPair ecAtCalo(elbcRef->position(),electronData_->elePos,eventData_->beamspot->position()) ;
  tcMatching.deltaEtaEleClusterAtCalo = ecAtCalo.dEta() ;
  tcMatching.deltaPhiEleClusterAtCalo = ecAtCalo.dPhi() ;


  //=======================================================
  // Track extrapolations
  //=======================================================

  reco::GsfElectron::TrackExtrapolations tkExtra ;
  ele_convert(electronData_->vtxPos,tkExtra.positionAtVtx) ;
  ele_convert(electronData_->sclPos,tkExtra.positionAtCalo) ;
  ele_convert(electronData_->vtxMom,tkExtra.momentumAtVtx) ;
  ele_convert(electronData_->sclMom,tkExtra.momentumAtCalo) ;
  ele_convert(electronData_->seedMom,tkExtra.momentumOut) ;
  ele_convert(electronData_->eleMom,tkExtra.momentumAtEleClus) ;
  ele_convert(electronData_->vtxMomWithConstraint,tkExtra.momentumAtVtxWithConstraint) ;


  //=======================================================
  // Closest Ctf Track
  //=======================================================

  reco::GsfElectron::ClosestCtfTrack ctfInfo ;
  ctfInfo.ctfTrack = electronData_->ctfTrackRef  ;
  ctfInfo.shFracInnerHits = electronData_->shFracInnerHits ;


  //====================================================
  // FiducialFlags, using nextToBoundary definition of gaps
  //====================================================

  reco::GsfElectron::FiducialFlags fiducialFlags ;
  int detector = seedXtalId.subdetId() ;
  double feta=std::abs(electronData_->superClusterRef->position().eta()) ;
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
  else
   { throw cms::Exception("GsfElectronAlgo|UnknownXtalRegion")<<"createElectron(): do not know if it is a barrel or endcap seed cluster !!!!" ; }


  //====================================================
  // ShowerShape
  //====================================================

  reco::GsfElectron::ShowerShape showerShape ;
  calculateShowerShape(electronData_->superClusterRef,!(electronData_->coreRef->ecalDrivenSeed()),showerShape) ;

  reco::GsfElectron::ShowerShape full5x5_showerShape ;
  calculateShowerShape_full5x5(electronData_->superClusterRef,!(electronData_->coreRef->ecalDrivenSeed()),full5x5_showerShape) ;

  //====================================================
  // ConversionRejection
  //====================================================

  eventData_->retreiveOriginalTrackCollections(electronData_->ctfTrackRef,electronData_->coreRef->gsfTrack()) ;

  ConversionFinder conversionFinder ;
  double BInTesla = eventSetupData_->magField->inTesla(GlobalPoint(0.,0.,0.)).z() ;
  edm::Handle<reco::TrackCollection> ctfTracks = eventData_->originalCtfTracks ;
  if (!ctfTracks.isValid()) { ctfTracks = eventData_->currentCtfTracks ; }

  // values of conversionInfo.flag()
  // -9999 : Partner track was not found
  // 0     : Partner track found in the CTF collection using
  // 1     : Partner track found in the CTF collection using
  // 2     : Partner track found in the GSF collection using
  // 3     : Partner track found in the GSF collection using the electron's GSF track
  ConversionInfo conversionInfo = conversionFinder.getConversionInfo
   (*electronData_->coreRef,ctfTracks,eventData_->originalGsfTracks,BInTesla) ;

  reco::GsfElectron::ConversionRejection conversionVars ;
  conversionVars.flags = conversionInfo.flag()  ;
  conversionVars.dist = conversionInfo.dist()  ;
  conversionVars.dcot = conversionInfo.dcot()  ;
  conversionVars.radius = conversionInfo.radiusOfConversion()  ;
  if ((conversionVars.flags==0)or(conversionVars.flags==1))
    conversionVars.partner = TrackBaseRef(conversionInfo.conversionPartnerCtfTk())  ;
  else if ((conversionVars.flags==2)or(conversionVars.flags==3))
    conversionVars.partner = TrackBaseRef(conversionInfo.conversionPartnerGsfTk())  ;


  //====================================================
  // Go !
  //====================================================

  GsfElectron * ele = new
    GsfElectron
     ( eleCharge,eleChargeInfo,electronData_->coreRef,
       tcMatching, tkExtra, ctfInfo,
       fiducialFlags,showerShape, full5x5_showerShape,
       conversionVars ) ;
  // Will be overwritten later in the case of the regression
  ele->setCorrectedEcalEnergyError(generalData_->superClusterErrorFunction->getValue(*(ele->superCluster()),0)) ;
  ele->setP4(GsfElectron::P4_FROM_SUPER_CLUSTER,momentum,0,true) ;


  //====================================================
  // brems fractions
  //====================================================

  if (electronData_->innMom.mag()>0.)
   { ele->setTrackFbrem((electronData_->innMom.mag()-electronData_->outMom.mag())/electronData_->innMom.mag()) ; }

  // the supercluster is the refined one The seed is not necessarily the first cluster
  // hence the use of the electronCluster
  SuperClusterRef sc = ele->superCluster() ;
  if (!(sc.isNull()))
   {
    CaloClusterPtr cl = ele->electronCluster() ;
    if (sc->clustersSize()>1)
     { 
       float pf_fbrem =( sc->energy() - cl->energy() ) / sc->energy();
       ele->setSuperClusterFbrem( pf_fbrem ) ;
     }
    else
      { 
	ele->setSuperClusterFbrem(0) ; 
      }
   }

  //====================================================
  // classification and corrections
  //====================================================
  // classification
  ElectronClassification theClassifier ;
  theClassifier.classify(*ele) ;
  theClassifier.refineWithPflow(*ele) ;
  // ecal energy
  ElectronEnergyCorrector theEnCorrector(generalData_->crackCorrectionFunction) ;
  if (generalData_->strategyCfg.useEcalRegression) // new 
    { 
      generalData_->regHelper->applyEcalRegression(*ele,
						   eventData_->vertices,
						   eventData_->barrelRecHits,
						   eventData_->endcapRecHits);
    }
  else  // original implementation
    {
      if (ele->core()->ecalDrivenSeed())
	{
	  if (generalData_->strategyCfg.ecalDrivenEcalEnergyFromClassBasedParameterization)
	    { theEnCorrector.classBasedParameterizationEnergy(*ele,*eventData_->beamspot) ; }
	  if (generalData_->strategyCfg.ecalDrivenEcalErrorFromClassBasedParameterization)
	    { theEnCorrector.classBasedParameterizationUncertainty(*ele) ; }
	}
      else
	{
	  if (generalData_->strategyCfg.pureTrackerDrivenEcalErrorFromSimpleParameterization)
	    { theEnCorrector.simpleParameterizationUncertainty(*ele) ; }
	}
    }
  
  // momentum
  // Keep the default correction running first. The track momentum error is computed in there
  if (ele->core()->ecalDrivenSeed())
    {
      ElectronMomentumCorrector theMomCorrector;
      theMomCorrector.correct(*ele,electronData_->vtxTSOS);
    }
  if(generalData_->strategyCfg.useCombinationRegression)  // new 
    {
      generalData_->regHelper->applyCombinationRegression(*ele);
    }

  //====================================================
  // now isolation variables
  //====================================================

  reco::GsfElectron::IsolationVariables dr03, dr04 ;
  dr03.tkSumPt = eventData_->tkIsolation03->getPtTracks(ele);
  dr03.hcalDepth1TowerSumEt = eventData_->hadDepth1Isolation03->getTowerEtSum(ele) ;
  dr03.hcalDepth2TowerSumEt = eventData_->hadDepth2Isolation03->getTowerEtSum(ele) ;
  dr03.hcalDepth1TowerSumEtBc = eventData_->hadDepth1Isolation03Bc->getTowerEtSum(ele,&(showerShape.hcalTowersBehindClusters)) ;
  dr03.hcalDepth2TowerSumEtBc = eventData_->hadDepth2Isolation03Bc->getTowerEtSum(ele,&(showerShape.hcalTowersBehindClusters)) ;
  dr03.ecalRecHitSumEt = eventData_->ecalBarrelIsol03->getEtSum(ele)+eventData_->ecalEndcapIsol03->getEtSum(ele);
  dr04.tkSumPt = eventData_->tkIsolation04->getPtTracks(ele);
  dr04.hcalDepth1TowerSumEt = eventData_->hadDepth1Isolation04->getTowerEtSum(ele);
  dr04.hcalDepth2TowerSumEt = eventData_->hadDepth2Isolation04->getTowerEtSum(ele);
  dr04.hcalDepth1TowerSumEtBc = eventData_->hadDepth1Isolation04Bc->getTowerEtSum(ele,&(showerShape.hcalTowersBehindClusters)) ;
  dr04.hcalDepth2TowerSumEtBc = eventData_->hadDepth2Isolation04Bc->getTowerEtSum(ele,&(showerShape.hcalTowersBehindClusters)) ;
  dr04.ecalRecHitSumEt = eventData_->ecalBarrelIsol04->getEtSum(ele)+eventData_->ecalEndcapIsol04->getEtSum(ele);
  ele->setIsolation03(dr03);
  ele->setIsolation04(dr04);


  //====================================================
  // preselection flag
  //====================================================

  setCutBasedPreselectionFlag(ele,*eventData_->beamspot) ;
  //setting mva flag, currently GedGsfElectron and GsfElectron pre-selection flags have desynced
  //this is for GedGsfElectrons, GsfElectrons (ie old pre 7X std reco) resets this later on
  //in the function "addPfInfo"
  //yes this is awful, we'll fix it once we work out how to...
  float mvaValue = hoc->sElectronMVAEstimator->mva( *(ele),*(eventData_->vertices));
  ele->setPassMvaPreselection(mvaValue>generalData_->strategyCfg.PreSelectMVA);

  //====================================================
  // Pixel match variables
  //====================================================
  setPixelMatchInfomation(ele) ;

  LogTrace("GsfElectronAlgo")<<"Constructed new electron with energy  "<< ele->p4().e() ;

  eventData_->electrons->push_back(ele) ;
 }


//=======================================================================================
// Ambiguity solving
//=======================================================================================

//bool better_electron( const reco::GsfElectron * e1, const reco::GsfElectron * e2 )
// { return (std::abs(e1->eSuperClusterOverP()-1)<std::abs(e2->eSuperClusterOverP()-1)) ; }

void GsfElectronAlgo::setAmbiguityData( bool ignoreNotPreselected )
 {
  GsfElectronPtrCollection::iterator e1, e2 ;
  if (generalData_->strategyCfg.ambSortingStrategy==0)
   { eventData_->electrons->sort(EgAmbiguityTools::isBetter) ; }
  else if (generalData_->strategyCfg.ambSortingStrategy==1)
   { eventData_->electrons->sort(EgAmbiguityTools::isInnerMost(eventSetupData_->trackerHandle)) ; }
  else
   { throw cms::Exception("GsfElectronAlgo|UnknownAmbiguitySortingStrategy")<<"value of generalData_->strategyCfg.ambSortingStrategy is : "<<generalData_->strategyCfg.ambSortingStrategy ; }

  // init
  for
   ( e1 = eventData_->electrons->begin() ;
     e1 != eventData_->electrons->end() ;
     ++e1 )
   {
    (*e1)->clearAmbiguousGsfTracks() ;
    (*e1)->setAmbiguous(false) ;
   }

  // get ambiguous from GsfPfRecTracks
  if (generalData_->strategyCfg.useGsfPfRecTracks)
   {
    for
     ( e1 = eventData_->electrons->begin() ;
       e1 != eventData_->electrons->end() ;
       ++e1 )
     {
      bool found = false ;
      const GsfPFRecTrackCollection * gsfPfRecTrackCollection = eventData_->gsfPfRecTracks.product() ;
      GsfPFRecTrackCollection::const_iterator gsfPfRecTrack ;
      for ( gsfPfRecTrack=gsfPfRecTrackCollection->begin() ;
            gsfPfRecTrack!=gsfPfRecTrackCollection->end() ;
            ++gsfPfRecTrack )
       {
        if (gsfPfRecTrack->gsfTrackRef()==(*e1)->gsfTrack())
         {
          if (found)
           {
            edm::LogWarning("GsfElectronAlgo")<<"associated gsfPfRecTrack already found" ;
           }
          else
           {
            found = true ;
            const std::vector<reco::GsfPFRecTrackRef> & duplicates(gsfPfRecTrack->convBremGsfPFRecTrackRef()) ;
            std::vector<reco::GsfPFRecTrackRef>::const_iterator duplicate ;
            for ( duplicate = duplicates.begin() ; duplicate != duplicates.end() ; duplicate ++ )
             { (*e1)->addAmbiguousGsfTrack((*duplicate)->gsfTrackRef()) ; }
           }
         }
       }
     }
   }
  // or search overlapping clusters
  else
   {
    for
     ( e1 = eventData_->electrons->begin() ;
       e1 != eventData_->electrons->end() ;
       ++e1 )
     {
      if ((*e1)->ambiguous()) continue ;
      if ( ignoreNotPreselected && !isPreselected(*e1) ) continue ;

      SuperClusterRef scRef1 = (*e1)->superCluster();
      CaloClusterPtr eleClu1 = (*e1)->electronCluster();
      LogDebug("GsfElectronAlgo")
        << "Blessing electron with E/P " << (*e1)->eSuperClusterOverP()
        << ", cluster " << scRef1.get()
        << " & track " << (*e1)->gsfTrack().get() ;

      for
       ( e2 = e1, ++e2 ;
         e2 != eventData_->electrons->end() ;
         ++e2 )
       {
        if ((*e2)->ambiguous()) continue ;
        if ( ignoreNotPreselected && !isPreselected(*e2) ) continue ;

        SuperClusterRef scRef2 = (*e2)->superCluster();
        CaloClusterPtr eleClu2 = (*e2)->electronCluster();

        // search if same cluster
        bool sameCluster = false ;
        if (generalData_->strategyCfg.ambClustersOverlapStrategy==0)
         { sameCluster = (scRef1==scRef2) ; }
        else if (generalData_->strategyCfg.ambClustersOverlapStrategy==1)
         {
          float eMin = 1. ;
          float threshold = eMin*cosh(EleRelPoint(scRef1->position(),eventData_->beamspot->position()).eta()) ;
          sameCluster =
           ( (EgAmbiguityTools::sharedEnergy(&(*eleClu1),&(*eleClu2),eventData_->barrelRecHits,eventData_->endcapRecHits)>=threshold) ||
             (EgAmbiguityTools::sharedEnergy(&(*scRef1->seed()),&(*eleClu2),eventData_->barrelRecHits,eventData_->endcapRecHits)>=threshold) ||
             (EgAmbiguityTools::sharedEnergy(&(*eleClu1),&(*scRef2->seed()),eventData_->barrelRecHits,eventData_->endcapRecHits)>=threshold) ||
             (EgAmbiguityTools::sharedEnergy(&(*scRef1->seed()),&(*scRef2->seed()),eventData_->barrelRecHits,eventData_->endcapRecHits)>=threshold) ) ;
         }
        else
         { throw cms::Exception("GsfElectronAlgo|UnknownAmbiguityClustersOverlapStrategy")<<"value of generalData_->strategyCfg.ambClustersOverlapStrategy is : "<<generalData_->strategyCfg.ambClustersOverlapStrategy ; }

        // main instructions
        if (sameCluster)
         {
          LogDebug("GsfElectronAlgo")
            << "Discarding electron with E/P " << (*e2)->eSuperClusterOverP()
            << ", cluster " << scRef2.get()
            << " and track " << (*e2)->gsfTrack().get() ;
          (*e1)->addAmbiguousGsfTrack((*e2)->gsfTrack()) ;
          (*e2)->setAmbiguous(true) ;
         }
        else if ((*e1)->gsfTrack()==(*e2)->gsfTrack())
         {
          edm::LogWarning("GsfElectronAlgo")
            << "Forgetting electron with E/P " << (*e2)->eSuperClusterOverP()
            << ", cluster " << scRef2.get()
            << " and track " << (*e2)->gsfTrack().get() ;
          (*e2)->setAmbiguous(true) ;
         }
       }
     }
   }
 }

void GsfElectronAlgo::removeAmbiguousElectrons()
 {
  GsfElectronPtrCollection::size_type ei = 1, emax = eventData_->electrons->size() ;
  GsfElectronPtrCollection::iterator eitr = eventData_->electrons->begin() ;
  while (eitr!=eventData_->electrons->end())
   {
    LogTrace("GsfElectronAlgo")<<"========== remove ambiguous "<<ei<<"/"<<emax<<"==========" ;
    if ((*eitr)->ambiguous())
     { delete (*eitr) ; eitr = eventData_->electrons->erase(eitr) ; ++ei ; }
    else
     { ++eitr ; ++ei ; }
   }
 }


// Pixel match variables
void GsfElectronAlgo::setPixelMatchInfomation(reco::GsfElectron* ele){
  int sd1     = 0 ;
  int sd2     = 0 ;
  float dPhi1 = 0 ;
  float dPhi2 = 0 ;
  float dRz1  = 0 ;
  float dRz2  = 0 ;
  edm::RefToBase<TrajectorySeed> seed = ele->gsfTrack()->extra()->seedRef();
  ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>();
  if(seed.isNull()){}
  else{
    if(elseed.isNull()){}
    else{
      sd1     = elseed->subDet1() ;
      sd2     = elseed->subDet2() ;
      dPhi1 = (ele->charge()>0) ? elseed->dPhi1Pos() : elseed->dPhi1() ;
      dPhi2 = (ele->charge()>0) ? elseed->dPhi2Pos() : elseed->dPhi2() ;
      dRz1  = (ele->charge()>0) ? elseed->dRz1Pos () : elseed->dRz1 () ;
      dRz2  = (ele->charge()>0) ? elseed->dRz2Pos () : elseed->dRz2 () ;
    }
  }
  ele->setPixelMatchSubdetectors(sd1,sd2) ;
  ele->setPixelMatchDPhi1(dPhi1) ;
  ele->setPixelMatchDPhi2(dPhi2) ;
  ele->setPixelMatchDRz1 (dRz1 ) ;
  ele->setPixelMatchDRz2 (dRz2 ) ;
}
