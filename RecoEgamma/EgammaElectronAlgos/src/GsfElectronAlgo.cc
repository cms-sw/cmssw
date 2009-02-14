// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      GsfElectronAlgo
//
/**\class GsfElectronAlgo EgammaElectronAlgos/GsfElectronAlgo

 Description: top algorithm producing TrackCandidate and Electron objects from supercluster
              driven pixel seeded Ckf tracking

*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Thu july 6 13:22:06 CEST 2006
// $Id: GsfElectronAlgo.cc,v 1.41 2009/02/06 13:29:28 chamont Exp $
//
//

#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <Math/VectorUtil.h>
#include <Math/Point3D.h>
#include <sstream>
#include <algorithm>


using namespace edm ;
using namespace std ;
using namespace reco ;

GsfElectronAlgo::GsfElectronAlgo
 ( const edm::ParameterSet & conf,
   double maxEOverPBarrel, double maxEOverPEndcaps,
   double minEOverPBarrel, double minEOverPEndcaps,
   double maxDeltaEta, double maxDeltaPhi,
   double hOverEConeSize, double hOverEPtMin,
   double maxHOverEDepth1, double maxHOverEDepth2,
   bool applyEtaCorrection, bool applyAmbResolution )
 : maxEOverPBarrel_(maxEOverPBarrel), maxEOverPEndcaps_(maxEOverPEndcaps),
   minEOverPBarrel_(minEOverPBarrel), minEOverPEndcaps_(minEOverPEndcaps),
   maxDeltaEta_(maxDeltaEta), maxDeltaPhi_(maxDeltaPhi),
   hOverEConeSize_(hOverEConeSize), hOverEPtMin_(hOverEPtMin),
   maxHOverEDepth1_(maxHOverEDepth1), maxHOverEDepth2_(maxHOverEDepth2),
   applyEtaCorrection_(applyEtaCorrection), applyAmbResolution_(applyAmbResolution),
   cacheIDGeom_(0),cacheIDTopo_(0),cacheIDTDGeom_(0),cacheIDMagField_(0)
 {
  // this is the new version allowing to configurate the algo
  // interfaces still need improvement!!
  mtsTransform_ = 0 ;

  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;

  // get input collections
  hcalTowers_ = conf.getParameter<edm::InputTag>("hcalTowers");
  tracks_ = conf.getParameter<edm::InputTag>("tracks");
  ctfTracks_ = conf.getParameter<edm::InputTag>("ctfTracks");
  reducedBarrelRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedBarrelRecHitCollection") ;
  reducedEndcapRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedEndcapRecHitCollection") ;
}

GsfElectronAlgo::~GsfElectronAlgo() {
  delete mtsTransform_;
}

void GsfElectronAlgo::setupES(const edm::EventSetup& es) {

  // get EventSetupRecords if needed
  bool updateField(false);
  if (cacheIDMagField_!=es.get<IdealMagneticFieldRecord>().cacheIdentifier()){
    updateField = true;
    cacheIDMagField_=es.get<IdealMagneticFieldRecord>().cacheIdentifier();
    es.get<IdealMagneticFieldRecord>().get(theMagField);
  }

  bool updateGeometry(false);
  if (cacheIDTDGeom_!=es.get<TrackerDigiGeometryRecord>().cacheIdentifier()){
    updateGeometry = true;
    cacheIDTDGeom_=es.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    es.get<TrackerDigiGeometryRecord>().get(trackerHandle_);
  }

  if ( updateField || updateGeometry ) {
    delete mtsTransform_;
    mtsTransform_ = new MultiTrajectoryStateTransform(trackerHandle_.product(),theMagField.product());
  }

  if (cacheIDGeom_!=es.get<CaloGeometryRecord>().cacheIdentifier()){
    cacheIDGeom_=es.get<CaloGeometryRecord>().cacheIdentifier();
    es.get<CaloGeometryRecord>().get(theCaloGeom);
  }

  if (cacheIDTopo_!=es.get<CaloTopologyRecord>().cacheIdentifier()){
    cacheIDTopo_=es.get<CaloTopologyRecord>().cacheIdentifier();
    es.get<CaloTopologyRecord>().get(theCaloTopo);
  }


}

void  GsfElectronAlgo::run(Event& e, GsfElectronCollection & outEle) {

  // get the input
  edm::Handle<GsfTrackCollection> tracksH;
  e.getByLabel(tracks_,tracksH);
  edm::Handle<TrackCollection> ctfTracksH;
  e.getByLabel(ctfTracks_, ctfTracksH);
  edm::Handle< EcalRecHitCollection > pEBRecHits;
  e.getByLabel( reducedBarrelRecHitCollection_, pEBRecHits );
  edm::Handle< EcalRecHitCollection > pEERecHits;
  e.getByLabel( reducedEndcapRecHitCollection_, pEERecHits ) ;
  edm::Handle<CaloTowerCollection> towersH;
  e.getByLabel(hcalTowers_, towersH);
  //towers_ = towersHandle.product();

  // get the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByType(recoBeamSpotHandle);
  const math::XYZPoint bsPosition = recoBeamSpotHandle->position();

  // temporay array for electron before amb. solving
  GsfElectronPtrCollection tempEle;

  // create electrons
  process(tracksH,ctfTracksH,towersH,pEBRecHits,pEERecHits,bsPosition,tempEle);

  std::ostringstream str;

  str << "\n========== GsfElectronAlgo Info (before amb. solving) ==========";
  str << "\nEvent " << e.id();
  str << "\nNumber of final electron tracks: " << tracksH.product()->size();
  str << "\nNumber of final electrons: " << tempEle.size();
  for (GsfElectronPtrCollection::const_iterator it = tempEle.begin(); it != tempEle.end(); it++) {
    str << "\nNew electron with charge, pt, eta, phi : "  << (*it)->charge() << " , "
        << (*it)->pt() << " , " << (*it)->eta() << " , " << (*it)->phi();
  }

  str << "\n=================================================";
  LogDebug("GsfElectronAlgo") << str.str();

  if (applyAmbResolution_)
   {

    resolveElectrons(tempEle, outEle);

    std::ostringstream str2 ;

    str2 << "\n========== GsfElectronAlgo Info (after amb. solving) ==========";
    str2 << "\nEvent " << e.id();
    str2 << "\nNumber of final electron tracks: " << tracksH.product()->size();
    str2 << "\nNumber of final electrons: " << outEle.size();
    for ( GsfElectronCollection::const_iterator it = outEle.begin(); it != outEle.end(); it++) {
      str2 << "\nNew electron with charge, pt, eta, phi : "  << it->charge() << " , "
          << it->pt() << " , " << it->eta() << " , " << it->phi();
    }
    str2 << "\n=================================================";
    LogDebug("GsfElectronAlgo") << str2.str() ;

   }
  else
   {
    for ( GsfElectronPtrCollection::const_iterator it = tempEle.begin() ; it != tempEle.end() ; it++ )
     { outEle.push_back(**it) ; }
   }

  return;
}

void GsfElectronAlgo::process(
  edm::Handle<GsfTrackCollection> gsfTracksH,
  edm::Handle<TrackCollection> ctfTracksH,
  edm::Handle<CaloTowerCollection> towersH,
  edm::Handle<EcalRecHitCollection> reducedEBRecHits,
  edm::Handle<EcalRecHitCollection> reducedEERecHits,
  const math::XYZPoint & bsPosition,
  GsfElectronPtrCollection & outEle )
 {
  // HCAL iso deposits
  EgammaTowerIsolation towerIso1(hOverEConeSize_,0.,hOverEPtMin_,1,towersH.product()) ;
  EgammaTowerIsolation towerIso2(hOverEConeSize_,0.,hOverEPtMin_,2,towersH.product()) ;

  const GsfTrackCollection * gsfTrackCollection = gsfTracksH.product() ;
  for (unsigned int i=0;i<gsfTrackCollection->size();++i) {

    // track -scl association

    const GsfTrack & t=(*gsfTrackCollection)[i];
    const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH,i);
    const SuperClusterRef & scRef=getTrSuperCluster(gsfTrackRef);
    const SuperCluster theClus=*scRef;
    const BasicClusterRef & elbcRef=getEleBasicCluster(gsfTrackRef,scRef);
    //std::vector<DetId> vecId=theClus.seed()->getHitsByDetId();
    //subdet_ =vecId[0].subdetId();
    subdet_ =theClus.seed()->hitsAndFractions()[0].first.subdetId();

    // calculate Trajectory StatesOnSurface....
    if (!calculateTSOS(t,theClus, bsPosition)) continue;
    mtsMode_->momentumFromModeCartesian(vtxTSOS_,vtxMom_);
    sclPos_=sclTSOS_.globalPosition();

    // hadronic energy
    double HoE1=towerIso1.getTowerESum(&theClus)/theClus.energy();
    double HoE2=towerIso2.getTowerESum(&theClus)/theClus.energy();

    if (preSelection(theClus, HoE1, HoE2))
     {
      pair<TrackRef,float> ctfpair = getCtfTrackRef(gsfTrackRef,ctfTracksH) ;
      const TrackRef ctfTrackRef = ctfpair.first ;
      const float fracShHits = ctfpair.second ;
      // interface to be improved...
      createElectron(scRef,elbcRef,gsfTrackRef,ctfTrackRef,fracShHits,HoE1,HoE2,reducedEBRecHits,reducedEERecHits,outEle) ;
      LogInfo("")<<"Constructed new electron with energy  "<< scRef->energy();
    }
  } // loop over tracks
}

bool GsfElectronAlgo::preSelection(const SuperCluster& clus, double HoE1, double HoE2)
{

  LogDebug("")<< "========== preSelection ==========";

  // E/p cut
  LogDebug("") << "E/p : " << clus.energy()/vtxMom_.mag();
  if ((subdet_==EcalBarrel) && (clus.energy()/vtxMom_.mag() > maxEOverPBarrel_)) return false;
  if ((subdet_==EcalEndcap) && (clus.energy()/vtxMom_.mag() > maxEOverPEndcaps_)) return false;
  if ((subdet_==EcalBarrel) && (clus.energy()/vtxMom_.mag() < minEOverPBarrel_)) return false;
  if ((subdet_==EcalEndcap) && (clus.energy()/vtxMom_.mag() < minEOverPEndcaps_)) return false;
  LogDebug("") << "E/p criteria is satisfied ";

  // HoE cuts
  LogDebug("") << "HoE1 : " << HoE1 << "HoE2 : " << HoE2;
  if ( HoE1 > maxHOverEDepth1_ || HoE2 > maxHOverEDepth2_ ) return false;
  LogDebug("") << "H/E criteria is satisfied ";

  // delta eta criteria
  double etaclu = clus.eta();
  double etatrk = sclPos_.eta();
  double deta = etaclu-etatrk;
  LogDebug("") << "delta eta : " << deta;
  if (fabs(deta) > maxDeltaEta_) return false;
  LogDebug("") << "Delta eta criteria is satisfied ";

  // delta phi criteria
  double phiclu = clus.phi();
  double phitrk = sclPos_.phi();
  double dphi = phiclu-phitrk;
  if (fabs(dphi)>CLHEP::pi)
    dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
  LogDebug("") << "delta phi : " << dphi;
  if (fabs(dphi) > maxDeltaPhi_) return false;
  LogDebug("") << "Delta phi criteria is satisfied ";

  LogDebug("") << "electron has passed preselection criteria ";
  LogDebug("") << "=================================================";
  return true;

}

// interface to be improved...
void GsfElectronAlgo::createElectron
 ( const SuperClusterRef & scRef,
   const BasicClusterRef & elbcRef,
   const GsfTrackRef & trackRef,
   const TrackRef & ctfTrackRef, const float shFracInnerHits,
   double HoE1, double HoE2,
   edm::Handle<EcalRecHitCollection> reducedEBRecHits,
   edm::Handle<EcalRecHitCollection> reducedEERecHits,
   GsfElectronPtrCollection & outEle )

 {
      GlobalVector innMom;
      mtsMode_->momentumFromModeCartesian(innTSOS_,innMom);
      GlobalPoint innPos=innTSOS_.globalPosition();
      GlobalVector seedMom;
      mtsMode_->momentumFromModeCartesian(seedTSOS_,seedMom);
      GlobalPoint  seedPos=seedTSOS_.globalPosition();
      GlobalVector eleMom;
      mtsMode_->momentumFromModeCartesian(eleTSOS_,eleMom);
      GlobalPoint  elePos=eleTSOS_.globalPosition();
      GlobalVector sclMom;
      mtsMode_->momentumFromModeCartesian(sclTSOS_,sclMom);

      GlobalPoint  vtxPos=vtxTSOS_.globalPosition();
      GlobalVector outMom;
      mtsMode_->momentumFromModeCartesian(outTSOS_,outMom);
      GlobalPoint  outPos=outTSOS_.globalPosition();

      // cluster shape
      const CaloTopology * topology = theCaloTopo.product() ;
      const CaloGeometry * geometry = theCaloGeom.product() ;
      const reco::BasicCluster & seedCluster = *(scRef->seed()) ;
      const EcalRecHitCollection * reducedRecHits = 0 ;
      //std::vector<DetId> vecId=seedCluster.getHitsByDetId() ;
      //int detector = vecId[0].subdetId() ;
      DetId seedXtalId = seedCluster.hitsAndFractions()[0].first;
      int detector = seedXtalId.subdetId() ;
      if (detector==EcalBarrel)
       { reducedRecHits = reducedEBRecHits.product() ; }
      else if (detector==EcalEndcap)
       { reducedRecHits = reducedEERecHits.product() ; }
      else
       { edm::LogWarning("")<<"GsfElectronAlgo::createElectron(): do not know if it is a barrel or endcap seed cluster !!!!" ; }
      std::vector<float> covariances = EcalClusterTools::covariances(seedCluster,reducedRecHits,topology,geometry) ;
      std::vector<float> localCovariances = EcalClusterTools::localCovariances(seedCluster,reducedRecHits,topology) ;
      float scSigmaEtaEta = sqrt(covariances[0]) ;
      float scSigmaIEtaIEta = sqrt(localCovariances[0]) ;
      float scE1x5 = EcalClusterTools::e1x5(seedCluster,reducedRecHits,topology)  ;
      float scE2x5 = EcalClusterTools::e2x5Max(seedCluster,reducedRecHits,topology)  ;
      float scE5x5 = EcalClusterTools::e5x5(seedCluster,reducedRecHits,topology) ;

      // momentum
      double scale = (*scRef).energy()/vtxMom_.mag();
      math::XYZTLorentzVectorD momentum= math::XYZTLorentzVector(vtxMom_.x()*scale,
								 vtxMom_.y()*scale,
								 vtxMom_.z()*scale,
								 (*scRef).energy());

      // now create electron
      GsfElectron * ele = new
       GsfElectron(momentum,scRef,trackRef,sclPos_,sclMom,seedPos,seedMom,innPos,innMom,vtxPos,vtxMom_,outPos,outMom,HoE1,
       HoE2,scSigmaEtaEta,scSigmaIEtaIEta,scE1x5,scE2x5,scE5x5,ctfTrackRef,shFracInnerHits,elbcRef,elePos,eleMom) ;

      // set fiducial region
      if (detector==EcalBarrel) 
	ele->setIsEB(true);
      else if (detector==EcalEndcap) 
	ele->setIsEE(true);
      else
        std::cout << "Electron not in EcalBarrel nor EcalEndcap!! " << std::endl;    
      double feta=fabs(scRef->position().eta());
      // using nextToBoundary definition of gaps
      if (fabs(feta-1.479)<.1 && ele->isEB() && EBDetId::isNextToEtaBoundary(EBDetId(seedXtalId))) ele->setIsEBEEGap(true); 
      if (fabs(feta-1.479)<.1 && ele->isEE() && EEDetId::isNextToRingBoundary(EEDetId(seedXtalId))) ele->setIsEBEEGap(true); 
      if (ele->isEB() && EBDetId::isNextToEtaBoundary(EBDetId(seedXtalId)) && !ele->isEBEEGap()) ele->setIsEBEtaGap(true); 
      if (ele->isEB() && EBDetId::isNextToPhiBoundary(EBDetId(seedXtalId))) ele->setIsEBPhiGap(true); 
      if (ele->isEE() && EEDetId::isNextToRingBoundary(EEDetId(seedXtalId)) && !ele->isEBEEGap()) ele->setIsEERingGap(true); 
      if (ele->isEE() && EEDetId::isNextToDBoundary(EEDetId(seedXtalId))) ele->setIsEEDeeGap(true); 
      
      // set corrections + classification
      ElectronClassification theClassifier;
      theClassifier.correct(*ele);
      ElectronEnergyCorrector theEnCorrector;
      theEnCorrector.correct(*ele, applyEtaCorrection_);
      ElectronMomentumCorrector theMomCorrector;
      theMomCorrector.correct(*ele,vtxTSOS_);

      outEle.push_back(ele);
}

const SuperClusterRef GsfElectronAlgo::getTrSuperCluster(const GsfTrackRef & trackRef) {
    edm::RefToBase<TrajectorySeed> seed = trackRef->extra()->seedRef();
    ElectronSeedRef elseed=seed.castTo<ElectronSeedRef>();
    edm::RefToBase<CaloCluster> caloCluster = elseed->caloCluster() ;
    return caloCluster.castTo<SuperClusterRef>() ;
}

const BasicClusterRef GsfElectronAlgo::getEleBasicCluster(const GsfTrackRef &t, const SuperClusterRef & scRef) {

    BasicClusterRef eleRef;
    TrajectoryStateOnSurface tempTSOS;
    TrajectoryStateOnSurface outTSOS = mtsTransform_->outerStateOnSurface(*t);
    float dphimin = 1.e30;
    for (basicCluster_iterator bc=scRef->clustersBegin(); bc!=scRef->clustersEnd(); bc++) {
      GlobalPoint posclu((*bc)->position().x(),(*bc)->position().y(),(*bc)->position().z());
      tempTSOS = mtsTransform_->extrapolatedState(outTSOS,posclu) ;
      if (!tempTSOS.isValid()) tempTSOS=outTSOS;
      GlobalPoint extrap = tempTSOS.globalPosition();
      float dphi = posclu.phi() - extrap.phi();
      if (fabs(dphi)>CLHEP::pi) dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
      if (fabs(dphi)<dphimin) {
        dphimin = fabs(dphi);
	eleRef = (*bc);
	eleTSOS_ = tempTSOS;
      }
    }
    return eleRef;

}

bool  GsfElectronAlgo::calculateTSOS(const GsfTrack &t,const SuperCluster & theClus, const math::XYZPoint &
bsPosition){

    //at innermost point
    innTSOS_ = mtsTransform_->innerStateOnSurface(t);
    if (!innTSOS_.isValid()) return false;

    //at vertex
    // innermost state propagation to the beam spot position
    vtxTSOS_ = mtsTransform_->extrapolatedState(innTSOS_,
						GlobalPoint(bsPosition.x(),bsPosition.y(),bsPosition.z()));
    if (!vtxTSOS_.isValid()) vtxTSOS_=innTSOS_;

    //at seed
    outTSOS_ = mtsTransform_->outerStateOnSurface(t);
    if (!outTSOS_.isValid()) return false;

    //    TrajectoryStateOnSurface seedTSOS
    seedTSOS_ = mtsTransform_->extrapolatedState(outTSOS_,
						 GlobalPoint(theClus.seed()->position().x(),
							     theClus.seed()->position().y(),
							     theClus.seed()->position().z()));
    if (!seedTSOS_.isValid()) seedTSOS_=outTSOS_;

    //at scl
    sclTSOS_ = mtsTransform_->extrapolatedState(innTSOS_,GlobalPoint(theClus.x(),theClus.y(),theClus.z()));
    if (!sclTSOS_.isValid()) sclTSOS_=outTSOS_;
    return true;
}


//=======================================================================================
// Ambiguity solving
//=======================================================================================

bool better_electron( const reco::GsfElectron * e1, const reco::GsfElectron * e2 )
 { return (fabs(e1->eSuperClusterOverP()-1)<fabs(e2->eSuperClusterOverP()-1)) ; }

void GsfElectronAlgo::resolveElectrons( GsfElectronPtrCollection & inEle, reco::GsfElectronCollection & outEle )
 {
  GsfElectronPtrCollection::iterator e1, e2 ;
  inEle.sort(better_electron) ;
  for( e1 = inEle.begin() ;  e1 != inEle.end() ; ++e1 )
   {
    LogDebug("GsfElectronAlgo")
      << "Blessing electron with E/P " << (*e1)->eSuperClusterOverP()
      << ", cluster " << (*e1)->superCluster().get()
      << " & track " << (*e1)->gsfTrack().get() ;
    for( e2 = e1, ++e2 ;  e2 != inEle.end() ; )
     {
      if ((*e1)->superCluster()==(*e2)->superCluster())
       {
        LogDebug("GsfElectronAlgo")
          << "Discarding electron with E/P " << (*e2)->eSuperClusterOverP()
          << ", cluster " << (*e2)->superCluster().get()
          << " and track " << (*e2)->gsfTrack().get() ;
        (*e1)->addAmbiguousGsfTrack((*e2)->gsfTrack()) ;
        e2 = inEle.erase(e2) ;
       }
      else if ((*e1)->gsfTrack()==(*e2)->gsfTrack())
       {
        LogDebug("GsfElectronAlgo")
          << "Forgetting electron with E/P " << (*e2)->eSuperClusterOverP()
          << ", cluster " << (*e2)->superCluster().get()
          << " and track " << (*e2)->gsfTrack().get() ;
        e2 = inEle.erase(e2) ;
       }
      else
       { ++e2 ; }
     }
    outEle.push_back(**e1) ;
   }
 }


//=======================================================================================
// Code from Puneeth Kalavase
//=======================================================================================

pair<TrackRef,float> GsfElectronAlgo::getCtfTrackRef(const GsfTrackRef& gsfTrackRef, edm::Handle<reco::TrackCollection> ctfTracksH ) {

  float maxFracShared = 0;
  TrackRef ctfTrackRef = TrackRef() ;
  const TrackCollection * ctfTrackCollection = ctfTracksH.product() ;

  //get the Hit Pattern for the gsfTrack
  const HitPattern& gsfHitPattern = gsfTrackRef->hitPattern();

  unsigned int counter ;
  TrackCollection::const_iterator ctfTkIter ;
  for ( ctfTkIter = ctfTrackCollection->begin() , counter = 0 ;
        ctfTkIter != ctfTrackCollection->end() ; ctfTkIter++, counter++ ) {

    double dEta = gsfTrackRef->eta() - ctfTkIter->eta();
    double dPhi = gsfTrackRef->phi() - ctfTkIter->phi();
    double pi = acos(-1.);
    if(fabs(dPhi) > pi) dPhi = 2*pi - fabs(dPhi);

    //dont want to look at every single track in the event!
    if(sqrt(dEta*dEta + dPhi*dPhi) > 0.3) continue;

    unsigned int shared = 0;
    int gsfHitCounter = 0;
    int numGsfInnerHits = 0;
    int numCtfInnerHits = 0;
    //get the CTF Track Hit Pattern
    const HitPattern& ctfHitPattern = ctfTkIter->hitPattern();

    for(trackingRecHit_iterator elHitsIt = gsfTrackRef->recHitsBegin();
        elHitsIt != gsfTrackRef->recHitsEnd(); elHitsIt++, gsfHitCounter++) {
      if(!((**elHitsIt).isValid()))  //count only valid Hits
	continue;

      //look only in the pixels/TIB/TID
      uint32_t gsfHit = gsfHitPattern.getHitPattern(gsfHitCounter);
      if(!(gsfHitPattern.pixelHitFilter(gsfHit) ||
	   gsfHitPattern.stripTIBHitFilter(gsfHit) ||
	   gsfHitPattern.stripTIDHitFilter(gsfHit) ) ) continue;
      numGsfInnerHits++;

      int ctfHitsCounter = 0;
      numCtfInnerHits = 0;
      for(trackingRecHit_iterator ctfHitsIt = ctfTkIter->recHitsBegin();
          ctfHitsIt != ctfTkIter->recHitsEnd(); ctfHitsIt++, ctfHitsCounter++) {
        if(!((**ctfHitsIt).isValid())) //count only valid Hits!
	  continue;

	uint32_t ctfHit = ctfHitPattern.getHitPattern(ctfHitsCounter);
	if( !(ctfHitPattern.pixelHitFilter(ctfHit) ||
	      ctfHitPattern.stripTIBHitFilter(ctfHit) ||
	      ctfHitPattern.stripTIDHitFilter(ctfHit) ) ) continue;
	numCtfInnerHits++;
        if( (**elHitsIt).sharesInput(&(**ctfHitsIt), TrackingRecHit::all) ) {
          shared++;
          break;
        }
      }//ctfHits iterator

    }//gsfHits iterator

    if ( static_cast<float>(shared)/min(numGsfInnerHits,numCtfInnerHits) > maxFracShared ) {
      maxFracShared = static_cast<float>(shared)/min(numGsfInnerHits, numCtfInnerHits);
      ctfTrackRef = TrackRef(ctfTracksH,counter);
    }

  }//ctfTrack iterator

  return make_pair(ctfTrackRef,maxFracShared);
}
