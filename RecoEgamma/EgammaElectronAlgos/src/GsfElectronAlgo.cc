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
// $Id: GsfElectronAlgo.cc,v 1.45 2009/03/20 22:59:18 chamont Exp $
//
//

#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"

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
   double minSCEtBarrel, double minSCEtEndcaps,
   double maxEOverPBarrel, double maxEOverPEndcaps,
   double minEOverPBarrel, double minEOverPEndcaps,
   double maxDeltaEtaBarrel, double maxDeltaEtaEndcaps,
   double maxDeltaPhiBarrel,double maxDeltaPhiEndcaps,
   double hOverEConeSize, double hOverEPtMin,
   double maxHOverEDepth1Barrel, double maxHOverEDepth1Endcaps,
   double maxHOverEDepth2,
   double maxSigmaIetaIetaBarrel, double maxSigmaIetaIetaEndcaps,
   double maxFbremBarrel, double maxFbremEndcaps,
   bool isBarrel, bool isEndcaps, bool isFiducial,
   bool seedFromTEC,
   bool applyPreselection, bool applyEtaCorrection, bool applyAmbResolution )
 : minSCEtBarrel_(minSCEtBarrel), minSCEtEndcaps_(minSCEtEndcaps), maxEOverPBarrel_(maxEOverPBarrel), maxEOverPEndcaps_(maxEOverPEndcaps),
   minEOverPBarrel_(minEOverPBarrel), minEOverPEndcaps_(minEOverPEndcaps),
   maxDeltaEtaBarrel_(maxDeltaEtaBarrel), maxDeltaEtaEndcaps_(maxDeltaEtaEndcaps),
   maxDeltaPhiBarrel_(maxDeltaPhiBarrel),maxDeltaPhiEndcaps_(maxDeltaPhiEndcaps),
   hOverEConeSize_(hOverEConeSize), hOverEPtMin_(hOverEPtMin),
   maxHOverEDepth1Barrel_(maxHOverEDepth1Barrel), maxHOverEDepth1Endcaps_(maxHOverEDepth1Endcaps),
   maxHOverEDepth2_(maxHOverEDepth2),
   maxSigmaIetaIetaBarrel_(maxSigmaIetaIetaBarrel), maxSigmaIetaIetaEndcaps_(maxSigmaIetaIetaEndcaps),
   maxFbremBarrel_(maxFbremBarrel), maxFbremEndcaps_(maxFbremEndcaps),
   isBarrel_(isBarrel), isEndcaps_(isEndcaps), isFiducial_(isFiducial),
   seedFromTEC_(seedFromTEC),
   applyPreselection_(applyPreselection), applyEtaCorrection_(applyEtaCorrection), applyAmbResolution_(applyAmbResolution),
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
  gsfElectronCores_ = conf.getParameter<edm::InputTag>("gsfElectronCores");
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
  edm::Handle<GsfElectronCoreCollection> coresH;
  e.getByLabel(gsfElectronCores_,coresH);
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

  // temporay array for electrons before preselection and before amb. solving
  GsfElectronPtrCollection tempEle, tempEle1;

  // create electrons
  process(tracksH,coresH,ctfTracksH,towersH,pEBRecHits,pEERecHits,bsPosition,tempEle);

  std::ostringstream str;

  str << "\n========== GsfElectronAlgo Info (before preselection) ==========";
  str << "\nEvent " << e.id();
  str << "\nNumber of electron tracks: " << tracksH.product()->size();
  str << "\nNumber of electrons: " << tempEle.size();
  for (GsfElectronPtrCollection::const_iterator it = tempEle.begin(); it != tempEle.end(); it++) {
    str << "\nNew electron with charge, pt, eta, phi : "  << (*it)->charge() << " , "
        << (*it)->pt() << " , " << (*it)->eta() << " , " << (*it)->phi();
  }

  if (applyPreselection_)
   {

    preselectElectrons(tempEle, tempEle1);

    std::ostringstream str1 ;

    str1 << "\n=================================================";
    LogDebug("GsfElectronAlgo") << str.str();

    str1 << "\n========== GsfElectronAlgo Info (before amb. solving) ==========";
    str1 << "\nEvent " << e.id();
    str1 << "\nNumber of preselected electrons: " << tempEle1.size();
    for (GsfElectronPtrCollection::const_iterator it = tempEle1.begin(); it != tempEle1.end(); it++) {
      str1 << "\nNew electron with charge, pt, eta, phi : "  << (*it)->charge() << " , "
          << (*it)->pt() << " , " << (*it)->eta() << " , " << (*it)->phi();
    }
   } 
  else
   {
    for ( GsfElectronPtrCollection::const_iterator it = tempEle.begin() ; it != tempEle.end() ; it++ )
     { tempEle1.push_back(*it) ; }
   }
  
  str << "\n=================================================";
  LogDebug("GsfElectronAlgo") << str.str();

  if (applyAmbResolution_)
   {
    resolveElectrons(tempEle1, outEle);

    std::ostringstream str2 ;

    str2 << "\n========== GsfElectronAlgo Info (after amb. solving) ==========";
    str2 << "\nEvent " << e.id();
    str2 << "\nNumber of preselected and resolved electrons: " << outEle.size();
    for ( GsfElectronCollection::const_iterator it = outEle.begin(); it != outEle.end(); it++) {
      str2 << "\nNew electron with charge, pt, eta, phi : "  << it->charge() << " , "
          << it->pt() << " , " << it->eta() << " , " << it->phi();
    }
    str2 << "\n=================================================";
    LogDebug("GsfElectronAlgo") << str2.str() ;

   }
  else
   {
    for ( GsfElectronPtrCollection::const_iterator it = tempEle1.begin() ; it != tempEle1.end() ; it++ )
     { outEle.push_back(**it) ; }
   }

  return;
}

void GsfElectronAlgo::process(
  edm::Handle<GsfTrackCollection> gsfTracksH,
  edm::Handle<GsfElectronCoreCollection> coresH,
  edm::Handle<TrackCollection> ctfTracksH,
  edm::Handle<CaloTowerCollection> towersH,
  edm::Handle<EcalRecHitCollection> reducedEBRecHits,
  edm::Handle<EcalRecHitCollection> reducedEERecHits,
  const math::XYZPoint & bsPosition,
  GsfElectronPtrCollection & outEle )
 {

  // Isolation algos

  float extRadiusSmall=0.3, extRadiusLarge=0.4, intRadius=0.015, ptMin=1.0, maxVtxDist=0.2, drb=0.1; 
  ElectronTkIsolation tkIsolation03(extRadiusSmall,intRadius,ptMin,maxVtxDist,drb,ctfTracksH.product(),bsPosition) ;
  ElectronTkIsolation tkIsolation04(extRadiusLarge,intRadius,ptMin,maxVtxDist,drb,ctfTracksH.product(),bsPosition) ;
  
  float egHcalIsoConeSizeOutSmall=0.3, egHcalIsoConeSizeOutLarge=0.4, egHcalIsoConeSizeIn=0.15,egHcalIsoPtMin=0.0;
  int egHcalDepth1=1, egHcalDepth2=2;  
  EgammaTowerIsolation hadDepth1Isolation03(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,towersH.product()) ;
  EgammaTowerIsolation hadDepth2Isolation03(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,towersH.product()) ;
  EgammaTowerIsolation hadDepth1Isolation04(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,towersH.product()) ;
  EgammaTowerIsolation hadDepth2Isolation04(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,towersH.product()) ;
  
  float egIsoConeSizeOutSmall=0.3, egIsoConeSizeOutLarge=0.4, egIsoJurassicWidth=0.02;
  float egIsoPtMinBarrel=-9999.,egIsoEMinBarrel=0.08, egIsoConeSizeInBarrel=0.045;
  float egIsoPtMinEndcap=-9999.,egIsoEMinEndcap=0.3, egIsoConeSizeInEndcap=0.07;
  EcalRecHitMetaCollection ecalBarrelHits(*reducedEBRecHits);
  EcalRecHitMetaCollection ecalEndcapHits(*reducedEERecHits);
  EgammaRecHitIsolation ecalBarrelIsol03(egIsoConeSizeOutSmall,egIsoConeSizeInBarrel,egIsoJurassicWidth,egIsoPtMinBarrel,egIsoEMinBarrel,theCaloGeom,&ecalBarrelHits,DetId::Ecal);
  EgammaRecHitIsolation ecalBarrelIsol04(egIsoConeSizeOutLarge,egIsoConeSizeInBarrel,egIsoJurassicWidth,egIsoPtMinBarrel,egIsoEMinBarrel,theCaloGeom,&ecalBarrelHits,DetId::Ecal);
  EgammaRecHitIsolation ecalEndcapIsol03(egIsoConeSizeOutSmall,egIsoConeSizeInEndcap,egIsoJurassicWidth,egIsoPtMinEndcap,egIsoEMinEndcap,theCaloGeom,&ecalEndcapHits,DetId::Ecal);
  EgammaRecHitIsolation ecalEndcapIsol04(egIsoConeSizeOutLarge,egIsoConeSizeInEndcap,egIsoJurassicWidth,egIsoPtMinEndcap,egIsoEMinEndcap,theCaloGeom,&ecalEndcapHits,DetId::Ecal);
  
  // HCAL isolation algo for H/E
  EgammaTowerIsolation towerIso1(hOverEConeSize_,0.,hOverEPtMin_,1,towersH.product()) ;
  EgammaTowerIsolation towerIso2(hOverEConeSize_,0.,hOverEPtMin_,2,towersH.product()) ;

  //const GsfTrackCollection * gsfTrackCollection = gsfTracksH.product() ;
  const GsfElectronCoreCollection * coreCollection = coresH.product() ;
  for (unsigned int i=0;i<coreCollection->size();++i) {

    // track -scl association

    //const GsfTrack & t=(*gsfTrackCollection)[i];
    const GsfElectronCoreRef coreRef = edm::Ref<GsfElectronCoreCollection>(coresH,i);
    const GsfTrackRef gsfTrackRef = coreRef->gsfTrack() ; //edm::Ref<GsfTrackCollection>(gsfTracksH,i);

    // Get the super cluster. If none, ignore the current track.
    SuperClusterRef scRef = coreRef->superCluster() ;
    if (scRef.isNull()) continue ;
    const SuperCluster theClus = *scRef ;

    BasicClusterRef elbcRef = getEleBasicCluster(gsfTrackRef,scRef) ;
    //std::vector<DetId> vecId=theClus.seed()->getHitsByDetId();
    //subdet_ =vecId[0].subdetId();
    subdet_ = theClus.seed()->hitsAndFractions()[0].first.subdetId();

    // calculate Trajectory StatesOnSurface....
    if (!calculateTSOS(*gsfTrackRef,theClus, bsPosition)) continue ;
    mtsMode_->momentumFromModeCartesian(vtxTSOS_,vtxMom_) ;
    sclPos_=sclTSOS_.globalPosition() ;
    
    // H/E
    double HoE1=towerIso1.getTowerESum(&theClus)/theClus.energy();
    double HoE2=towerIso2.getTowerESum(&theClus)/theClus.energy();

    pair<TrackRef,float> ctfpair = getCtfTrackRef(gsfTrackRef,ctfTracksH) ;
    const TrackRef ctfTrackRef = ctfpair.first ;
    const float fracShHits = ctfpair.second ;

    if (subdet_==EcalBarrel)
     createElectron(coreRef,elbcRef,ctfTrackRef,fracShHits,HoE1,HoE2,tkIsolation03,tkIsolation04,
      hadDepth1Isolation03,hadDepth2Isolation03,hadDepth1Isolation04,hadDepth2Isolation04,
      ecalBarrelIsol03,ecalBarrelIsol04,reducedEBRecHits,outEle) ;
    else if (subdet_==EcalEndcap)
     createElectron(coreRef,elbcRef,ctfTrackRef,fracShHits,HoE1,HoE2,tkIsolation03,tkIsolation04,
      hadDepth1Isolation03,hadDepth2Isolation03,hadDepth1Isolation04,hadDepth2Isolation04,
      ecalEndcapIsol03,ecalEndcapIsol04,reducedEERecHits,outEle) ;      

    LogInfo("")<<"Constructed new electron with energy  "<< scRef->energy();

  } // loop over tracks
}

void GsfElectronAlgo::preselectElectrons( GsfElectronPtrCollection & inEle, GsfElectronPtrCollection & outEle )
 {
  GsfElectronPtrCollection::iterator e1;
  for( e1 = inEle.begin() ;  e1 != inEle.end() ; ++e1 )
   {

    LogDebug("")<< "========== preSelection ==========";
 
    // Et cut
    LogDebug("") << "Et : " << (*e1)->superCluster()->energy()/cosh((*e1)->superCluster()->eta());
    if ((*e1)->isEB() && ((*e1)->superCluster()->energy()/cosh((*e1)->superCluster()->eta()) < minSCEtBarrel_)) continue;
    if ((*e1)->isEE() && ((*e1)->superCluster()->energy()/cosh((*e1)->superCluster()->eta()) < minSCEtEndcaps_)) continue;

    // E/p cut
    LogDebug("") << "E/p : " << (*e1)->eSuperClusterOverP();
    if ((*e1)->isEB() && ((*e1)->eSuperClusterOverP() > maxEOverPBarrel_)) continue;
    if ((*e1)->isEE() && ((*e1)->eSuperClusterOverP() > maxEOverPEndcaps_)) continue;
    if ((*e1)->isEB() && ((*e1)->eSuperClusterOverP() < minEOverPBarrel_)) continue;
    if ((*e1)->isEE() && ((*e1)->eSuperClusterOverP() < minEOverPEndcaps_)) continue;
    LogDebug("") << "E/p criteria is satisfied ";

    // HoE cuts
    LogDebug("") << "HoE1 : " << (*e1)->hcalDepth1OverEcal() << "HoE2 : " << (*e1)->hcalDepth2OverEcal();
    if ( (*e1)->isEB() && ((*e1)->hcalDepth1OverEcal() > maxHOverEDepth1Barrel_) ) continue;
    if ( (*e1)->isEE() && ((*e1)->hcalDepth1OverEcal() > maxHOverEDepth1Endcaps_) ) continue;
    if ( (*e1)->hcalDepth2OverEcal() > maxHOverEDepth2_ ) continue;
    LogDebug("") << "H/E criteria is satisfied ";

    // delta eta criteria
    double deta = (*e1)->deltaEtaSuperClusterTrackAtVtx();
    LogDebug("") << "delta eta : " << deta;
    if ((*e1)->isEB() && (fabs(deta) > maxDeltaEtaBarrel_)) continue;
    if ((*e1)->isEE() && (fabs(deta) > maxDeltaEtaEndcaps_)) continue;
    LogDebug("") << "Delta eta criteria is satisfied ";

    // delta phi criteria
    double dphi = (*e1)->deltaPhiSuperClusterTrackAtVtx();
    LogDebug("") << "delta phi : " << dphi;
    if ((*e1)->isEB() && (fabs(dphi) > maxDeltaPhiBarrel_)) continue;
    if ((*e1)->isEE() && (fabs(dphi) > maxDeltaPhiEndcaps_)) continue;
    LogDebug("") << "Delta phi criteria is satisfied ";

    if ((*e1)->isEB() && ((*e1)->sigmaIetaIeta() > maxSigmaIetaIetaBarrel_)) continue;
    if ((*e1)->isEE() && ((*e1)->sigmaIetaIeta() > maxSigmaIetaIetaEndcaps_)) continue;

    // fiducial
    if (!(*e1)->isEB() && isBarrel_) continue;
    if (!(*e1)->isEE() && isEndcaps_) continue;
    if (((*e1)->isEBEEGap() || (*e1)->isEBEtaGap() || (*e1)->isEBPhiGap() || (*e1)->isEERingGap() || (*e1)->isEEDeeGap()) 
     && isFiducial_) continue;

    // seed in TEC
    if (!seedFromTEC_) { 
      edm::RefToBase<TrajectorySeed> seed = (*e1)->gsfTrack()->extra()->seedRef() ;
      ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>() ;
      if (elseed.isNull())
	 { edm::LogError("GsfElectronAlgo")<<"The GsfTrack seed is not an ElectronSeed ?!" ; }
	else
	 {  
	  if (elseed->subDet2()==6) continue;
	 } 
    }

    LogDebug("") << "electron has passed preselection criteria ";
    LogDebug("") << "=================================================";

    outEle.push_back(*e1) ;
   
   }
 }
 
// utilities for constructor
float normalized_dphi( float dphi )
 {
  if (fabs(dphi)>pi) return (dphi<0?pi2+dphi:dphi-pi2) ;
  else return dphi ;
 }

math::XYZPoint convert( const GlobalPoint & gp )
 { return math::XYZPoint(gp.x(),gp.y(),gp.z()) ; }

math::XYZVector convert( const GlobalVector & gv )
 { return math::XYZVector(gv.x(),gv.y(),gv.z()) ; }

// interface to be improved...
void GsfElectronAlgo::createElectron
 ( const GsfElectronCoreRef & coreRef,
   const BasicClusterRef & elbcRef,
   const TrackRef & ctfTrackRef, const float shFracInnerHits,
   double HoE1, double HoE2,
   ElectronTkIsolation & tkIso03, ElectronTkIsolation & tkIso04,
   EgammaTowerIsolation & had1Iso03, EgammaTowerIsolation & had2Iso03, 
   EgammaTowerIsolation & had1Iso04, EgammaTowerIsolation & had2Iso04, 
   EgammaRecHitIsolation & ecalIso03,EgammaRecHitIsolation & ecalIso04,
   edm::Handle<EcalRecHitCollection> reducedRecHits,
   GsfElectronPtrCollection & outEle )

 {
  SuperClusterRef scRef = coreRef->superCluster() ;
  GsfTrackRef trackRef = coreRef->gsfTrack() ;

  // Seed info
  const reco::BasicCluster & seedCluster = *(scRef->seed()) ;
  DetId seedXtalId = seedCluster.hitsAndFractions()[0].first ;

  // various useful positions and momemtums
  GlobalVector innMom, seedMom, eleMom, sclMom, outMom ;
  mtsMode_->momentumFromModeCartesian(innTSOS_,innMom) ;
  GlobalPoint innPos=innTSOS_.globalPosition() ;
  mtsMode_->momentumFromModeCartesian(seedTSOS_,seedMom) ;
  GlobalPoint  seedPos=seedTSOS_.globalPosition() ;
  mtsMode_->momentumFromModeCartesian(eleTSOS_,eleMom) ;
  GlobalPoint  elePos=eleTSOS_.globalPosition() ;
  mtsMode_->momentumFromModeCartesian(sclTSOS_,sclMom) ;
  // sclPos_ already here
  // vtxMom_ already here
  GlobalPoint vtxPos=vtxTSOS_.globalPosition() ;
  mtsMode_->momentumFromModeCartesian(outTSOS_,outMom);
  GlobalPoint outPos=outTSOS_.globalPosition() ;


  //====================================================
  // Candidate attributes
  //====================================================

  double scale = (*scRef).energy()/vtxMom_.mag() ;
  Candidate::LorentzVector momentum = Candidate::LorentzVector
   ( vtxMom_.x()*scale,vtxMom_.y()*scale,vtxMom_.z()*scale,
	 (*scRef).energy() ) ;


  //====================================================
  // Track-Cluster Matching
  //====================================================

  reco::GsfElectron::TrackClusterMatching tcMatching ;
  tcMatching.electronCluster = elbcRef ;
  tcMatching.eSuperClusterOverP = (vtxMom_.mag()>0)?(scRef->energy()/vtxMom_.mag()):(-1.) ;
  tcMatching.eSeedClusterOverP = (vtxMom_.mag()>0.)?(seedCluster.energy()/vtxMom_.mag()):(-1) ;
  tcMatching.eSeedClusterOverPout = (seedMom.mag()>0.)?(seedCluster.energy()/seedMom.mag()):(-1.) ;
  tcMatching.eEleClusterOverPout = (eleMom.mag()>0.)?(elbcRef->energy()/eleMom.mag()):(-1.) ;
  tcMatching.deltaEtaSuperClusterAtVtx = scRef->eta()-sclPos_.eta() ;
  tcMatching.deltaEtaSeedClusterAtCalo = seedCluster.eta() - seedPos.eta() ;
  tcMatching.deltaEtaEleClusterAtCalo = elbcRef->eta() - seedPos.eta() ;
  tcMatching.deltaPhiEleClusterAtCalo = normalized_dphi(elbcRef->phi()-elePos.phi()) ;
  tcMatching.deltaPhiSuperClusterAtVtx = normalized_dphi(scRef->phi()-sclPos_.phi()) ;
  tcMatching.deltaPhiSeedClusterAtCalo = normalized_dphi(seedCluster.phi()-seedPos.phi()) ;


  //=======================================================
  // Track extrapolations
  //=======================================================

  reco::GsfElectron::TrackExtrapolations tkExtra ;
  tkExtra.positionAtVtx = convert(vtxPos) ;
  tkExtra.positionAtCalo = convert(sclPos_) ;
  tkExtra.momentumAtVtx = convert(vtxMom_) ;
  tkExtra.momentumAtCalo = convert(sclMom) ;
  tkExtra.momentumOut = convert(seedMom) ;
  tkExtra.momentumAtEleClus = convert(eleMom) ;


  //=======================================================
  // Closest Ctf Track
  //=======================================================

  reco::GsfElectron::ClosestCtfTrack ctfInfo ;
  ctfInfo.ctfTrack = ctfTrackRef  ;
  ctfInfo.shFracInnerHits = shFracInnerHits ;


  //====================================================
  // FiducialFlags, using nextToBoundary definition of gaps
  //====================================================

  reco::GsfElectron::FiducialFlags fiducialFlags ;
  int detector = seedXtalId.subdetId() ;
  double feta=fabs(scRef->position().eta()) ;
  if (detector==EcalBarrel)
   {
	fiducialFlags.isEB = true ;
	if (EBDetId::isNextToEtaBoundary(EBDetId(seedXtalId)))
	 {
	  if (fabs(feta-1.479)<.1)
	   { fiducialFlags.isEBEEGap = true ; }
	  else
	   { fiducialFlags.isEBEtaGap = true ; }
	 }
	if (EBDetId::isNextToPhiBoundary(EBDetId(seedXtalId)))
	 { fiducialFlags.isEBPhiGap = true ; }
   }
  else if (detector==EcalEndcap)
   {
	fiducialFlags.isEE = true ;
	if (EEDetId::isNextToRingBoundary(EEDetId(seedXtalId)))
	 {
	  if (fabs(feta-1.479)<.1)
	   { fiducialFlags.isEBEEGap = true ; }
	  else
	   { fiducialFlags.isEERingGap = true ; }
	 }
	if (EEDetId::isNextToDBoundary(EEDetId(seedXtalId)))
	 { fiducialFlags.isEEDeeGap = true ; }
   }
  else
   { edm::LogWarning("")<<"GsfElectronAlgo::createElectron(): do not know if it is a barrel or endcap seed cluster !!!!" ; }


  //====================================================
  // ShowerShape
  //====================================================

  reco::GsfElectron::ShowerShape showerShape ;
  const CaloTopology * topology = theCaloTopo.product() ;
  const CaloGeometry * geometry = theCaloGeom.product() ;
  const EcalRecHitCollection * reducedRHits = reducedRecHits.product() ;
  std::vector<float> covariances = EcalClusterTools::covariances(seedCluster,reducedRHits,topology,geometry) ;
  std::vector<float> localCovariances = EcalClusterTools::localCovariances(seedCluster,reducedRHits,topology) ;
  showerShape.sigmaEtaEta = sqrt(covariances[0]) ;
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]) ;
  showerShape.e1x5 = EcalClusterTools::e1x5(seedCluster,reducedRHits,topology)  ;
  showerShape.e2x5Max = EcalClusterTools::e2x5Max(seedCluster,reducedRHits,topology)  ;
  showerShape.e5x5 = EcalClusterTools::e5x5(seedCluster,reducedRHits,topology) ;
  showerShape.hcalDepth1OverEcal = HoE1 ;
  showerShape.hcalDepth2OverEcal = HoE2 ;


  //====================================================
  // brems fraction
  //====================================================

  float fbrem = (outMom.mag()>0.)?((innMom.mag()-outMom.mag())/outMom.mag()):1.e30 ;


  //====================================================
  // Go !
  //====================================================

  GsfElectron * ele = new
	GsfElectron
	 ( momentum,coreRef,
	   tcMatching, tkExtra, ctfInfo,
	   fiducialFlags,showerShape,
	   fbrem,0) ;

  // set corrections + classification
  ElectronClassification theClassifier;
  theClassifier.correct(*ele);
  ElectronEnergyCorrector theEnCorrector;
  theEnCorrector.correct(*ele, applyEtaCorrection_);
  ElectronMomentumCorrector theMomCorrector;
  theMomCorrector.correct(*ele,vtxTSOS_);

  // now isolation variables
  reco::GsfElectron::IsolationVariables dr03, dr04 ;
  dr03.tkSumPt = tkIso03.getPtTracks(ele);
  dr03.hcalDepth1TowerSumEt = had1Iso03.getTowerEtSum(ele);
  dr03.hcalDepth2TowerSumEt = had2Iso03.getTowerEtSum(ele);
  dr03.ecalRecHitSumEt = ecalIso03.getEtSum(ele);
  dr04.tkSumPt = tkIso04.getPtTracks(ele);
  dr04.hcalDepth1TowerSumEt = had1Iso04.getTowerEtSum(ele);
  dr04.hcalDepth2TowerSumEt = had2Iso04.getTowerEtSum(ele);
  dr04.ecalRecHitSumEt = ecalIso04.getEtSum(ele);
  ele->setIsolation03(dr03);
  ele->setIsolation04(dr04);

  outEle.push_back(ele) ;
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
