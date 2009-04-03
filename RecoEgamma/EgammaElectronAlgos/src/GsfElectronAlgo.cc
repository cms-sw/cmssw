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
// $Id: GsfElectronAlgo.cc,v 1.51 2009/03/28 22:27:13 charlot Exp $
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
   bool applyPreselection, bool applyEtaCorrection, bool applyAmbResolution, 
   bool addPflowElectrons,
   double extRadiusTkSmall, double extRadiusTkLarge, double intRadiusTk,
   double ptMinTk, double maxVtxDistTk, double maxDrbTk,
   double extRadiusHcalSmall, double extRadiusHcalLarge, double intRadiusHcal,
   double etMinHcal, double extRadiusEcalSmall, double extRadiusEcalLarge,
   double intRadiusEcalBarrel, double intRadiusEcalEndcaps, double jurassicWidth,
   double etMinBarrel, double eMinBarrel, double etMinEndcaps, double eMinEndcaps)    
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
   addPflowElectrons_(addPflowElectrons),
   extRadiusTkSmall_(extRadiusTkSmall),  extRadiusTkLarge_(extRadiusTkLarge),  intRadiusTk_(intRadiusTk),
   ptMinTk_(ptMinTk),  maxVtxDistTk_(maxVtxDistTk),  maxDrbTk_(maxDrbTk),
   extRadiusHcalSmall_(extRadiusHcalSmall),  extRadiusHcalLarge_(extRadiusHcalLarge),  intRadiusHcal_(intRadiusHcal),
   etMinHcal_(etMinHcal),  extRadiusEcalSmall_(extRadiusEcalSmall),  extRadiusEcalLarge_(extRadiusEcalLarge),
   intRadiusEcalBarrel_(),  intRadiusEcalEndcaps_(),  jurassicWidth_(),
   etMinBarrel_(etMinBarrel),  eMinBarrel_(eMinBarrel),  etMinEndcaps_(etMinEndcaps),  eMinEndcaps_(eMinEndcaps),    
   cacheIDGeom_(0),cacheIDTopo_(0),cacheIDTDGeom_(0),cacheIDMagField_(0)
 {
  // this is the new version allowing to configurate the algo
  // interfaces still need improvement!!
  mtsTransform_ = 0 ;
  constraintAtVtx_ = 0;
  
  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;

  // get input collections
  hcalTowers_ = conf.getParameter<edm::InputTag>("hcalTowers");
  tracks_ = conf.getParameter<edm::InputTag>("tracks");
  gsfElectronCores_ = conf.getParameter<edm::InputTag>("gsfElectronCores");
  ctfTracks_ = conf.getParameter<edm::InputTag>("ctfTracks");
  reducedBarrelRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedBarrelRecHitCollection") ;
  reducedEndcapRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedEndcapRecHitCollection") ;
  pfMVA_ = conf.getParameter<edm::InputTag>("pfMVA") ;
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

  if ( updateField || updateGeometry ) {
    delete constraintAtVtx_;
    constraintAtVtx_ = new GsfConstraintAtVertex(es);
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
  edm::Handle<edm::ValueMap<float> > pfMVAH;
  e.getByLabel(pfMVA_,pfMVAH); 

  // get the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByType(recoBeamSpotHandle);
  const BeamSpot bs = *recoBeamSpotHandle;

  // temporay array for electrons before preselection and before amb. solving
  GsfElectronPtrCollection tempEle, tempEle1;

  // create electrons
  process(tracksH,coresH,ctfTracksH,pfMVAH,towersH,pEBRecHits,pEERecHits,bs,tempEle);

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
  edm::Handle<edm::ValueMap<float> > pfMVAH,
  edm::Handle<CaloTowerCollection> towersH,
  edm::Handle<EcalRecHitCollection> reducedEBRecHits,
  edm::Handle<EcalRecHitCollection> reducedEERecHits,
  const BeamSpot & bs,
  GsfElectronPtrCollection & outEle )
 {

  const edm::ValueMap<float> & pfmvas = *pfMVAH.product() ;

  // Isolation algos

  float extRadiusSmall=extRadiusTkSmall_, extRadiusLarge=extRadiusTkLarge_, intRadius=intRadiusTk_;
  float ptMin=ptMinTk_, maxVtxDist=maxVtxDistTk_, drb=maxDrbTk_; 
  ElectronTkIsolation tkIsolation03(extRadiusSmall,intRadius,ptMin,maxVtxDist,drb,ctfTracksH.product(),bs.position()) ;
  ElectronTkIsolation tkIsolation04(extRadiusLarge,intRadius,ptMin,maxVtxDist,drb,ctfTracksH.product(),bs.position()) ;
  
  float egHcalIsoConeSizeOutSmall=extRadiusHcalSmall_, egHcalIsoConeSizeOutLarge=extRadiusHcalLarge_;
  float egHcalIsoConeSizeIn=intRadiusHcal_,egHcalIsoPtMin=etMinHcal_;
  int egHcalDepth1=1, egHcalDepth2=2;  
  EgammaTowerIsolation hadDepth1Isolation03(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,towersH.product()) ;
  EgammaTowerIsolation hadDepth2Isolation03(egHcalIsoConeSizeOutSmall,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,towersH.product()) ;
  EgammaTowerIsolation hadDepth1Isolation04(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth1,towersH.product()) ;
  EgammaTowerIsolation hadDepth2Isolation04(egHcalIsoConeSizeOutLarge,egHcalIsoConeSizeIn,egHcalIsoPtMin,egHcalDepth2,towersH.product()) ;
  
  float egIsoConeSizeOutSmall=extRadiusEcalSmall_, egIsoConeSizeOutLarge=extRadiusEcalLarge_, egIsoJurassicWidth=jurassicWidth_;
  float egIsoPtMinBarrel=etMinBarrel_,egIsoEMinBarrel=eMinBarrel_, egIsoConeSizeInBarrel=intRadiusEcalBarrel_;
  float egIsoPtMinEndcap=etMinEndcaps_,egIsoEMinEndcap=eMinEndcaps_, egIsoConeSizeInEndcap=intRadiusEcalEndcaps_;
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

    // retreive core, track and scl
    const GsfElectronCoreRef coreRef = edm::Ref<GsfElectronCoreCollection>(coresH,i);
    const GsfTrackRef gsfTrackRef = coreRef->gsfTrack() ; //edm::Ref<GsfTrackCollection>(gsfTracksH,i);
   
    // Get the super cluster
    SuperClusterRef scRef = coreRef->superCluster() ;

   // don't add pflow only electrons one so wish
    if (!coreRef->isEcalDriven() && !addPflowElectrons_) continue;

    // Get the pflow super cluster
    SuperClusterRef pfscRef = coreRef->pflowSuperCluster() ;

    if (scRef.isNull()&&pfscRef.isNull()) continue ;

    // affect main electron cluster according to provenance
    SuperCluster theClus;
    if (coreRef->isEcalDriven()) {
      std::cout << "[GsfElectronAlgo] found by e/g " << std::endl;
      theClus = *scRef ;
    } else {
      std::cout << "[GsfElectronAlgo] NOT found by e/g " << std::endl;      
      theClus = *pfscRef ;
    }
     
    // mva
    float mva=0.;
    if (coreRef->isTrackerDriven()) mva = pfmvas[gsfTrackRef];

    // electron basic cluster
    CaloClusterPtr elbcRef = getEleBasicCluster(gsfTrackRef,&theClus) ;

    // calculate Trajectory StatesOnSurface....
    if (!calculateTSOS(*gsfTrackRef,theClus, bs)) continue ;
    mtsMode_->momentumFromModeCartesian(vtxTSOS_,vtxMom_) ;
    sclPos_=sclTSOS_.globalPosition() ;
    
    // H/E
    double HoE1=towerIso1.getTowerESum(&theClus)/theClus.energy();
    double HoE2=towerIso2.getTowerESum(&theClus)/theClus.energy();

    pair<TrackRef,float> ctfpair = getCtfTrackRef(gsfTrackRef,ctfTracksH) ;
    const TrackRef ctfTrackRef = ctfpair.first ;
    const float fracShHits = ctfpair.second ;

    createElectron(coreRef,elbcRef,ctfTrackRef,fracShHits,HoE1,HoE2,tkIsolation03,tkIsolation04,
     hadDepth1Isolation03,hadDepth2Isolation03,hadDepth1Isolation04,hadDepth2Isolation04,
     ecalBarrelIsol03,ecalBarrelIsol04,ecalEndcapIsol03,ecalEndcapIsol04,reducedEBRecHits,
     reducedEERecHits,mva,outEle) ;

     LogInfo("")<<"Constructed new electron with energy  "<< theClus.energy();

  } // loop over tracks
}

void GsfElectronAlgo::preselectElectrons( GsfElectronPtrCollection & inEle, GsfElectronPtrCollection & outEle )
 {
  GsfElectronPtrCollection::iterator e1;
  for( e1 = inEle.begin() ;  e1 != inEle.end() ; ++e1 )
   {

    LogDebug("")<< "========== preSelection ==========";
    
    
    // pflow only case
    if ((*e1)->core()->isTrackerDriven() && !(*e1)->core()->isEcalDriven()) {
    
    outEle.push_back(*e1) ;
   
    } else {
        
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
   const CaloClusterPtr & elbcRef,
   const TrackRef & ctfTrackRef, const float shFracInnerHits,
   double HoE1, double HoE2,
   ElectronTkIsolation & tkIso03, ElectronTkIsolation & tkIso04,
   EgammaTowerIsolation & had1Iso03, EgammaTowerIsolation & had2Iso03, 
   EgammaTowerIsolation & had1Iso04, EgammaTowerIsolation & had2Iso04, 
   EgammaRecHitIsolation & ecalBarrelIso03,EgammaRecHitIsolation & ecalEndcapsIso03,
   EgammaRecHitIsolation & ecalBarrelIso04,EgammaRecHitIsolation & ecalEndcapsIso04,
   edm::Handle<EcalRecHitCollection> reducedEBRecHits,edm::Handle<EcalRecHitCollection> reducedEERecHits,
   float mva, GsfElectronPtrCollection & outEle )

 {
  SuperClusterRef egscRef = coreRef->superCluster() ;
  SuperClusterRef pfscRef = coreRef->pflowSuperCluster() ;
  GsfTrackRef trackRef = coreRef->gsfTrack() ;

  SuperClusterRef scRef;
  if (egscRef.isNull()&&pfscRef.isNull()) return ;

  if (coreRef->isEcalDriven()) scRef = egscRef ;  // electron wwill be built from e/g sc
  else scRef=pfscRef ;                            // electron wwill be built from pflow sc 

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
  GlobalVector vtxMomWithConstraint;
  bool success = mtsMode_->momentumFromModeCartesian(constrainedVtxTSOS_,vtxMomWithConstraint);
//    if ( success )
//      std::cout << " pt = " << momentum.perp() << std::endl;
//    else
//      std::cout << " FAILURE!!!" << std::endl;


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
  tkExtra.momentumAtVtxWithConstraint = convert(vtxMomWithConstraint);

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
  const EcalRecHitCollection * reducedRecHits = 0 ;
  if (fiducialFlags.isEB)
   { reducedRecHits = reducedEBRecHits.product() ; }
  else
   { reducedRecHits = reducedEERecHits.product() ; }
  std::vector<float> covariances = EcalClusterTools::covariances(seedCluster,reducedRecHits,topology,geometry) ;
  std::vector<float> localCovariances = EcalClusterTools::localCovariances(seedCluster,reducedRecHits,topology) ;
  showerShape.sigmaEtaEta = sqrt(covariances[0]) ;
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]) ;
  showerShape.e1x5 = EcalClusterTools::e1x5(seedCluster,reducedRecHits,topology)  ;
  showerShape.e2x5Max = EcalClusterTools::e2x5Max(seedCluster,reducedRecHits,topology)  ;
  showerShape.e5x5 = EcalClusterTools::e5x5(seedCluster,reducedRecHits,topology) ;
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
	   fbrem,mva) ;

  // set corrections + classification
  // temporary, only if ecalDriven
  if (ele->core()->isEcalDriven()) {
    ElectronClassification theClassifier;
    theClassifier.correct(*ele);
    ElectronEnergyCorrector theEnCorrector;
    theEnCorrector.correct(*ele, applyEtaCorrection_);
    ElectronMomentumCorrector theMomCorrector;
    theMomCorrector.correct(*ele,vtxTSOS_);
  }
  
  // now isolation variables
  reco::GsfElectron::IsolationVariables dr03, dr04 ;
  // temporary, only if ecalDriven
  if (ele->core()->isEcalDriven()) {
  dr03.tkSumPt = tkIso03.getPtTracks(ele);
  dr03.hcalDepth1TowerSumEt = had1Iso03.getTowerEtSum(ele);
  dr03.hcalDepth2TowerSumEt = had2Iso03.getTowerEtSum(ele);
  dr03.ecalRecHitSumEt = ecalBarrelIso03.getEtSum(ele)+ecalEndcapsIso03.getEtSum(ele);
  dr04.tkSumPt = tkIso04.getPtTracks(ele);
  dr04.hcalDepth1TowerSumEt = had1Iso04.getTowerEtSum(ele);
  dr04.hcalDepth2TowerSumEt = had2Iso04.getTowerEtSum(ele);
  dr04.ecalRecHitSumEt = ecalBarrelIso04.getEtSum(ele)+ecalEndcapsIso04.getEtSum(ele);
  }
  ele->setIsolation03(dr03);
  ele->setIsolation04(dr04);
  outEle.push_back(ele) ;
 }


const CaloClusterPtr GsfElectronAlgo::getEleBasicCluster(const GsfTrackRef &t, const SuperCluster *scRef) {

    CaloClusterPtr eleRef;
    TrajectoryStateOnSurface tempTSOS;
    TrajectoryStateOnSurface outTSOS = mtsTransform_->outerStateOnSurface(*t);
    float dphimin = 1.e30;
    for (CaloCluster_iterator bc=scRef->clustersBegin(); bc!=scRef->clustersEnd(); bc++) {
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

bool  GsfElectronAlgo::calculateTSOS(const GsfTrack &t,const SuperCluster & theClus, const BeamSpot & bs){

    //at innermost point
    innTSOS_ = mtsTransform_->innerStateOnSurface(t);
    if (!innTSOS_.isValid()) return false;

    //at vertex
    // innermost state propagation to the beam spot position
    vtxTSOS_ = mtsTransform_->extrapolatedState(innTSOS_,
	GlobalPoint(bs.position().x(),bs.position().y(),bs.position().z()));
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

    // constrained momentum
    constrainedVtxTSOS_ = constraintAtVtx_->constrainAtBeamSpot(t,bs);    

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

  // resolve when e/g SC is found
  for( e1 = inEle.begin() ;  e1 != inEle.end() ; ++e1 )
   {
    SuperClusterRef scRef1 = (*e1)->superCluster();
    if (scRef1.isNull()) scRef1 = (*e1)->pflowSuperCluster();
    LogDebug("GsfElectronAlgo")
      << "Blessing electron with E/P " << (*e1)->eSuperClusterOverP()
      << ", cluster " << scRef1.get()
      << " & track " << (*e1)->gsfTrack().get() ;
    
    for( e2 = e1, ++e2 ;  e2 != inEle.end() ; )
     {
      SuperClusterRef scRef2 = (*e2)->superCluster();
      if (scRef2.isNull()) scRef2 = (*e2)->pflowSuperCluster();
      if (scRef1==scRef2)
       {
        LogDebug("GsfElectronAlgo")
          << "Discarding electron with E/P " << (*e2)->eSuperClusterOverP()
          << ", cluster " << scRef2.get()
          << " and track " << (*e2)->gsfTrack().get() ;
        (*e1)->addAmbiguousGsfTrack((*e2)->gsfTrack()) ;
        e2 = inEle.erase(e2) ;
       }
      else if ((*e1)->gsfTrack()==(*e2)->gsfTrack())
       {
        LogDebug("GsfElectronAlgo")
          << "Forgetting electron with E/P " << (*e2)->eSuperClusterOverP()
          << ", cluster " << scRef2.get()
          << " and track " << (*e2)->gsfTrack().get() ;
        e2 = inEle.erase(e2) ;
       }
      else
       { ++e2 ; }
     }
    outEle.push_back(**e1) ;
   }
   
//   // nex resolve when pflow only SC is found
//   for( e1 = inEle.begin() ;  e1 != inEle.end() ; ++e1 )
//    {
//     SuperClusterRef scRef1 = (*e1)->superCluster();
//     SuperClusterRef pfscRef1 = (*e1)->pflowSuperCluster();
//     if (!scRef1.isNull() || pfscRef1.isNull()) continue;
//    
//     LogDebug("GsfElectronAlgo")
//       << "Blessing electron with E/P " << (*e1)->eSuperClusterOverP()
//       << ", cluster " << pfscRef1.get()
//       << " & track " << (*e1)->gsfTrack().get() ;
//     
//     for( e2 = e1, ++e2 ;  e2 != inEle.end() ; )
//      {
//       SuperClusterRef scRef2 = (*e2)->superCluster();
//       SuperClusterRef pfscRef2 = (*e2)->pflowSuperCluster();
//       if (!scRef2.isNull() || pfscRef2.isNull()) continue;
//       if (pfscRef1==pfscRef2)
//        {
//         LogDebug("GsfElectronAlgo")
//           << "Discarding electron with E/P " << (*e2)->eSuperClusterOverP()
//           << ", cluster " << pfscRef2.get()
//           << " and track " << (*e2)->gsfTrack().get() ;
//         (*e1)->addAmbiguousGsfTrack((*e2)->gsfTrack()) ;
//         e2 = inEle.erase(e2) ;
//        }
//       else if ((*e1)->gsfTrack()==(*e2)->gsfTrack())
//        {
//         LogDebug("GsfElectronAlgo")
//           << "Forgetting electron with E/P " << (*e2)->eSuperClusterOverP()
//           << ", cluster " << pfscRef2.get()
//           << " and track " << (*e2)->gsfTrack().get() ;
//         e2 = inEle.erase(e2) ;
//        }
//       else
//        { ++e2 ; }
//      }
//     outEle.push_back(**e1) ;
//    }
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
