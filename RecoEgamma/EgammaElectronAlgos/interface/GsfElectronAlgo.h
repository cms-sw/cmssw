#ifndef GsfElectronAlgo_H
#define GsfElectronAlgo_H

/** \class GsfElectronAlgo

  Top algorithm producing GsfElectron objects from supercluster driven Gsf tracking

  \author U.Berthon, C.Charlot, LLR Palaiseau

  \version   2nd Version Oct 10, 2006

 ************************************************************/

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <list>

class MultiTrajectoryStateTransform;
class MultiTrajectoryStateMode;
//class GsfConstraintAtVertex;
#include "TrackingTools/GsfTracking/interface/GsfConstraintAtVertex.h"

class GsfElectronAlgo {

  public:

    GsfElectronAlgo(
      const edm::ParameterSet & conf,
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
      double minMVA,
      bool applyPreselection, bool applyEtaCorrection, bool applyAmbResolution,
      bool addPflowElectrons,
      double intRadiusTk, double ptMinTk, double maxVtxDistTk, double maxDrbTk,
      double intRadiusHcal, double etMinHcal, 
      double intRadiusEcalBarrel, double intRadiusEcalEndcaps, double jurassicWidth,
      double etMinBarrel, double eMinBarrel, double etMinEndcaps, double eMinEndcaps,
      bool vetoClustered, bool useNumCrystals) ;

    ~GsfElectronAlgo() ;

    void setupES( const edm::EventSetup & setup ) ;
    void run( edm::Event &, reco::GsfElectronCollection & ) ;

  private :

    // temporary collection of electrons
    typedef std::list<reco::GsfElectron *> GsfElectronPtrCollection ;

    // create electrons from superclusters, tracks and Hcal rechits
    void process
     ( //edm::Handle<reco::GsfTrackCollection> gsfTracksH,
       edm::Handle<reco::GsfElectronCoreCollection> coresH,
       edm::Handle<reco::TrackCollection> ctfTracksH,
       edm::Handle<edm::ValueMap<float> > pfMVAH,
       edm::Handle<CaloTowerCollection> towersH,
       edm::Handle<EcalRecHitCollection> reducedEBRecHits,
       edm::Handle<EcalRecHitCollection> reducedEERecHits,
       const reco::BeamSpot &bs,
       GsfElectronPtrCollection & outEle);

    // interface to be improved...
    void createElectron
     ( const reco::GsfElectronCoreRef & coreRef,
       const reco::CaloClusterPtr & elbcRef,
       const reco::TrackRef & ctfTrackRef, const float shFracInnerHits,
       double HoE1, double HoE2,
       ElectronTkIsolation & tkIso03, ElectronTkIsolation & tkIso04,
       EgammaTowerIsolation & had1Iso03, EgammaTowerIsolation & had2Iso03,
       EgammaTowerIsolation & had1Iso04, EgammaTowerIsolation & had2Iso04,
       EgammaRecHitIsolation & ecalBarrelIso03,EgammaRecHitIsolation & ecalEndcapsIso03,
       EgammaRecHitIsolation & ecalBarrelIso04,EgammaRecHitIsolation & ecalEndcapsIso04,
       edm::Handle<EcalRecHitCollection> reducedEBRecHits,edm::Handle<EcalRecHitCollection> reducedEERecHits,
       float mva, GsfElectronPtrCollection & outEle ) ;

    void preselectElectrons( GsfElectronPtrCollection &, GsfElectronPtrCollection & outEle ) ;

    void resolveElectrons( GsfElectronPtrCollection &, reco::GsfElectronCollection & outEle ) ;

    //Gsf mode calculations
    GlobalVector computeMode(const TrajectoryStateOnSurface &tsos);

    // associations
    const reco::SuperClusterRef getTrSuperCluster(const reco::GsfTrackRef & trackRef);

    const reco::CaloClusterPtr getEleBasicCluster(const reco::GsfTrackRef &
     trackRef, const reco::SuperCluster *scRef);

    // From Puneeth Kalavase : returns the CTF track that has the highest fraction
    // of shared hits in Pixels and the inner strip tracker with the electron Track
    std::pair<reco::TrackRef,float> getCtfTrackRef
     ( const reco::GsfTrackRef &, edm::Handle<reco::TrackCollection> ctfTracksH ) ;

    // intermediate calculations
    bool calculateTSOS(const reco::GsfTrack &t,const reco::SuperCluster & theClus, const
     reco::BeamSpot& bs);

    // preselection parameters
    // minimum SC Et
    double minSCEtBarrel_;
    double minSCEtEndcaps_;
    // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
    double maxEOverPBarrel_;
    double maxEOverPEndcaps_;
    // minimum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
    double minEOverPBarrel_;
    double minEOverPEndcaps_;
    // maximum eta difference between the supercluster position and the track position at the closest impact to the supercluster
    double maxDeltaEtaBarrel_;
    double maxDeltaEtaEndcaps_;
    // maximum phi difference between the supercluster position and the track position at the closest impact to the supercluster
    // position to the supercluster
    double maxDeltaPhiBarrel_;
    double maxDeltaPhiEndcaps_;
    // cone size for H/E evaluation
    double hOverEConeSize_;
    // min tower Et for H/E evaluation
    double hOverEPtMin_;
    // maximum H/E for depth1
    double maxHOverEDepth1Barrel_;
    double maxHOverEDepth1Endcaps_;
    // maximum H/E for depth2
    double maxHOverEDepth2_;
    // maximum sigma ieta ieta
    double maxSigmaIetaIetaBarrel_;
    double maxSigmaIetaIetaEndcaps_;
    // maximum fbrem
    double maxFbremBarrel_;
    double maxFbremEndcaps_;

    // fiducial regions
    bool isBarrel_;
    bool isEndcaps_;
    bool isFiducial_;

    // electron seed
    // select or not electrons with seed having second hit in TEC layers
    bool seedFromTEC_;

    // for tracker driven preselection
    double minMVA_;

    // if this parameter is true, electron preselection is applied
    bool applyPreselection_;

    // if this parameter is true, electron level escale corrections are used on top
    // of the cluster level corrections
    bool applyEtaCorrection_;

    // if this parameter is true, "double" electrons are resolved
    bool applyAmbResolution_;

    // if this parameter is true, trackerDriven electrons are added
    bool addPflowElectrons_;

    // isolation variables parameters
    double intRadiusTk_;
    double ptMinTk_;
    double maxVtxDistTk_;
    double maxDrbTk_;
    double intRadiusHcal_;
    double etMinHcal_;
    double intRadiusEcalBarrel_;
    double intRadiusEcalEndcaps_;
    double jurassicWidth_;
    double etMinBarrel_;
    double eMinBarrel_;
    double etMinEndcaps_;
    double eMinEndcaps_;
    bool vetoClustered_;
    bool useNumCrystals_;
    
    // input configuration
    edm::InputTag barrelSuperClusters_;
    edm::InputTag endcapSuperClusters_;
    //edm::InputTag tracks_;
    edm::InputTag gsfElectronCores_ ;
    edm::InputTag ctfTracks_;
    edm::InputTag hcalTowers_;
    edm::InputTag reducedBarrelRecHitCollection_ ;
    edm::InputTag reducedEndcapRecHitCollection_ ;
    edm::InputTag pfMVA_;


    edm::ESHandle<MagneticField>                theMagField;
    edm::ESHandle<CaloGeometry>                 theCaloGeom;
    edm::ESHandle<CaloTopology>                 theCaloTopo;
    edm::ESHandle<TrackerGeometry>              trackerHandle_;

    const MultiTrajectoryStateTransform *mtsTransform_;
    const MultiTrajectoryStateMode *mtsMode_;
    GsfConstraintAtVertex *constraintAtVtx_;

    // internal variables
    int subdet_; //subdetector for this cluster
    GlobalPoint sclPos_;
    GlobalVector vtxMom_;
    TrajectoryStateOnSurface innTSOS_;
    TrajectoryStateOnSurface outTSOS_;
    TrajectoryStateOnSurface vtxTSOS_;
    TrajectoryStateOnSurface sclTSOS_;
    TrajectoryStateOnSurface seedTSOS_;
    TrajectoryStateOnSurface eleTSOS_;
    TrajectoryStateOnSurface constrainedVtxTSOS_;

    unsigned long long cacheIDGeom_;
    unsigned long long cacheIDTopo_;
    unsigned long long cacheIDTDGeom_;
    unsigned long long cacheIDMagField_;

 } ;

#endif // GsfElectronAlgo_H


