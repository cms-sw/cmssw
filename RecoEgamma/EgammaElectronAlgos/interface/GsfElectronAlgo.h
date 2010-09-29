
#ifndef GsfElectronAlgo_H
#define GsfElectronAlgo_H

/************************************************************

  \class GsfElectronAlgo

  Top algorithm producing GsfElectron objects from supercluster driven Gsf tracking

  \author U.Berthon, C.Charlot, LLR Palaiseau

  \version   2nd Version Oct 10, 2006

 ************************************************************/

class ElectronHcalHelper ;
class MultiTrajectoryStateTransform ;
class MultiTrajectoryStateMode ;
class EcalClusterFunctionBaseClass ;

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTracking/interface/GsfConstraintAtVertex.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include <list>
#include <string>

class GsfElectronAlgo {

  public:

    GsfElectronAlgo
     (
      const edm::ParameterSet & conf,
      double minSCEtBarrel, double minSCEtEndcaps,
      double maxEOverPBarrel, double maxEOverPEndcaps,
      double minEOverPBarrel, double minEOverPEndcaps,
      double maxDeltaEtaBarrel, double maxDeltaEtaEndcaps,
      double maxDeltaPhiBarrel,double maxDeltaPhiEndcaps,
      double maxSigmaIetaIetaBarrel, double maxSigmaIetaIetaEndcaps,
      double maxFbremBarrel, double maxFbremEndcaps,
      bool isBarrel, bool isEndcaps, bool isFiducial,
      bool seedFromTEC,
      double minMVA, double maxTIP,
      double minSCEtBarrelPflow, double minSCEtEndcapsPflow,
      double maxEOverPBarrelPflow, double maxEOverPEndcapsPflow,
      double minEOverPBarrelPflow, double minEOverPEndcapsPflow,
      double maxDeltaEtaBarrelPflow, double maxDeltaEtaEndcapsPflow,
      double maxDeltaPhiBarrelPflow,double maxDeltaPhiEndcapsPflow,
//      double hOverEConeSizePflow, double hOverEPtMinPflow,
//      double maxHOverEDepth1BarrelPflow, double maxHOverEDepth1EndcapsPflow,
//      double maxHOverEDepth2Pflow,
      double maxSigmaIetaIetaBarrelPflow, double maxSigmaIetaIetaEndcapsPflow,
      double maxFbremBarrelPflow, double maxFbremEndcapsPflow,
      bool isBarrelPflow, bool isEndcapsPflow, bool isFiducialPflow,
      double minMVAPflow, double maxTIPPflow,
      bool applyPreselection, bool applyEtaCorrection,
      bool applyAmbResolution, unsigned ambSortingStrategy, unsigned ambClustersOverlapStrategy,
      bool addPflowElectrons,
      double intRadiusBarrelTk, double intRadiusEndcapTk, double stripBarrelTk, double stripEndcapTk,
      double ptMinTk, double maxVtxDistTk, double maxDrbTk,
      double intRadiusHcal, double etMinHcal,
      double intRadiusEcalBarrel, double intRadiusEcalEndcaps, double jurassicWidth,
      double etMinBarrel, double eMinBarrel, double etMinEndcaps, double eMinEndcaps,
      bool vetoClustered, bool useNumCrystals, int severityLevelCut_,
      float severityRecHitThreshold_, float spikeIdThreshold_, std::string spikeIdString_
     ) ;

    ~GsfElectronAlgo() ;

    void setupES( const edm::EventSetup & setup ) ;
    void run( edm::Event &, reco::GsfElectronCollection & ) ;

  private :

    // for temporary collection of electrons
    typedef std::list<reco::GsfElectron *> GsfElectronPtrCollection ;

    // create electrons from superclusters, tracks and Hcal rechits
    void process
     ( edm::Handle<reco::GsfElectronCoreCollection> coresH,
       edm::Handle<reco::TrackCollection> ctfTracksH,
       edm::Handle<edm::ValueMap<float> > pfMVAH,
       edm::Handle<CaloTowerCollection> towersH,
       edm::Handle<EcalRecHitCollection> reducedEBRecHits,
       edm::Handle<EcalRecHitCollection> reducedEERecHits,
       const reco::BeamSpot & bs,
       GsfElectronPtrCollection & outEle ) ;

    void createElectron
     ( const reco::GsfElectronCoreRef & coreRef,
       int charge, const reco::GsfElectron::ChargeInfo & chargeInfo,
       const reco::CaloClusterPtr & elbcRef,
       const reco::TrackRef & ctfTrackRef, const float shFracInnerHits,
       double HoE1, double HoE2,
       ElectronTkIsolation & tkIso03, ElectronTkIsolation & tkIso04,
       EgammaTowerIsolation & had1Iso03, EgammaTowerIsolation & had2Iso03,
       EgammaTowerIsolation & had1Iso04, EgammaTowerIsolation & had2Iso04,
       EgammaRecHitIsolation & ecalBarrelIso03,EgammaRecHitIsolation & ecalEndcapsIso03,
       EgammaRecHitIsolation & ecalBarrelIso04,EgammaRecHitIsolation & ecalEndcapsIso04,
       edm::Handle<EcalRecHitCollection> reducedEBRecHits,edm::Handle<EcalRecHitCollection> reducedEERecHits,
       float mva, const reco::BeamSpot & bs, GsfElectronPtrCollection & outEle ) ;

    void preselectElectrons( GsfElectronPtrCollection & inEle, GsfElectronPtrCollection & outEle, const reco::BeamSpot& ) ;
    bool preselectCutBasedFlag( reco::GsfElectron * ele, const reco::BeamSpot & ) ;
    bool preselectMvaFlag( reco::GsfElectron * ele ) ;

    void resolveElectrons
     ( GsfElectronPtrCollection &, reco::GsfElectronCollection & outEle,
       edm::Handle<EcalRecHitCollection> & reducedEBRecHits,
       edm::Handle<EcalRecHitCollection> & reducedEERecHits,
       const reco::BeamSpot & bs ) ;

    // associations
    const reco::SuperClusterRef getTrSuperCluster(const reco::GsfTrackRef & trackRef );

    const reco::CaloClusterPtr getEleBasicCluster
     ( const reco::GsfTrackRef & trackRef,
       const reco::SuperCluster * scRef,
       const reco::BeamSpot & bs ) ;

    // From Puneeth Kalavase : returns the CTF track that has the highest fraction
    // of shared hits in Pixels and the inner strip tracker with the electron Track
    std::pair<reco::TrackRef,float> getCtfTrackRef
     ( const reco::GsfTrackRef &, edm::Handle<reco::TrackCollection> ctfTracksH ) ;

    // intermediate calculations
    bool calculateTSOS(const reco::GsfTrack &t,const reco::SuperCluster & theClus, const
     reco::BeamSpot& bs);

    void computeCharge
     ( const reco::GsfTrackRef & tk,
       const reco::TrackRef & ctf,
       const reco::SuperClusterRef & sc,
       const reco::BeamSpot & bs,
       int & charge, reco::GsfElectron::ChargeInfo & info ) ;

    // preselection parameters (ecal driven electrons)
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

    // H/E evaluation
    //bool useHcalRecHits_ ;
    ElectronHcalHelper * hcalHelper_, * hcalHelperPflow_ ;
    //bool useHcalTowers_ ;
    edm::InputTag hcalTowers_;      // parameter if use towers
    double hOverEConeSize_;         // parameter if use towers
    double hOverEPtMin_;            // parameter if use towers : min tower Et for H/E evaluation
    //double maxHOverEDepth1Barrel_;  // parameter if use towers : maximum H/E for depth1
    //double maxHOverEDepth1Endcaps_; // parameter if use towers : maximum H/E for depth1
    //double maxHOverEDepth2_;        // parameter if use towers : maximum H/E for depth2
    double maxHOverEBarrel_;  // parameter if use towers : maximum H/E for Barrel
    double maxHOverEEndcaps_; // parameter if use towers : maximum H/E for Endcaps
    double maxHBarrel_;  // parameter if use towers : maximum H for Barrel
    double maxHEndcaps_; // parameter if use towers : maximum H for Endcaps

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
    // select electrons with seed second hit in TEC layers
    bool seedFromTEC_;
    // BDT output (if available)
    double minMVA_;
    // transverse impact parameter wrt beam spot
    double maxTIP_;

    // preselection parameters (tracker driven only electrons)
    // minimum SC Et
    double minSCEtBarrelPflow_;
    double minSCEtEndcapsPflow_;
    // maximum E/p where E is the pflow supercluster energy and p the track momentum at innermost state
    double maxEOverPBarrelPflow_;
    double maxEOverPEndcapsPflow_;
    // minimum E/p where E is the pflow supercluster energy and p the track momentum at innermost state
    double minEOverPBarrelPflow_;
    double minEOverPEndcapsPflow_;
    // maximum eta difference between the pflow supercluster position and the track position at the closest impact to the supercluster
    double maxDeltaEtaBarrelPflow_;
    double maxDeltaEtaEndcapsPflow_;
    // maximum phi difference between the pflow supercluster position and the track position at the closest impact to the supercluster
    // position to the supercluster
    double maxDeltaPhiBarrelPflow_;
    double maxDeltaPhiEndcapsPflow_;
    // cone size for H/E evaluation
    double hOverEConeSizePflow_;
    // min tower Et for H/E evaluation
    double hOverEPtMinPflow_;
    //// maximum H/E for depth1
    //double maxHOverEDepth1BarrelPflow_;
    //double maxHOverEDepth1EndcapsPflow_;
    //// maximum H/E for depth2
    //double maxHOverEDepth2Pflow_;
    // maximum H/E
    double maxHOverEBarrelPflow_;
    double maxHOverEEndcapsPflow_;
    // maximum H
    double maxHBarrelPflow_;
    double maxHEndcapsPflow_;
    // maximum sigma ieta ieta
    double maxSigmaIetaIetaBarrelPflow_;
    double maxSigmaIetaIetaEndcapsPflow_;
    // maximum fbrem
    double maxFbremBarrelPflow_;
    double maxFbremEndcapsPflow_;
    // fiducial regions
    bool isBarrelPflow_;
    bool isEndcapsPflow_;
    bool isFiducialPflow_;
    // BDT output
    double minMVAPflow_;
    // transverse impact parameter wrt beam spot
    double maxTIPPflow_;


    // if this parameter is true, electron preselection is applied
    bool applyPreselection_;

    // if this parameter is true, electron level escale corrections are used on top
    // of the cluster level corrections
    bool applyEtaCorrection_;

    // ambiguity solving
    bool applyAmbResolution_ ; // if not true, ambiguity solving is not applied
    unsigned ambSortingStrategy_ ; // 0:isBetter, 1:isInnerMost
    unsigned ambClustersOverlapStrategy_ ; // 0:sc adresses, 1:bc shared energy


    // if this parameter is true, trackerDriven electrons are added
    bool addPflowElectrons_;

    // isolation variables parameters
    double intRadiusBarrelTk_;
    double intRadiusEndcapTk_;
    double stripBarrelTk_;
    double stripEndcapTk_;
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

    int    severityLevelCut_;
    float  severityRecHitThreshold_;
    float spikeIdThreshold_;
    std::string spikeIdString_;
    EcalSeverityLevelAlgo::SpikeId spId_;


    // input configuration
    edm::InputTag barrelSuperClusters_;
    edm::InputTag endcapSuperClusters_;
    //edm::InputTag tracks_;
    edm::InputTag gsfElectronCores_ ;
    edm::InputTag reducedBarrelRecHitCollection_ ;
    edm::InputTag reducedEndcapRecHitCollection_ ;
    edm::InputTag pfMVA_;
    edm::InputTag seedsTag_;
    bool ctfTracksCheck_ ;
    edm::InputTag ctfTracks_;
    edm::InputTag beamSpotTag_;

    edm::ESHandle<MagneticField>                theMagField;
    edm::ESHandle<CaloGeometry>                 theCaloGeom;
    edm::ESHandle<CaloTopology>                 theCaloTopo;
    edm::ESHandle<TrackerGeometry>              trackerHandle_;
    edm::ESHandle<EcalChannelStatus>            theChStatus;

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
    unsigned long long cacheChStatus_;

    EcalClusterFunctionBaseClass * superClusterErrorFunction_ ;

    bool pfTranslatorParametersChecked_ ;
    void checkPfTranslatorParameters( edm::ParameterSetID const & ) ;
    bool ecalSeedingParametersChecked_ ;
    void checkEcalSeedingParameters( edm::ParameterSetID const & ) ;

 } ;

#endif // GsfElectronAlgo_H


