#ifndef GsfElectronAlgo_H
#define GsfElectronAlgo_H

/** \class GsfElectronAlgo

  Top algorithm producing GsfElectron objects from supercluster driven Gsf tracking

  \author U.Berthon, C.Charlot, LLR Palaiseau

  \version   2nd Version Oct 10, 2006

 ************************************************************/

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include <list>

class MultiTrajectoryStateTransform;
class GsfPropagatorAdapter;

class GsfElectronAlgo {

  public:

    GsfElectronAlgo(
      const edm::ParameterSet & conf,
      double maxEOverPBarrel, double maxEOverPEndcaps,
      double minEOverPBarrel, double minEOverPEndcaps,
      double maxDeltaEta, double maxDeltaPhi,
      double hOverEConeSize, double hOverEPtMin,
      double maxHOverEDepth1, double maxHOverEDepth2,
      bool applyEtaCorrection, bool applyAmbResolution
		) ;
    ~GsfElectronAlgo() ;

    void setupES( const edm::EventSetup & setup ) ;
    void run( edm::Event &, reco::GsfElectronCollection & ) ;

  private :

    // temporary collection of electrons
    typedef std::list<reco::GsfElectron *> GsfElectronPtrCollection ;

    // create electrons from superclusters, tracks and Hcal rechits
    void process
     ( edm::Handle<reco::GsfTrackCollection> tracksH,
       edm::Handle<reco::TrackCollection> ctfTracksH,
       edm::Handle<CaloTowerCollection> towersH,
       edm::Handle<EcalRecHitCollection> reducedEBRecHits,
       edm::Handle<EcalRecHitCollection> reducedEERecHits,
       const math::XYZPoint &bs,
       GsfElectronPtrCollection & outEle);

    // preselection method
    bool preSelection( const reco::SuperCluster &, double HoE1, double HoE2 ) ;

    // interface to be improved...
    void createElectron
     ( const reco::SuperClusterRef & scRef,
       const reco::BasicClusterRef & elbcRef, 
       const reco::GsfTrackRef & trackRef,
       const reco::TrackRef & ctfTrackRef, const float shFracInnerHits,
       double HoE1, double HoE2,
       edm::Handle<EcalRecHitCollection> reducedEBRecHits,
       edm::Handle<EcalRecHitCollection> reducedEERecHits,
       GsfElectronPtrCollection & outEle ) ;

    void resolveElectrons( GsfElectronPtrCollection &, reco::GsfElectronCollection & outEle ) ;

    //Gsf mode calculations
    GlobalVector computeMode(const TrajectoryStateOnSurface &tsos);

    // associations
    const reco::SuperClusterRef getTrSuperCluster(const reco::GsfTrackRef & trackRef);

    const reco::BasicClusterRef getEleBasicCluster(const reco::GsfTrackRef &
     trackRef, const reco::SuperClusterRef & scRef);

    // From Puneeth Kalavase : returns the CTF track that has the highest fraction
    // of shared hits in Pixels and the inner strip tracker with the electron Track
    std::pair<reco::TrackRef,float> getCtfTrackRef
     ( const reco::GsfTrackRef &, edm::Handle<reco::TrackCollection> ctfTracksH ) ;

    // intermediate calculations
    bool calculateTSOS(const reco::GsfTrack &t,const reco::SuperCluster & theClus, const math::XYZPoint & bs);

    // preselection parameters
    // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
    double maxEOverPBarrel_;
    double maxEOverPEndcaps_;
    // minimum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
    double minEOverPBarrel_;
    double minEOverPEndcaps_;
    // maximum eta difference between the supercluster position and the track position at the closest impact to the supercluster
    double maxDeltaEta_;
    // maximum phi difference between the supercluster position and the track position at the closest impact to the supercluster
    // position to the supercluster
    double maxDeltaPhi_;
    // cone size for H/E evaluation
    double hOverEConeSize_;
    // min tower Et for H/E evaluation
    double hOverEPtMin_;
    // minimum H/E for depth1
    double maxHOverEDepth1_;
    // minimum H/E for depth2
    double maxHOverEDepth2_;

    // if this parameter is false, only SC level Escale correctoins are applied
    bool applyEtaCorrection_;

    // if this parameter is true, "double" electrons are resolved
    bool applyAmbResolution_;

    // input configuration
    edm::InputTag barrelSuperClusters_;
    edm::InputTag endcapSuperClusters_;
    edm::InputTag tracks_;
    edm::InputTag ctfTracks_;
    edm::InputTag hcalTowers_;
    edm::InputTag reducedBarrelRecHitCollection_ ;
    edm::InputTag reducedEndcapRecHitCollection_ ;


    edm::ESHandle<MagneticField>                theMagField;
    edm::ESHandle<CaloGeometry>                 theCaloGeom;
    edm::ESHandle<CaloTopology>                 theCaloTopo;
    edm::ESHandle<TrackerGeometry>              trackerHandle_;

    const MultiTrajectoryStateTransform *mtsTransform_;
    const GsfPropagatorAdapter *geomPropBw_;
    const GsfPropagatorAdapter *geomPropFw_;

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

    unsigned long long cacheIDGeom_;
    unsigned long long cacheIDTopo_;
    unsigned long long cacheIDTDGeom_;
    unsigned long long cacheIDMagField_;
 } ;

#endif // GsfElectronAlgo_H


