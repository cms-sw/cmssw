#ifndef GsfElectronAlgo_H
#define GsfElectronAlgo_H

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/RegressionHelper.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"

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
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "RecoEgamma/ElectronIdentification/interface/SoftElectronMVAEstimator.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"

#include <list>
#include <string>

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgoHeavyObjectCache.h"

class GsfElectronAlgo {

  public:

    struct InputTagsConfiguration
     {
       edm::EDGetTokenT<reco::GsfElectronCollection> previousGsfElectrons ;
       edm::EDGetTokenT<reco::GsfElectronCollection> pflowGsfElectronsTag ;
       edm::EDGetTokenT<reco::GsfElectronCoreCollection> gsfElectronCores ;
       edm::EDGetTokenT<CaloTowerCollection> hcalTowersTag ;
       edm::EDGetTokenT<reco::SuperClusterCollection> barrelSuperClusters ;
       edm::EDGetTokenT<reco::SuperClusterCollection> endcapSuperClusters ;
      //edm::EDGetTokenT tracks ;
       edm::EDGetTokenT<EcalRecHitCollection> barrelRecHitCollection ;
       edm::EDGetTokenT<EcalRecHitCollection> endcapRecHitCollection ;
       edm::EDGetTokenT<edm::ValueMap<float> > pfMVA ;
       edm::EDGetTokenT<reco::ElectronSeedCollection> seedsTag ;
       edm::EDGetTokenT<reco::TrackCollection> ctfTracks ;
       edm::EDGetTokenT<reco::BeamSpot> beamSpotTag ;
       edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracksTag ;
       edm::EDGetTokenT<reco::VertexCollection> vtxCollectionTag;

      //IsoVals (PF and EcalDriven)
      edm::ParameterSet pfIsoVals;
      edm::ParameterSet edIsoVals;

     } ;

    struct StrategyConfiguration
     {
      bool useGsfPfRecTracks ;
      // if true, electron preselection is applied
      bool applyPreselection ;
      // if true, electron level escale corrections are
      // used on top of the cluster level corrections
      bool ecalDrivenEcalEnergyFromClassBasedParameterization ;
      bool ecalDrivenEcalErrorFromClassBasedParameterization ;
      bool pureTrackerDrivenEcalErrorFromSimpleParameterization ;
      // ambiguity solving
      bool applyAmbResolution  ; // if not true, ambiguity solving is not applied
      unsigned ambSortingStrategy  ; // 0:isBetter, 1:isInnerMost
      unsigned ambClustersOverlapStrategy  ; // 0:sc adresses, 1:bc shared energy
      // if true, trackerDriven electrons are added
      bool addPflowElectrons ;
      // for backward compatibility
      bool ctfTracksCheck ;
      bool gedElectronMode;
      float PreSelectMVA;	
      float MaxElePtForOnlyMVA;
      // GED-Regression (ECAL and combination)
      bool useEcalRegression;
      bool useCombinationRegression;  
     } ;

    struct CutsConfiguration
     {
      // minimum SC Et
      double minSCEtBarrel ;
      double minSCEtEndcaps ;
      // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
      double maxEOverPBarrel ;
      double maxEOverPEndcaps ;
      // minimum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
      double minEOverPBarrel ;
      double minEOverPEndcaps ;

      // H/E
      double maxHOverEBarrelCone ;
      double maxHOverEEndcapsCone ;
      double maxHBarrelCone ;
      double maxHEndcapsCone ;
      double maxHOverEBarrelTower ;
      double maxHOverEEndcapsTower ;
      double maxHBarrelTower ;
      double maxHEndcapsTower ;

      // maximum eta difference between the supercluster position and the track position at the closest impact to the supercluster
      double maxDeltaEtaBarrel ;
      double maxDeltaEtaEndcaps ;

      // maximum phi difference between the supercluster position and the track position at the closest impact to the supercluster
      // position to the supercluster
      double maxDeltaPhiBarrel ;
      double maxDeltaPhiEndcaps ;

      // maximum sigma ieta ieta
      double maxSigmaIetaIetaBarrel ;
      double maxSigmaIetaIetaEndcaps ;
      // maximum fbrem

      double maxFbremBarrel ;
      double maxFbremEndcaps ;

      // fiducial regions
      bool isBarrel ;
      bool isEndcaps ;
      bool isFiducial ;

      // BDT output (if available)
      double minMVA ;
      double minMvaByPassForIsolated ;

      // transverse impact parameter wrt beam spot
      double maxTIP ;

      // only make sense for ecal driven electrons
      bool seedFromTEC ;
     } ;

    // Ecal rec hits
    struct EcalRecHitsConfiguration
     {
      std::vector<int> recHitFlagsToBeExcludedBarrel ;
      std::vector<int> recHitFlagsToBeExcludedEndcaps ;
      std::vector<int> recHitSeverityToBeExcludedBarrel ;
      std::vector<int> recHitSeverityToBeExcludedEndcaps ;
      //int severityLevelCut ;
     } ;

    // isolation variables parameters
    struct IsolationConfiguration
     {
      double intRadiusHcal ;
      double etMinHcal ;
      double intRadiusEcalBarrel ;
      double intRadiusEcalEndcaps ;
      double jurassicWidth ;
      double etMinBarrel ;
      double eMinBarrel ;
      double etMinEndcaps ;
      double eMinEndcaps ;
      bool vetoClustered ;
      bool useNumCrystals ;
     } ;

    GsfElectronAlgo
     (
      const InputTagsConfiguration &,
      const StrategyConfiguration &,
      const CutsConfiguration & cutsCfg,
      const CutsConfiguration & cutsCfgPflow,
      const ElectronHcalHelper::Configuration & hcalCfg,
      const ElectronHcalHelper::Configuration & hcalCfgPflow,
      const IsolationConfiguration &,
      const EcalRecHitsConfiguration &,
      EcalClusterFunctionBaseClass * superClusterErrorFunction,
      EcalClusterFunctionBaseClass * crackCorrectionFunction,
      const RegressionHelper::Configuration & regCfg,
      const edm::ParameterSet& tkIsol03Cfg,
      const edm::ParameterSet& tkIsol04Cfg
      
      ) ;

    // typedefs
    typedef std::list<reco::GsfElectron> GsfElectronList ; // for temporary collections

    // main methods
    void checkSetup( const edm::EventSetup & ) ;
    void beginEvent( edm::Event & ) ;
    void displayInternalElectrons( const std::string & title ) const ;
    void clonePreviousElectrons() ;
    void completeElectrons(const gsfAlgoHelpers::HeavyObjectCache*) ; // do not redo cloned electrons done previously
    void addPflowInfo() ; // now deprecated
    void setAmbiguityData( bool ignoreNotPreselected = true ) ;
    void removeNotPreselectedElectrons() ;
    void removeAmbiguousElectrons() ;
    void moveElectrons( reco::GsfElectronCollection & ) ;
    void setMVAInputs(const std::map<reco::GsfTrackRef,reco::GsfElectron::MvaInput> & mvaInputs)  ;
    void setMVAOutputs(const gsfAlgoHelpers::HeavyObjectCache*,
                       const std::map<reco::GsfTrackRef,reco::GsfElectron::MvaOutput> & mvaOutputs) ;
    void endEvent() ;

  private :

    // internal structures

    //===================================================================
    // GsfElectronAlgo::GeneralData
    //===================================================================

    // general data and helpers
    struct GeneralData
     {
      // configurables
      const InputTagsConfiguration inputCfg ;
      const StrategyConfiguration strategyCfg ;
      const CutsConfiguration cutsCfg ;
      const CutsConfiguration cutsCfgPflow ;
      const IsolationConfiguration isoCfg ;
      const EcalRecHitsConfiguration recHitsCfg ;

      // additional configuration and helpers
      ElectronHcalHelper hcalHelper;
      ElectronHcalHelper hcalHelperPflow ;
      EcalClusterFunctionBaseClass * superClusterErrorFunction ;
      EcalClusterFunctionBaseClass * crackCorrectionFunction ;
      RegressionHelper regHelper;
     } ;

    //===================================================================
    // GsfElectronAlgo::EventSetupData
    //===================================================================

    struct EventSetupData
     {
       EventSetupData() ;

       unsigned long long cacheIDGeom ;
       unsigned long long cacheIDTopo ;
       unsigned long long cacheIDTDGeom ;
       unsigned long long cacheIDMagField ;
       unsigned long long cacheSevLevel ;

       edm::ESHandle<MagneticField> magField ;
       edm::ESHandle<CaloGeometry> caloGeom ;
       edm::ESHandle<CaloTopology> caloTopo ;
       edm::ESHandle<TrackerGeometry> trackerHandle ;
       edm::ESHandle<EcalSeverityLevelAlgo> sevLevel;

       std::unique_ptr<const MultiTrajectoryStateTransform> mtsTransform ;
       std::unique_ptr<GsfConstraintAtVertex> constraintAtVtx ;
       const MultiTrajectoryStateMode mtsMode ;
    } ;


    //===================================================================
    // GsfElectronAlgo::EventData
    //===================================================================

    struct EventData
     {
      // general
      edm::Event * event ;
      const reco::BeamSpot * beamspot ;
      GsfElectronList electrons ;

      EventData() ;

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
      std::unique_ptr<EgammaTowerIsolation> hadDepth1Isolation03, hadDepth1Isolation04 ;
      std::unique_ptr<EgammaTowerIsolation> hadDepth2Isolation03, hadDepth2Isolation04 ;
      std::unique_ptr<EgammaTowerIsolation> hadDepth1Isolation03Bc, hadDepth1Isolation04Bc ;
      std::unique_ptr<EgammaTowerIsolation> hadDepth2Isolation03Bc, hadDepth2Isolation04Bc ;
      std::unique_ptr<EgammaRecHitIsolation> ecalBarrelIsol03, ecalBarrelIsol04 ;
      std::unique_ptr<EgammaRecHitIsolation> ecalEndcapIsol03, ecalEndcapIsol04 ;

      //Isolation Value Maps for PF and EcalDriven electrons
      typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsolationValueMaps;
      IsolationValueMaps pfIsolationValues;
      IsolationValueMaps edIsolationValues;
     } ;

    //===================================================================
    // GsfElectronAlgo::ElectronData
    //===================================================================

    struct ElectronData
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

      // utilities
      void checkCtfTrack( edm::Handle<reco::TrackCollection> currentCtfTracks ) ;
      void computeCharge( int & charge, reco::GsfElectron::ChargeInfo & info ) ;
      reco::CaloClusterPtr getEleBasicCluster( MultiTrajectoryStateTransform const& ) ;
      bool calculateTSOS( MultiTrajectoryStateTransform const&, GsfConstraintAtVertex const& ) ;
      void calculateMode( MultiTrajectoryStateMode const& mtsMode ) ;
      reco::Candidate::LorentzVector calculateMomentum() ;

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

    std::unique_ptr<GeneralData> generalData_ ;
    std::unique_ptr<EventSetupData> eventSetupData_ ;
    std::unique_ptr<EventData> eventData_ ;
    std::unique_ptr<ElectronData> electronData_ ;

    EleTkIsolFromCands tkIsol03Calc_;
    EleTkIsolFromCands tkIsol04Calc_;

    void createElectron(const gsfAlgoHelpers::HeavyObjectCache*) ;

    void setMVAepiBasedPreselectionFlag(reco::GsfElectron & ele);
    void setCutBasedPreselectionFlag( reco::GsfElectron & ele, const reco::BeamSpot & ) ;
    void setPflowPreselectionFlag( reco::GsfElectron & ele ) ;
    bool isPreselected( reco::GsfElectron & ele ) ;
    void calculateShowerShape( const reco::SuperClusterRef &, bool pflow, 
                               reco::GsfElectron::ShowerShape & ) ;
    void calculateShowerShape_full5x5( const reco::SuperClusterRef &, bool pflow,
                                       reco::GsfElectron::ShowerShape & ) ;
    void calculateSaturationInfo(const reco::SuperClusterRef&, reco::GsfElectron::SaturationInfo&);

    // associations
    const reco::SuperClusterRef getTrSuperCluster( const reco::GsfTrackRef & trackRef ) ;
    
    // Pixel match variables
    void setPixelMatchInfomation(reco::GsfElectron &) ;
    
 } ;

#endif // GsfElectronAlgo_H
