
#ifndef GsfElectronAlgo_H
#define GsfElectronAlgo_H

class MultiTrajectoryStateTransform ;
class MultiTrajectoryStateMode ;
class EcalClusterFunctionBaseClass ;

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/RegressionHelper.h"
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


#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "RecoEgamma/ElectronIdentification/interface/SoftElectronMVAEstimator.h"


#include <list>
#include <string>


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
      double maxHOverEBarrel ;
      double maxHOverEEndcaps ;
      double maxHBarrel ;
      double maxHEndcaps ;

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
      double intRadiusBarrelTk ;
      double intRadiusEndcapTk ;
      double stripBarrelTk ;
      double stripEndcapTk ;
      double ptMinTk ;
      double maxVtxDistTk ;
      double maxDrbTk ;
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
      const SoftElectronMVAEstimator::Configuration & mvaCfg,	
      const RegressionHelper::Configuration & regCfg
      ) ;

    ~GsfElectronAlgo() ;

    // typedefs
    typedef std::list<reco::GsfElectron *> GsfElectronPtrCollection ; // for temporary collections

    // main methods
    void checkSetup( const edm::EventSetup & ) ;
    void beginEvent( edm::Event & ) ;
    void displayInternalElectrons( const std::string & title ) const ;
    void clonePreviousElectrons() ;
    void completeElectrons() ; // do not redo cloned electrons done previously
    void addPflowInfo() ; // now deprecated
    void setAmbiguityData( bool ignoreNotPreselected = true ) ;
    void removeNotPreselectedElectrons() ;
    void removeAmbiguousElectrons() ;
    void copyElectrons( reco::GsfElectronCollection & ) ;
    void setMVAInputs(const std::map<reco::GsfTrackRef,reco::GsfElectron::MvaInput> & mvaInputs)  ;
    void setMVAOutputs(const std::map<reco::GsfTrackRef,reco::GsfElectron::MvaOutput> & mvaOutputs) ;
    void endEvent() ;

  private :

    // internal structures
    struct GeneralData ;
    struct EventSetupData ;
    struct EventData ;
    struct ElectronData ;
    GeneralData * generalData_ ;
    EventSetupData * eventSetupData_ ;
    EventData * eventData_ ;
    ElectronData * electronData_ ;

    void createElectron() ;

    void setMVAepiBasedPreselectionFlag(reco::GsfElectron * ele);
    void setCutBasedPreselectionFlag( reco::GsfElectron * ele, const reco::BeamSpot & ) ;
    void setPflowPreselectionFlag( reco::GsfElectron * ele ) ;
    bool isPreselected( reco::GsfElectron * ele ) ;
    void calculateShowerShape( const reco::SuperClusterRef &, bool pflow, reco::GsfElectron::ShowerShape & ) ;


    // associations
    const reco::SuperClusterRef getTrSuperCluster( const reco::GsfTrackRef & trackRef ) ;
 } ;

#endif // GsfElectronAlgo_H


