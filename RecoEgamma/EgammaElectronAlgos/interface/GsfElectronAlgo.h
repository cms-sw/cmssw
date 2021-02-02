#ifndef GsfElectronAlgo_H
#define GsfElectronAlgo_H

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/RegressionHelper.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "RecoEgamma/ElectronIdentification/interface/SoftElectronMVAEstimator.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTracking/interface/GsfConstraintAtVertex.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ConversionFinder.h"

class GsfElectronAlgo {
public:
  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet&);
    std::unique_ptr<const SoftElectronMVAEstimator> sElectronMVAEstimator;
    std::unique_ptr<const ElectronMVAEstimator> iElectronMVAEstimator;
  };

  struct Tokens {
    edm::EDGetTokenT<reco::GsfElectronCoreCollection> gsfElectronCores;
    edm::EDGetTokenT<CaloTowerCollection> hcalTowersTag;
    edm::EDGetTokenT<reco::SuperClusterCollection> barrelSuperClusters;
    edm::EDGetTokenT<reco::SuperClusterCollection> endcapSuperClusters;
    edm::EDGetTokenT<EcalRecHitCollection> barrelRecHitCollection;
    edm::EDGetTokenT<EcalRecHitCollection> endcapRecHitCollection;
    edm::EDGetTokenT<reco::ElectronSeedCollection> seedsTag;
    edm::EDGetTokenT<reco::TrackCollection> ctfTracks;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotTag;
    edm::EDGetTokenT<reco::VertexCollection> vtxCollectionTag;
    edm::EDGetTokenT<reco::ConversionCollection> conversions;
  };

  struct StrategyConfiguration {
    // if true, electron preselection is applied
    bool applyPreselection;
    // if true, electron level escale corrections are
    // used on top of the cluster level corrections
    bool ecalDrivenEcalEnergyFromClassBasedParameterization;
    bool ecalDrivenEcalErrorFromClassBasedParameterization;
    bool pureTrackerDrivenEcalErrorFromSimpleParameterization;
    // ambiguity solving
    bool applyAmbResolution;  // if not true, ambiguity solving is not applied
    bool ignoreNotPreselected;
    unsigned ambSortingStrategy;          // 0:isBetter, 1:isInnermost
    unsigned ambClustersOverlapStrategy;  // 0:sc adresses, 1:bc shared energy
    // for backward compatibility
    bool ctfTracksCheck;
    float PreSelectMVA;
    float MaxElePtForOnlyMVA;
    bool useDefaultEnergyCorrection;
    // GED-Regression (ECAL and combination)
    bool useEcalRegression;
    bool useCombinationRegression;
    //heavy ion in 2015 has no conversions and so cant fill conv vtx fit prob so this bool
    //stops it from being filled
    bool fillConvVtxFitProb;
  };

  struct CutsConfiguration {
    // minimum SC Et
    double minSCEtBarrel;
    double minSCEtEndcaps;
    // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
    double maxEOverPBarrel;
    double maxEOverPEndcaps;
    // minimum E/p where E is the supercluster corrected energy and p the track momentum at innermost state
    double minEOverPBarrel;
    double minEOverPEndcaps;

    // H/E
    double maxHOverEBarrelCone;
    double maxHOverEEndcapsCone;
    double maxHBarrelCone;
    double maxHEndcapsCone;
    double maxHOverEBarrelTower;
    double maxHOverEEndcapsTower;
    double maxHBarrelTower;
    double maxHEndcapsTower;

    // maximum eta difference between the supercluster position and the track position at the closest impact to the supercluster
    double maxDeltaEtaBarrel;
    double maxDeltaEtaEndcaps;

    // maximum phi difference between the supercluster position and the track position at the closest impact to the supercluster
    // position to the supercluster
    double maxDeltaPhiBarrel;
    double maxDeltaPhiEndcaps;

    // maximum sigma ieta ieta
    double maxSigmaIetaIetaBarrel;
    double maxSigmaIetaIetaEndcaps;
    // maximum fbrem

    double maxFbremBarrel;
    double maxFbremEndcaps;

    // fiducial regions
    bool isBarrel;
    bool isEndcaps;
    bool isFiducial;

    // transverse impact parameter wrt beam spot
    double maxTIP;

    // only make sense for ecal driven electrons
    bool seedFromTEC;
  };

  // Ecal rec hits
  struct EcalRecHitsConfiguration {
    std::vector<int> recHitFlagsToBeExcludedBarrel;
    std::vector<int> recHitFlagsToBeExcludedEndcaps;
    std::vector<int> recHitSeverityToBeExcludedBarrel;
    std::vector<int> recHitSeverityToBeExcludedEndcaps;
    //int severityLevelCut ;
  };

  // isolation variables parameters
  struct IsolationConfiguration {
    double intRadiusHcal;
    double etMinHcal;
    double intRadiusEcalBarrel;
    double intRadiusEcalEndcaps;
    double jurassicWidth;
    double etMinBarrel;
    double eMinBarrel;
    double etMinEndcaps;
    double eMinEndcaps;
    bool vetoClustered;
    bool useNumCrystals;
  };

  GsfElectronAlgo(const Tokens&,
                  const StrategyConfiguration&,
                  const CutsConfiguration& cutsCfg,
                  const ElectronHcalHelper::Configuration& hcalCfg,
                  const IsolationConfiguration&,
                  const EcalRecHitsConfiguration&,
                  std::unique_ptr<EcalClusterFunctionBaseClass>&& crackCorrectionFunction,
                  const RegressionHelper::Configuration& regCfg,
                  const edm::ParameterSet& tkIsol03Cfg,
                  const edm::ParameterSet& tkIsol04Cfg,
                  const edm::ParameterSet& tkIsolHEEP03Cfg,
                  const edm::ParameterSet& tkIsolHEEP04Cfg,
                  edm::ConsumesCollector&& cc);

  // main methods
  reco::GsfElectronCollection completeElectrons(edm::Event const& event,
                                                edm::EventSetup const& eventSetup,
                                                const HeavyObjectCache* hoc);

private:
  // internal structures

  struct Configuration {
    // configurables
    const Tokens tokens;
    const StrategyConfiguration strategy;
    const CutsConfiguration cuts;
    const IsolationConfiguration iso;
    const EcalRecHitsConfiguration recHits;
  };

  struct EventData;
  struct ElectronData;

  void checkSetup(edm::EventSetup const& eventSetup);
  EventData beginEvent(edm::Event const& event,
                       CaloGeometry const& caloGeometry,
                       EcalSeverityLevelAlgo const& ecalSeveretyLevelAlgo);

  void createElectron(reco::GsfElectronCollection& electrons,
                      ElectronData& electronData,
                      EventData& eventData,
                      CaloTopology const& topology,
                      CaloGeometry const& geometry,
                      MultiTrajectoryStateTransform const& mtsTransform,
                      double magneticFieldInTesla,
                      const HeavyObjectCache*,
                      egamma::conv::TrackTableView ctfTable,
                      egamma::conv::TrackTableView gsfTable);

  void setCutBasedPreselectionFlag(reco::GsfElectron& ele, const reco::BeamSpot&) const;

  template <bool full5x5>
  reco::GsfElectron::ShowerShape calculateShowerShape(const reco::SuperClusterRef&,
                                                      ElectronHcalHelper const& hcalHelper,
                                                      EventData const& eventData,
                                                      CaloTopology const& topology,
                                                      CaloGeometry const& geometry) const;
  reco::GsfElectron::SaturationInfo calculateSaturationInfo(const reco::SuperClusterRef&,
                                                            EventData const& eventData) const;

  // Pixel match variables
  void setPixelMatchInfomation(reco::GsfElectron&) const;

  // constant class members
  const Configuration cfg_;

  const EleTkIsolFromCands::Configuration tkIsol03CalcCfg_;
  const EleTkIsolFromCands::Configuration tkIsol04CalcCfg_;
  const EleTkIsolFromCands::Configuration tkIsolHEEP03CalcCfg_;
  const EleTkIsolFromCands::Configuration tkIsolHEEP04CalcCfg_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  const edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> ecalSeveretyLevelAlgoToken_;

  // additional configuration and helpers
  ElectronHcalHelper hcalHelper_;
  std::unique_ptr<EcalClusterFunctionBaseClass> crackCorrectionFunction_;
  RegressionHelper regHelper_;
};

#endif  // GsfElectronAlgo_H
