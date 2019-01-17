#include "GsfElectronBaseProducer.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"

#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"


#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalSeverityLevel.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

using namespace reco;

void GsfElectronBaseProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  edm::ParameterSetDescription desc ;
  // input collections
  desc.add<edm::InputTag>("gsfElectronCoresTag", edm::InputTag("gedGsfElectronCores"));
  desc.add<edm::InputTag>("pflowGsfElectronsTag", edm::InputTag(""));
  desc.add<edm::InputTag>("pfMvaTag", edm::InputTag(""));
  desc.add<edm::InputTag>("previousGsfElectronsTag", edm::InputTag(""));
  desc.add<edm::InputTag>("hcalTowers", edm::InputTag("towerMaker"));
  desc.add<edm::InputTag>("vtxTag", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("gsfPfRecTracksTag", edm::InputTag("pfTrackElec"));
  desc.add<edm::InputTag>("barrelRecHitCollectionTag", edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>("endcapRecHitCollectionTag", edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  desc.add<edm::InputTag>("seedsTag", edm::InputTag("ecalDrivenElectronSeeds"));
  desc.add<edm::InputTag>("beamSpotTag", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("egmPFCandidatesTag", edm::InputTag("particleFlowEGamma"));
  desc.add<bool>("checkHcalStatus", true);

  // steering
  desc.add<bool>("gedElectronMode", true);
  desc.add<bool>("useCombinationRegression", true);
  desc.add<bool>("ecalDrivenEcalEnergyFromClassBasedParameterization", false);
  desc.add<bool>("ecalDrivenEcalErrorFromClassBasedParameterization", false);
  desc.add<bool>("applyPreselection", true);
  desc.add<bool>("useEcalRegression", true);
  desc.add<bool>("applyAmbResolution", false);
  desc.add<bool>("useGsfPfRecTracks", true);
  desc.add<bool>("pureTrackerDrivenEcalErrorFromSimpleParameterization", true);
  desc.add<unsigned int>("ambSortingStrategy", 1);
  desc.add<unsigned int>("ambClustersOverlapStrategy", 1);
  desc.add<bool>("addPflowElectrons", true); // this one should be transfered to the "core" level

  // Ecal rec hits configuration
  desc.add<std::vector<std::string>>("recHitFlagsToBeExcludedBarrel");
  desc.add<std::vector<std::string>>("recHitFlagsToBeExcludedEndcaps");
  desc.add<std::vector<std::string>>("recHitSeverityToBeExcludedBarrel");
  desc.add<std::vector<std::string>>("recHitSeverityToBeExcludedEndcaps");

  // Isolation algos configuration
  desc.add("trkIsol03Cfg",EleTkIsolFromCands::pSetDescript());
  desc.add("trkIsol04Cfg",EleTkIsolFromCands::pSetDescript());
  desc.add<bool>("useNumCrystals", true);
  desc.add<double>("etMinBarrel", 0.0);
  desc.add<double>("etMinEndcaps", 0.11);
  desc.add<double>("etMinHcal", 0.0);
  desc.add<double>("eMinBarrel", 0.095);
  desc.add<double>("eMinEndcaps", 0.0);
  desc.add<double>("intRadiusEcalBarrel", 3.0);
  desc.add<double>("intRadiusEcalEndcaps", 3.0);
  desc.add<double>("intRadiusHcal", 0.15);
  desc.add<double>("jurassicWidth", 1.5);
  desc.add<bool>("vetoClustered", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("propagatorAlongTISE", "PropagatorWithMaterial");
    psd0.add<std::string>("propagatorOppositeTISE", "PropagatorWithMaterialOpposite");
    desc.add<edm::ParameterSetDescription>("TransientInitialStateEstimatorParameters", psd0);
  }

  // backward compatibility mechanism for ctf tracks
  desc.add<bool>("ctfTracksCheck", true);
  desc.add<edm::InputTag>("ctfTracksTag", edm::InputTag("generalTracks"));

  desc.add<double>("MaxElePtForOnlyMVA",50.0);
  desc.add<double>("PreSelectMVA", -0.1);

  {
    edm::ParameterSetDescription psd0;
    psd0.add<double>("minSCEtBarrel", 0.0);
    psd0.add<double>("minSCEtEndcaps", 0.0);
    psd0.add<double>("minEOverPBarrel", 0.0);
    psd0.add<double>("minEOverPEndcaps", 0.0);
    psd0.add<double>("maxEOverPBarrel", 999999999.0);
    psd0.add<double>("maxEOverPEndcaps", 999999999.0);
    psd0.add<double>("maxDeltaEtaBarrel", 999999999.0);
    psd0.add<double>("maxDeltaEtaEndcaps", 999999999.0);
    psd0.add<double>("maxDeltaPhiBarrel", 999999999.0);
    psd0.add<double>("maxDeltaPhiEndcaps", 999999999.0);
    psd0.add<double>("hOverEConeSize", 0.15);
    psd0.add<double>("hOverEPtMin", 0.0);
    psd0.add<double>("maxHOverEBarrel", 999999999.0);
    psd0.add<double>("maxHOverEEndcaps", 999999999.0);
    psd0.add<double>("maxHBarrel", 0.0);
    psd0.add<double>("maxHEndcaps", 0.0);
    psd0.add<double>("maxSigmaIetaIetaBarrel", 999999999.0);
    psd0.add<double>("maxSigmaIetaIetaEndcaps", 999999999.0);
    psd0.add<double>("maxFbremBarrel", 999999999.0);
    psd0.add<double>("maxFbremEndcaps", 999999999.0);
    psd0.add<bool>("isBarrel", false);
    psd0.add<bool>("isEndcaps", false);
    psd0.add<bool>("isFiducial", false);
    psd0.add<bool>("seedFromTEC", true);
    psd0.add<double>("maxTIP", 999999999.0);
    psd0.add<double>("minMVA", -0.4);
    psd0.add<double>("minMvaByPassForIsolated", -0.4);
    // preselection parameters (ecal driven electrons)
    desc.add<edm::ParameterSetDescription>("preselection", psd0);
    // preselection parameters (tracker driven only electrons)
    desc.add<edm::ParameterSetDescription>("preselectionPflow", psd0);
  }

  // Corrections
  desc.add<std::string>("superClusterErrorFunction", "EcalClusterEnergyUncertaintyObjectSpecific");
  desc.add<std::string>("crackCorrectionFunction", "EcalClusterCrackCorrection");

  desc.add<bool>("ecalWeightsFromDB", true);
  desc.add<std::vector<std::string>>("ecalRefinedRegressionWeightFiles", {})->setComment(
          "if not from DB. Otherwise, keep empty");
  desc.add<bool>("combinationWeightsFromDB", true);
  desc.add<std::vector<std::string>>("combinationRegressionWeightFile", {})->setComment(
          "if not from DB. Otherwise, keep empty");

  // regression. The labels are needed in all cases.
  desc.add<std::vector<std::string>>("ecalRefinedRegressionWeightLabels", {});
  desc.add<std::vector<std::string>>("combinationRegressionWeightLabels", {});

  // Iso values
  desc.add<bool>("useIsolationValues", false);

  {
    edm::ParameterSetDescription psd0;
    psd0.add<edm::InputTag>("edSumPhotonEt", edm::InputTag("elEDIsoValueGamma04"));
    psd0.add<edm::InputTag>("edSumNeutralHadronEt", edm::InputTag("elEDIsoValueNeutral04"));
    psd0.add<edm::InputTag>("edSumChargedHadronPt", edm::InputTag("elEDIsoValueCharged04"));
    desc.add<edm::ParameterSetDescription>("edIsolationValues", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<edm::InputTag>("pfSumNeutralHadronEt", edm::InputTag("elPFIsoValueNeutral04"));
    psd0.add<edm::InputTag>("pfSumChargedHadronPt", edm::InputTag("elPFIsoValueCharged04"));
    psd0.add<edm::InputTag>("pfSumPhotonEt", edm::InputTag("elPFIsoValueGamma04"));
    desc.add<edm::ParameterSetDescription>("pfIsolationValues", psd0);
  }

  desc.add<std::vector<std::string>>("ElecMVAFilesString", {
    "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_10_17Feb2011.weights.xml",
    "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_12_17Feb2011.weights.xml",
    "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_20_17Feb2011.weights.xml",
    "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_22_17Feb2011.weights.xml",
  });
  desc.add<std::vector<std::string>>("SoftElecMVAFilesString", {
         "RecoEgamma/ElectronIdentification/data/TMVA_BDTSoftElectrons_7Feb2014.weights.xml",
  });

  descriptions.addDefault(desc);
}

namespace {
  GsfElectronAlgo::CutsConfiguration makeCutsConfiguration(edm::ParameterSet const& pset)
  {
    return GsfElectronAlgo::CutsConfiguration{
        .minSCEtBarrel = pset.getParameter<double>("minSCEtBarrel"),
        .minSCEtEndcaps = pset.getParameter<double>("minSCEtEndcaps"),
        .maxEOverPBarrel = pset.getParameter<double>("maxEOverPBarrel"),
        .maxEOverPEndcaps = pset.getParameter<double>("maxEOverPEndcaps"),
        .minEOverPBarrel = pset.getParameter<double>("minEOverPBarrel"),
        .minEOverPEndcaps = pset.getParameter<double>("minEOverPEndcaps"),
        .maxHOverEBarrel = pset.getParameter<double>("maxHOverEBarrel"),
        .maxHOverEEndcaps = pset.getParameter<double>("maxHOverEEndcaps"),
        .maxHBarrel = pset.getParameter<double>("maxHBarrel"),
        .maxHEndcaps = pset.getParameter<double>("maxHEndcaps"),
        .maxDeltaEtaBarrel = pset.getParameter<double>("maxDeltaEtaBarrel"),
        .maxDeltaEtaEndcaps = pset.getParameter<double>("maxDeltaEtaEndcaps"),
        .maxDeltaPhiBarrel = pset.getParameter<double>("maxDeltaPhiBarrel"),
        .maxDeltaPhiEndcaps = pset.getParameter<double>("maxDeltaPhiEndcaps"),
        .maxSigmaIetaIetaBarrel = pset.getParameter<double>("maxSigmaIetaIetaBarrel"),
        .maxSigmaIetaIetaEndcaps = pset.getParameter<double>("maxSigmaIetaIetaEndcaps"),
        .maxFbremBarrel = pset.getParameter<double>("maxFbremBarrel"),
        .maxFbremEndcaps = pset.getParameter<double>("maxFbremEndcaps"),
        .isBarrel = pset.getParameter<bool>("isBarrel"),
        .isEndcaps = pset.getParameter<bool>("isEndcaps"),
        .isFiducial = pset.getParameter<bool>("isFiducial"),
        .minMVA = pset.getParameter<double>("minMVA"),
        .minMvaByPassForIsolated = pset.getParameter<double>("minMvaByPassForIsolated"),
        .maxTIP = pset.getParameter<double>("maxTIP"),
        .seedFromTEC = pset.getParameter<bool>("seedFromTEC"),
    };
  }
};

GsfElectronBaseProducer::GsfElectronBaseProducer( const edm::ParameterSet& cfg, const gsfAlgoHelpers::HeavyObjectCache* )
 : cutsCfg_(makeCutsConfiguration(cfg.getParameter<edm::ParameterSet>("preselection")))
 , cutsCfgPflow_(makeCutsConfiguration(cfg.getParameter<edm::ParameterSet>("preselectionPflow")))
 , ecalSeedingParametersChecked_(false)
{
  produces<GsfElectronCollection>();

  inputCfg_.previousGsfElectrons = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("previousGsfElectronsTag"));
  inputCfg_.pflowGsfElectronsTag = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("pflowGsfElectronsTag"));
  inputCfg_.gsfElectronCores = consumes<reco::GsfElectronCoreCollection>(cfg.getParameter<edm::InputTag>("gsfElectronCoresTag"));
  inputCfg_.hcalTowersTag = consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers"));
  inputCfg_.barrelRecHitCollection = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("barrelRecHitCollectionTag"));
  inputCfg_.endcapRecHitCollection = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("endcapRecHitCollectionTag"));
  inputCfg_.pfMVA = consumes<edm::ValueMap<float> >(cfg.getParameter<edm::InputTag>("pfMvaTag"));
  inputCfg_.ctfTracks = consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("ctfTracksTag"));
  inputCfg_.seedsTag = consumes<reco::ElectronSeedCollection>(cfg.getParameter<edm::InputTag>("seedsTag")); // used to check config consistency with seeding
  inputCfg_.beamSpotTag = consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpotTag"));
  inputCfg_.gsfPfRecTracksTag = consumes<reco::GsfPFRecTrackCollection>(cfg.getParameter<edm::InputTag>("gsfPfRecTracksTag"));
  inputCfg_.vtxCollectionTag = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vtxTag"));

  if ( cfg.getParameter<bool>("useIsolationValues") ) {
    inputCfg_.pfIsoVals = cfg.getParameter<edm::ParameterSet> ("pfIsolationValues");
    for(const std::string& name : inputCfg_.pfIsoVals.getParameterNamesForType<edm::InputTag>()) {
      edm::InputTag tag = inputCfg_.pfIsoVals.getParameter<edm::InputTag>(name);
      mayConsume<edm::ValueMap<double> >(tag);
    }

    inputCfg_.edIsoVals = cfg.getParameter<edm::ParameterSet> ("edIsolationValues");
    for(const std::string& name : inputCfg_.edIsoVals.getParameterNamesForType<edm::InputTag>()) {
      edm::InputTag tag = inputCfg_.edIsoVals.getParameter<edm::InputTag>(name);
      mayConsume<edm::ValueMap<double> >(tag);
    }
  }

  strategyCfg_.useGsfPfRecTracks = cfg.getParameter<bool>("useGsfPfRecTracks") ;
  strategyCfg_.applyPreselection = cfg.getParameter<bool>("applyPreselection") ;
  strategyCfg_.ecalDrivenEcalEnergyFromClassBasedParameterization = cfg.getParameter<bool>("ecalDrivenEcalEnergyFromClassBasedParameterization") ;
  strategyCfg_.ecalDrivenEcalErrorFromClassBasedParameterization = cfg.getParameter<bool>("ecalDrivenEcalErrorFromClassBasedParameterization") ;
  strategyCfg_.pureTrackerDrivenEcalErrorFromSimpleParameterization = cfg.getParameter<bool>("pureTrackerDrivenEcalErrorFromSimpleParameterization") ;
  strategyCfg_.applyAmbResolution = cfg.getParameter<bool>("applyAmbResolution") ;
  strategyCfg_.ambSortingStrategy = cfg.getParameter<unsigned>("ambSortingStrategy") ;
  strategyCfg_.ambClustersOverlapStrategy = cfg.getParameter<unsigned>("ambClustersOverlapStrategy") ;
  strategyCfg_.addPflowElectrons = cfg.getParameter<bool>("addPflowElectrons") ;
  strategyCfg_.ctfTracksCheck = cfg.getParameter<bool>("ctfTracksCheck");
  strategyCfg_.gedElectronMode = cfg.getParameter<bool>("gedElectronMode");
  strategyCfg_.PreSelectMVA = cfg.getParameter<double>("PreSelectMVA");
  strategyCfg_.MaxElePtForOnlyMVA = cfg.getParameter<double>("MaxElePtForOnlyMVA");
  strategyCfg_.useEcalRegression = cfg.getParameter<bool>("useEcalRegression");
  strategyCfg_.useCombinationRegression = cfg.getParameter<bool>("useCombinationRegression");

  // hcal helpers
  auto const& psetPreselection = cfg.getParameter<edm::ParameterSet>("preselection");
  hcalCfg_.hOverEConeSize = psetPreselection.getParameter<double>("hOverEConeSize") ;
  if (hcalCfg_.hOverEConeSize>0)
   {
    hcalCfg_.useTowers = true ;
    hcalCfg_.checkHcalStatus = cfg.getParameter<bool>("checkHcalStatus") ;
    hcalCfg_.hcalTowers = consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers")) ;
    hcalCfg_.hOverEPtMin = psetPreselection.getParameter<double>("hOverEPtMin") ;
   }
  auto const& psetPreselectionPflow = cfg.getParameter<edm::ParameterSet>("preselectionPflow");
  hcalCfgPflow_.hOverEConeSize = psetPreselectionPflow.getParameter<double>("hOverEConeSize") ;
  if (hcalCfgPflow_.hOverEConeSize>0)
   {
    hcalCfgPflow_.useTowers = true ;
    hcalCfgPflow_.checkHcalStatus = cfg.getParameter<bool>("checkHcalStatus") ;
    hcalCfgPflow_.hcalTowers = consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers")) ;
    hcalCfgPflow_.hOverEPtMin = psetPreselectionPflow.getParameter<double>("hOverEPtMin") ;
   }

  // Ecal rec hits configuration
  GsfElectronAlgo::EcalRecHitsConfiguration recHitsCfg ;
  const std::vector<std::string> flagnamesbarrel = cfg.getParameter<std::vector<std::string> >("recHitFlagsToBeExcludedBarrel");
  recHitsCfg.recHitFlagsToBeExcludedBarrel = StringToEnumValue<EcalRecHit::Flags>(flagnamesbarrel);
  const std::vector<std::string> flagnamesendcaps = cfg.getParameter<std::vector<std::string> >("recHitFlagsToBeExcludedEndcaps");
  recHitsCfg.recHitFlagsToBeExcludedEndcaps = StringToEnumValue<EcalRecHit::Flags>(flagnamesendcaps);
  const std::vector<std::string> severitynamesbarrel = cfg.getParameter<std::vector<std::string> >("recHitSeverityToBeExcludedBarrel");
  recHitsCfg.recHitSeverityToBeExcludedBarrel = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesbarrel);
  const std::vector<std::string> severitynamesendcaps = cfg.getParameter<std::vector<std::string> >("recHitSeverityToBeExcludedEndcaps");
  recHitsCfg.recHitSeverityToBeExcludedEndcaps = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesendcaps);
  //recHitsCfg.severityLevelCut = cfg.getParameter<int>("severityLevelCut") ;

  // isolation
  const GsfElectronAlgo::IsolationConfiguration isoCfg{
      .intRadiusHcal = cfg.getParameter<double>("intRadiusHcal"),
      .etMinHcal = cfg.getParameter<double>("etMinHcal"),
      .intRadiusEcalBarrel = cfg.getParameter<double>("intRadiusEcalBarrel"),
      .intRadiusEcalEndcaps = cfg.getParameter<double>("intRadiusEcalEndcaps"),
      .jurassicWidth = cfg.getParameter<double>("jurassicWidth"),
      .etMinBarrel = cfg.getParameter<double>("etMinBarrel"),
      .eMinBarrel = cfg.getParameter<double>("eMinBarrel"),
      .etMinEndcaps = cfg.getParameter<double>("etMinEndcaps"),
      .eMinEndcaps = cfg.getParameter<double>("eMinEndcaps"),
      .vetoClustered = cfg.getParameter<bool>("vetoClustered"),
      .useNumCrystals = cfg.getParameter<bool>("useNumCrystals")
  };

  const RegressionHelper::Configuration regressionCfg{
      .ecalRegressionWeightLabels = cfg.getParameter<std::vector<std::string> >("ecalRefinedRegressionWeightLabels"),
      .ecalWeightsFromDB = cfg.getParameter<bool>("ecalWeightsFromDB"),
      .ecalRegressionWeightFiles = cfg.getParameter<std::vector<std::string> >("ecalRefinedRegressionWeightFiles"),
      .combinationRegressionWeightLabels = cfg.getParameter<std::vector<std::string> >("combinationRegressionWeightLabels"),
      .combinationWeightsFromDB = cfg.getParameter<bool>("combinationWeightsFromDB"),
      .combinationRegressionWeightFiles = cfg.getParameter<std::vector<std::string> >("combinationRegressionWeightFile")
  };

  // functions for corrector
  EcalClusterFunctionBaseClass * const superClusterErrorFunction =
      EcalClusterFunctionFactory::get()->create(cfg.getParameter<std::string>("superClusterErrorFunction"),cfg);
  EcalClusterFunctionBaseClass * const crackCorrectionFunction =
      EcalClusterFunctionFactory::get()->create(cfg.getParameter<std::string>("crackCorrectionFunction"),cfg) ;

  mva_NIso_Cfg_.vweightsfiles = cfg.getParameter<std::vector<std::string>>("SoftElecMVAFilesString");
  mva_Iso_Cfg_.vweightsfiles  = cfg.getParameter<std::vector<std::string>>("ElecMVAFilesString");

  // create algo
  algo_ = new GsfElectronAlgo
   ( inputCfg_, strategyCfg_,
     cutsCfg_,cutsCfgPflow_,
     hcalCfg_,hcalCfgPflow_,
     isoCfg,recHitsCfg,
     superClusterErrorFunction,
     crackCorrectionFunction,
     mva_NIso_Cfg_,
     mva_Iso_Cfg_,
     regressionCfg,
     cfg.getParameter<edm::ParameterSet>("trkIsol03Cfg"),
     cfg.getParameter<edm::ParameterSet>("trkIsol04Cfg")
   ) ;
 }

GsfElectronBaseProducer::~GsfElectronBaseProducer() { delete algo_ ; }

void GsfElectronBaseProducer::beginEvent( edm::Event & event, const edm::EventSetup & setup )
 {
  // check configuration
  if (!ecalSeedingParametersChecked_)
   {
    ecalSeedingParametersChecked_ = true ;
    edm::Handle<reco::ElectronSeedCollection> seeds ;
    event.getByToken(inputCfg_.seedsTag,seeds) ;
    if (!seeds.isValid())
     {
      edm::LogWarning("GsfElectronAlgo|UnreachableSeedsProvenance")
        <<"Cannot check consistency of parameters with ecal seeding ones,"
        <<" because the original collection of seeds is not any more available." ;
     }
    else
     {
      checkEcalSeedingParameters(edm::parameterSet(*seeds.provenance())) ;
     }
   }

  // init the algo
  algo_->checkSetup(setup) ;
  algo_->beginEvent(event) ;
 }

void GsfElectronBaseProducer::fillEvent( edm::Event & event )
 {
  // all electrons
  algo_->displayInternalElectrons("GsfElectronAlgo Info (before preselection)") ;
  // preselection
  if (strategyCfg_.applyPreselection)
   {
    algo_->removeNotPreselectedElectrons() ;
    algo_->displayInternalElectrons("GsfElectronAlgo Info (after preselection)") ;
   }
  // ambiguity
  algo_->setAmbiguityData() ;
  if (strategyCfg_.applyAmbResolution)
   {
    algo_->removeAmbiguousElectrons() ;
    algo_->displayInternalElectrons("GsfElectronAlgo Info (after amb. solving)") ;
   }
  // final filling
  auto finalCollection = std::make_unique<GsfElectronCollection>();
  algo_->copyElectrons(*finalCollection) ;
  orphanHandle_ = event.put(std::move(finalCollection));
}

void GsfElectronBaseProducer::endEvent() { algo_->endEvent(); }

void GsfElectronBaseProducer::checkEcalSeedingParameters( edm::ParameterSet const & pset )
 {
  if ( !pset.exists("SeedConfiguration") ) { return; }
  edm::ParameterSet seedConfiguration = pset.getParameter<edm::ParameterSet>("SeedConfiguration") ;

  if (seedConfiguration.getParameter<bool>("applyHOverECut"))
   {
    if ((hcalCfg_.hOverEConeSize!=0)&&(hcalCfg_.hOverEConeSize!=seedConfiguration.getParameter<double>("hOverEConeSize")))
     { edm::LogWarning("GsfElectronAlgo|InconsistentParameters") <<"The H/E cone size ("<<hcalCfg_.hOverEConeSize<<") is different from ecal seeding ("<<seedConfiguration.getParameter<double>("hOverEConeSize")<<")." ; }
    if (cutsCfg_.maxHOverEBarrel<seedConfiguration.getParameter<double>("maxHOverEBarrel"))
     { edm::LogWarning("GsfElectronAlgo|InconsistentParameters") <<"The max barrel H/E is lower than during ecal seeding." ; }
    if (cutsCfg_.maxHOverEEndcaps<seedConfiguration.getParameter<double>("maxHOverEEndcaps"))
     { edm::LogWarning("GsfElectronAlgo|InconsistentParameters") <<"The max endcaps H/E is lower than during ecal seeding." ; }
   }

  if (cutsCfg_.minSCEtBarrel<seedConfiguration.getParameter<double>("SCEtCut"))
   { edm::LogWarning("GsfElectronAlgo|InconsistentParameters") <<"The minimum super-cluster Et in barrel is lower than during ecal seeding." ; }
  if (cutsCfg_.minSCEtEndcaps<seedConfiguration.getParameter<double>("SCEtCut"))
   { edm::LogWarning("GsfElectronAlgo|InconsistentParameters") <<"The minimum super-cluster Et in endcaps is lower than during ecal seeding." ; }
 }
