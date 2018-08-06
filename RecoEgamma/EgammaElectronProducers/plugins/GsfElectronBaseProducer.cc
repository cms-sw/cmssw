
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

#include <iostream>

using namespace reco;

void GsfElectronBaseProducer::fillDescription( edm::ParameterSetDescription & desc)
 {
  // input collections
  desc.add<edm::InputTag>("previousGsfElectronsTag",edm::InputTag("ecalDrivenGsfElectrons")) ;
  desc.add<edm::InputTag>("pflowGsfElectronsTag",edm::InputTag("pflowGsfElectrons")) ;
  desc.add<edm::InputTag>("gsfElectronCoresTag",edm::InputTag("gsfElectronCores")) ;
  desc.add<edm::InputTag>("hcalTowers",edm::InputTag("towerMaker")) ;
  desc.add<edm::InputTag>("barrelRecHitCollectionTag",edm::InputTag("ecalRecHit","EcalRecHitsEB")) ;
  desc.add<edm::InputTag>("endcapRecHitCollectionTag",edm::InputTag("ecalRecHit","EcalRecHitsEE")) ;
  //desc.add<edm::InputTag>("pfMvaTag",edm::InputTag("pfElectronTranslator:pf")) ;
  desc.add<edm::InputTag>("seedsTag",edm::InputTag("ecalDrivenElectronSeeds")) ;
  desc.add<edm::InputTag>("beamSpotTag",edm::InputTag("offlineBeamSpot")) ;
  desc.add<edm::InputTag>("gsfPfRecTracksTag",edm::InputTag("pfTrackElec")) ;
  //desc.add<std::vector<std::string>>("SoftElecMVAFilesString",std::vector<std::string> ("SoftElecMVAFile"));


  // backward compatibility mechanism for ctf tracks
  desc.add<bool>("ctfTracksCheck",true) ;
  desc.add<edm::InputTag>("ctfTracksTag",edm::InputTag("generalTracks")) ;

  desc.add<bool>("gedElectronMode",true) ;
  desc.add<double>("PreSelectMVA",-0.1) ;
  desc.add<double>("MaxElePtForOnlyMVA",50.0) ;

  // steering
  desc.add<bool>("useGsfPfRecTracks",true) ;
  desc.add<bool>("applyPreselection",false) ;
  desc.add<bool>("ecalDrivenEcalEnergyFromClassBasedParameterization",false) ;
  desc.add<bool>("ecalDrivenEcalErrorFromClassBasedParameterization",false) ;
  desc.add<bool>("pureTrackerDrivenEcalErrorFromSimpleParameterization",false) ;
  desc.add<bool>("applyAmbResolution",false) ;
  desc.add<unsigned>("ambSortingStrategy",1) ;
  desc.add<unsigned>("ambClustersOverlapStrategy",1) ;
  //desc.add<bool>("addPflowElectrons",true) ;

//  // preselection parameters (ecal driven electrons)
//  desc.add<bool>("seedFromTEC",true) ;
//  desc.add<double>("minSCEtBarrel",4.0) ;
//  desc.add<double>("minSCEtEndcaps",4.0) ;
//  desc.add<double>("minEOverPBarrel",0.0) ;
//  desc.add<double>("maxEOverPBarrel",999999999.) ;
//  desc.add<double>("minEOverPEndcaps",0.0) ;
//  desc.add<double>("maxEOverPEndcaps",999999999.) ;
//  desc.add<double>("maxDeltaEtaBarrel",0.02) ;
//  desc.add<double>("maxDeltaEtaEndcaps",0.02) ;
//  desc.add<double>("maxDeltaPhiBarrel",0.15) ;
//  desc.add<double>("maxDeltaPhiEndcaps",0.15) ;
//  desc.add<double>("hOverEConeSize",0.15) ;
//  desc.add<double>("hOverEPtMin",0.) ;
//  desc.add<double>("maxHOverEBarrel",0.15) ;
//  desc.add<double>("maxHOverEEndcaps",0.15) ;
//  desc.add<double>("maxHBarrel",0.0) ;
//  desc.add<double>("maxHEndcaps",0.0) ;
//  desc.add<double>("maxSigmaIetaIetaBarrel",999999999.) ;
//  desc.add<double>("maxSigmaIetaIetaEndcaps",999999999.) ;
//  desc.add<double>("maxFbremBarrel",999999999.) ;
//  desc.add<double>("maxFbremEndcaps",999999999.) ;
//  desc.add<bool>("isBarrel",false) ;
//  desc.add<bool>("isEndcaps",false) ;
//  desc.add<bool>("isFiducial",false) ;
//  desc.add<double>("maxTIP",999999999.) ;
//  desc.add<double>("minMVA",-0.4) ;
//
//  // preselection parameters (tracker driven only electrons)
//  desc.add<double>("minSCEtBarrelPflow",0.0) ;
//  desc.add<double>("minSCEtEndcapsPflow",0.0) ;
//  desc.add<double>("minEOverPBarrelPflow",0.0) ;
//  desc.add<double>("maxEOverPBarrelPflow",999999999.) ;
//  desc.add<double>("minEOverPEndcapsPflow",0.0) ;
//  desc.add<double>("maxEOverPEndcapsPflow",999999999.) ;
//  desc.add<double>("maxDeltaEtaBarrelPflow",999999999.) ;
//  desc.add<double>("maxDeltaEtaEndcapsPflow",999999999.) ;
//  desc.add<double>("maxDeltaPhiBarrelPflow",999999999.) ;
//  desc.add<double>("maxDeltaPhiEndcapsPflow",999999999.) ;
//  desc.add<double>("hOverEConeSizePflow",0.15) ;
//  desc.add<double>("hOverEPtMinPflow",0.) ;
//  desc.add<double>("maxHOverEBarrelPflow",999999999.) ;
//  desc.add<double>("maxHOverEEndcapsPflow",999999999.) ;
//  desc.add<double>("maxHBarrelPflow",0.0) ;
//  desc.add<double>("maxHEndcapsPflow",0.0) ;
//  desc.add<double>("maxSigmaIetaIetaBarrelPflow",999999999.) ;
//  desc.add<double>("maxSigmaIetaIetaEndcapsPflow",999999999.) ;
//  desc.add<double>("maxFbremBarrelPflow",999999999.) ;
//  desc.add<double>("maxFbremEndcapsPflow",999999999.) ;
//  desc.add<bool>("isBarrelPflow",false) ;
//  desc.add<bool>("isEndcapsPflow",false) ;
//  desc.add<bool>("isFiducialPflow",false) ;
//  desc.add<double>("maxTIPPflow",999999999.) ;
//  desc.add<double>("minMVAPflow",-0.4) ;

  // Ecal rec hits configuration
  desc.add<std::vector<int> >("recHitFlagsToBeExcludedBarrel") ;
  desc.add<std::vector<int> >("recHitFlagsToBeExcludedEndcaps") ;
  desc.add<std::vector<int> >("recHitSeverityToBeExcludedBarrel") ;
  desc.add<std::vector<int> >("recHitSeverityToBeExcludedEndcaps") ;
  //desc.add<int>("severityLevelCut",4) ;

  // Isolation algos configuration
  desc.add("trkIsol03Cfg",EleTkIsolFromCands::pSetDescript());
  desc.add("trkIsol04Cfg",EleTkIsolFromCands::pSetDescript());
  desc.add<double>("intRadiusHcal",0.15) ;
  desc.add<double>("etMinHcal",0.0) ;
  desc.add<double>("intRadiusEcalBarrel",3.0) ;
  desc.add<double>("intRadiusEcalEndcaps",3.0) ;
  desc.add<double>("jurassicWidth",1.5) ;
  desc.add<double>("etMinBarrel",0.0) ;
  desc.add<double>("eMinBarrel",0.08) ;
  desc.add<double>("etMinEndcaps",0.1) ;
  desc.add<double>("eMinEndcaps",0.0) ;
  desc.add<bool>("vetoClustered",false) ;
  desc.add<bool>("useNumCrystals",true) ;

  edm::ParameterSetDescription descNested ;
  descNested.add<std::string>("propagatorAlongTISE","PropagatorWithMaterial") ;
  descNested.add<std::string>("propagatorOppositeTISE","PropagatorWithMaterialOpposite") ;
  desc.add<edm::ParameterSetDescription>("TransientInitialStateEstimatorParameters",descNested) ;

  // Corrections
  desc.add<std::string>("superClusterErrorFunction","EcalClusterEnergyUncertaintyObjectSpecific") ;
  desc.add<std::string>("crackCorrectionFunction","EcalClusterCrackCorrection") ;
 }

GsfElectronBaseProducer::GsfElectronBaseProducer( const edm::ParameterSet& cfg, const gsfAlgoHelpers::HeavyObjectCache* )
 : ecalSeedingParametersChecked_(false)
 {
  produces<GsfElectronCollection>();

  inputCfg_.previousGsfElectrons = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("previousGsfElectronsTag"));
  inputCfg_.pflowGsfElectronsTag = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("pflowGsfElectronsTag"));
  inputCfg_.gsfElectronCores = consumes<reco::GsfElectronCoreCollection>(cfg.getParameter<edm::InputTag>("gsfElectronCoresTag"));
  inputCfg_.hcalTowersTag = consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers"));
  //inputCfg_.tracks_ = cfg.getParameter<edm::InputTag>("tracks");
  inputCfg_.barrelRecHitCollection = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("barrelRecHitCollectionTag"));
  inputCfg_.endcapRecHitCollection = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("endcapRecHitCollectionTag"));
  inputCfg_.pfMVA = consumes<edm::ValueMap<float> >(cfg.getParameter<edm::InputTag>("pfMvaTag"));
  inputCfg_.ctfTracks = consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("ctfTracksTag"));
  inputCfg_.seedsTag = consumes<reco::ElectronSeedCollection>(cfg.getParameter<edm::InputTag>("seedsTag")); // used to check config consistency with seeding
  inputCfg_.beamSpotTag = consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpotTag"));
  inputCfg_.gsfPfRecTracksTag = consumes<reco::GsfPFRecTrackCollection>(cfg.getParameter<edm::InputTag>("gsfPfRecTracksTag"));
  inputCfg_.vtxCollectionTag = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vtxTag"));

  bool useIsolationValues = cfg.getParameter<bool>("useIsolationValues") ;
  if ( useIsolationValues ) {
    if( ! cfg.exists("pfIsolationValues") ) {
      throw cms::Exception("GsfElectronBaseProducer|InternalError")
	<<"Missing ParameterSet pfIsolationValues" ;
    } else {
      inputCfg_.pfIsoVals = 
	cfg.getParameter<edm::ParameterSet> ("pfIsolationValues");
      std::vector<std::string> isoNames = 
	inputCfg_.pfIsoVals.getParameterNamesForType<edm::InputTag>();
      for(const std::string& name : isoNames) {
	edm::InputTag tag = 
	  inputCfg_.pfIsoVals.getParameter<edm::InputTag>(name);
	mayConsume<edm::ValueMap<double> >(tag);
      }
    }
      
    if ( ! cfg.exists("edIsolationValues") ) {
      throw cms::Exception("GsfElectronBaseProducer|InternalError")
	<<"Missing ParameterSet edIsolationValues" ;
    } else {
      inputCfg_.edIsoVals = 
	cfg.getParameter<edm::ParameterSet> ("edIsolationValues");
      std::vector<std::string> isoNames = 
	inputCfg_.edIsoVals.getParameterNamesForType<edm::InputTag>();
      for(const std::string& name : isoNames) {
	edm::InputTag tag = 
	  inputCfg_.edIsoVals.getParameter<edm::InputTag>(name);
	mayConsume<edm::ValueMap<double> >(tag);
      }
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

  cutsCfg_.minSCEtBarrel = cfg.getParameter<double>("minSCEtBarrel") ;
  cutsCfg_.minSCEtEndcaps = cfg.getParameter<double>("minSCEtEndcaps") ;
  cutsCfg_.maxEOverPBarrel = cfg.getParameter<double>("maxEOverPBarrel") ;
  cutsCfg_.maxEOverPEndcaps = cfg.getParameter<double>("maxEOverPEndcaps") ;
  cutsCfg_.minEOverPBarrel = cfg.getParameter<double>("minEOverPBarrel") ;
  cutsCfg_.minEOverPEndcaps = cfg.getParameter<double>("minEOverPEndcaps") ;

  // H/E
  cutsCfg_.maxHOverEBarrel = cfg.getParameter<double>("maxHOverEBarrel") ;
  cutsCfg_.maxHOverEEndcaps = cfg.getParameter<double>("maxHOverEEndcaps") ;
  cutsCfg_.maxHBarrel = cfg.getParameter<double>("maxHBarrel") ;
  cutsCfg_.maxHEndcaps = cfg.getParameter<double>("maxHEndcaps") ;

  cutsCfg_.maxDeltaEtaBarrel = cfg.getParameter<double>("maxDeltaEtaBarrel") ;
  cutsCfg_.maxDeltaEtaEndcaps = cfg.getParameter<double>("maxDeltaEtaEndcaps") ;
  cutsCfg_.maxDeltaPhiBarrel = cfg.getParameter<double>("maxDeltaPhiBarrel") ;
  cutsCfg_.maxDeltaPhiEndcaps = cfg.getParameter<double>("maxDeltaPhiEndcaps") ;
  cutsCfg_.maxSigmaIetaIetaBarrel = cfg.getParameter<double>("maxSigmaIetaIetaBarrel") ;
  cutsCfg_.maxSigmaIetaIetaEndcaps = cfg.getParameter<double>("maxSigmaIetaIetaEndcaps") ;
  cutsCfg_.maxFbremBarrel = cfg.getParameter<double>("maxFbremBarrel") ;
  cutsCfg_.maxFbremEndcaps = cfg.getParameter<double>("maxFbremEndcaps") ;
  cutsCfg_.isBarrel = cfg.getParameter<bool>("isBarrel") ;
  cutsCfg_.isEndcaps = cfg.getParameter<bool>("isEndcaps") ;
  cutsCfg_.isFiducial = cfg.getParameter<bool>("isFiducial") ;
  cutsCfg_.minMVA = cfg.getParameter<double>("minMVA") ;
  cutsCfg_.minMvaByPassForIsolated = cfg.getParameter<double>("minMvaByPassForIsolated") ;
  cutsCfg_.maxTIP = cfg.getParameter<double>("maxTIP") ;
  cutsCfg_.seedFromTEC = cfg.getParameter<bool>("seedFromTEC") ;

  cutsCfgPflow_.minSCEtBarrel = cfg.getParameter<double>("minSCEtBarrelPflow") ;
  cutsCfgPflow_.minSCEtEndcaps = cfg.getParameter<double>("minSCEtEndcapsPflow") ;
  cutsCfgPflow_.maxEOverPBarrel = cfg.getParameter<double>("maxEOverPBarrelPflow") ;
  cutsCfgPflow_.maxEOverPEndcaps = cfg.getParameter<double>("maxEOverPEndcapsPflow") ;
  cutsCfgPflow_.minEOverPBarrel = cfg.getParameter<double>("minEOverPBarrelPflow") ;
  cutsCfgPflow_.minEOverPEndcaps = cfg.getParameter<double>("minEOverPEndcapsPflow") ;

  // H/E
  cutsCfgPflow_.maxHOverEBarrel = cfg.getParameter<double>("maxHOverEBarrelPflow") ;
  cutsCfgPflow_.maxHOverEEndcaps = cfg.getParameter<double>("maxHOverEEndcapsPflow") ;
  cutsCfgPflow_.maxHBarrel = cfg.getParameter<double>("maxHBarrelPflow") ;
  cutsCfgPflow_.maxHEndcaps = cfg.getParameter<double>("maxHEndcapsPflow") ;

  cutsCfgPflow_.maxDeltaEtaBarrel = cfg.getParameter<double>("maxDeltaEtaBarrelPflow") ;
  cutsCfgPflow_.maxDeltaEtaEndcaps = cfg.getParameter<double>("maxDeltaEtaEndcapsPflow") ;
  cutsCfgPflow_.maxDeltaPhiBarrel = cfg.getParameter<double>("maxDeltaPhiBarrelPflow") ;
  cutsCfgPflow_.maxDeltaPhiEndcaps = cfg.getParameter<double>("maxDeltaPhiEndcapsPflow") ;
  cutsCfgPflow_.maxDeltaPhiBarrel = cfg.getParameter<double>("maxDeltaPhiBarrelPflow") ;
  cutsCfgPflow_.maxDeltaPhiEndcaps = cfg.getParameter<double>("maxDeltaPhiEndcapsPflow") ;
  cutsCfgPflow_.maxDeltaPhiBarrel = cfg.getParameter<double>("maxDeltaPhiBarrelPflow") ;
  cutsCfgPflow_.maxDeltaPhiEndcaps = cfg.getParameter<double>("maxDeltaPhiEndcapsPflow") ;
  cutsCfgPflow_.maxSigmaIetaIetaBarrel = cfg.getParameter<double>("maxSigmaIetaIetaBarrelPflow") ;
  cutsCfgPflow_.maxSigmaIetaIetaEndcaps = cfg.getParameter<double>("maxSigmaIetaIetaEndcapsPflow") ;
  cutsCfgPflow_.maxFbremBarrel = cfg.getParameter<double>("maxFbremBarrelPflow") ;
  cutsCfgPflow_.maxFbremEndcaps = cfg.getParameter<double>("maxFbremEndcapsPflow") ;
  cutsCfgPflow_.isBarrel = cfg.getParameter<bool>("isBarrelPflow") ;
  cutsCfgPflow_.isEndcaps = cfg.getParameter<bool>("isEndcapsPflow") ;
  cutsCfgPflow_.isFiducial = cfg.getParameter<bool>("isFiducialPflow") ;
  cutsCfgPflow_.minMVA = cfg.getParameter<double>("minMVAPflow") ;
  cutsCfgPflow_.minMvaByPassForIsolated = cfg.getParameter<double>("minMvaByPassForIsolatedPflow") ;
  cutsCfgPflow_.maxTIP = cfg.getParameter<double>("maxTIPPflow") ;
  cutsCfgPflow_.seedFromTEC = true ; // not applied for pflow

  // hcal helpers
  hcalCfg_.hOverEConeSize = cfg.getParameter<double>("hOverEConeSize") ;
  if (hcalCfg_.hOverEConeSize>0)
   {
    hcalCfg_.useTowers = true ;
    hcalCfg_.checkHcalStatus = cfg.getParameter<bool>("checkHcalStatus") ;
    hcalCfg_.hcalTowers = 
      consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers")) ;
    hcalCfg_.hOverEPtMin = cfg.getParameter<double>("hOverEPtMin") ;
   }
  hcalCfgPflow_.hOverEConeSize = cfg.getParameter<double>("hOverEConeSizePflow") ;
  if (hcalCfgPflow_.hOverEConeSize>0)
   {
    hcalCfgPflow_.useTowers = true ;
    hcalCfgPflow_.checkHcalStatus = cfg.getParameter<bool>("checkHcalStatus") ;
    hcalCfgPflow_.hcalTowers = 
      consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("hcalTowers")) ;
    hcalCfgPflow_.hOverEPtMin = cfg.getParameter<double>("hOverEPtMinPflow") ;
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
  GsfElectronAlgo::IsolationConfiguration isoCfg ;
  isoCfg.intRadiusHcal = cfg.getParameter<double>("intRadiusHcal") ;
  isoCfg.etMinHcal = cfg.getParameter<double>("etMinHcal") ;
  isoCfg.intRadiusEcalBarrel = cfg.getParameter<double>("intRadiusEcalBarrel") ;
  isoCfg.intRadiusEcalEndcaps = cfg.getParameter<double>("intRadiusEcalEndcaps") ;
  isoCfg.jurassicWidth = cfg.getParameter<double>("jurassicWidth") ;
  isoCfg.etMinBarrel = cfg.getParameter<double>("etMinBarrel") ;
  isoCfg.eMinBarrel = cfg.getParameter<double>("eMinBarrel") ;
  isoCfg.etMinEndcaps = cfg.getParameter<double>("etMinEndcaps") ;
  isoCfg.eMinEndcaps = cfg.getParameter<double>("eMinEndcaps") ;
  isoCfg.vetoClustered = cfg.getParameter<bool>("vetoClustered") ;
  isoCfg.useNumCrystals = cfg.getParameter<bool>("useNumCrystals") ;

 
  RegressionHelper::Configuration regressionCfg ;
  regressionCfg.ecalRegressionWeightLabels = cfg.getParameter<std::vector<std::string> >("ecalRefinedRegressionWeightLabels");
  regressionCfg.combinationRegressionWeightLabels = cfg.getParameter<std::vector<std::string> >("combinationRegressionWeightLabels");
  regressionCfg.ecalRegressionWeightFiles = cfg.getParameter<std::vector<std::string> >("ecalRefinedRegressionWeightFiles");
  regressionCfg.combinationRegressionWeightFiles = cfg.getParameter<std::vector<std::string> >("combinationRegressionWeightFile");
  regressionCfg.ecalWeightsFromDB = cfg.getParameter<bool>("ecalWeightsFromDB");
  regressionCfg.combinationWeightsFromDB = cfg.getParameter<bool>("combinationWeightsFromDB");
  // functions for corrector
  EcalClusterFunctionBaseClass * superClusterErrorFunction = nullptr ;
  std::string superClusterErrorFunctionName
   = cfg.getParameter<std::string>("superClusterErrorFunction") ;
  if (!superClusterErrorFunctionName.empty())
   {
    superClusterErrorFunction
     = EcalClusterFunctionFactory::get()->create(superClusterErrorFunctionName,cfg) ;
   }
  else
  {
   superClusterErrorFunction
    = EcalClusterFunctionFactory::get()->create("EcalClusterEnergyUncertaintyObjectSpecific",cfg) ;
  }
  EcalClusterFunctionBaseClass * crackCorrectionFunction = nullptr ;
  std::string crackCorrectionFunctionName
   = cfg.getParameter<std::string>("crackCorrectionFunction") ;
  if (!crackCorrectionFunctionName.empty())
   {
    crackCorrectionFunction
     = EcalClusterFunctionFactory::get()->create(crackCorrectionFunctionName,cfg) ;
   }


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

GsfElectronBaseProducer::~GsfElectronBaseProducer()
 { delete algo_ ; }

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

void GsfElectronBaseProducer::endEvent()
 {
  algo_->endEvent() ;
 }

void GsfElectronBaseProducer::checkEcalSeedingParameters( edm::ParameterSet const & pset )
 {
  edm::ParameterSet seedConfiguration = pset.getParameter<edm::ParameterSet>("SeedConfiguration") ;
  //edm::ParameterSet orderedHitsFactoryPSet = seedConfiguration.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet") ;
  //edm::ParameterSet seedParameters = seedConfiguration.getParameter<edm::ParameterSet>("ecalDrivenElectronSeedsParameters") ;

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


