
#include "GsfElectronBaseProducer.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


#include <iostream>

using namespace reco;

void GsfElectronBaseProducer::fillDescription( edm::ParameterSetDescription & desc )
 {
  // input collections
  desc.add<edm::InputTag>("previousGsfElectronsTag",edm::InputTag("ecalDrivenGsfElectrons")) ;
  desc.add<edm::InputTag>("gsfElectronCoresTag",edm::InputTag("gsfElectronCores")) ;
  desc.add<edm::InputTag>("hcalTowers",edm::InputTag("towerMaker")) ;
  desc.add<edm::InputTag>("reducedBarrelRecHitCollectionTag",edm::InputTag("ecalRecHit","EcalRecHitsEB")) ;
  desc.add<edm::InputTag>("reducedEndcapRecHitCollectionTag",edm::InputTag("ecalRecHit","EcalRecHitsEE")) ;
  //desc.add<edm::InputTag>("pfMvaTag",edm::InputTag("pfElectronTranslator:pf")) ;
  desc.add<edm::InputTag>("seedsTag",edm::InputTag("ecalDrivenElectronSeeds")) ;
  desc.add<edm::InputTag>("beamSpotTag",edm::InputTag("offlineBeamSpot")) ;

  // backward compatibility mechanism for ctf tracks
  desc.add<bool>("ctfTracksCheck",true) ;
  desc.add<edm::InputTag>("ctfTracksTag",edm::InputTag("generalTracks")) ;

  // steering
  desc.add<bool>("applyPreselection",false) ;
  desc.add<bool>("applyEtaCorrection",false) ;
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

  // Isolation algos configuration
  desc.add<double>("intRadiusBarrelTk",0.015) ;
  desc.add<double>("intRadiusEndcapTk",0.015) ;
  desc.add<double>("stripBarrelTk",0.015) ;
  desc.add<double>("stripEndcapTk",0.015) ;
  desc.add<double>("ptMinTk",0.7) ;
  desc.add<double>("maxVtxDistTk",0.2) ;
  desc.add<double>("maxDrbTk",999999999.) ;
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
  desc.add<int>("severityLevelCut",4) ;
  desc.add<double>("severityRecHitThreshold",5.0) ;
  desc.add<double>("spikeIdThreshold",0.95) ;
  desc.add<std::string>("spikeIdString","kSwissCrossBordersIncluded") ;
  desc.add<std::vector<int> >("recHitFlagsToBeExcluded") ;

  edm::ParameterSetDescription descNested ;
  descNested.add<std::string>("propagatorAlongTISE","PropagatorWithMaterial") ;
  descNested.add<std::string>("propagatorOppositeTISE","PropagatorWithMaterialOpposite") ;
  desc.add<edm::ParameterSetDescription>("TransientInitialStateEstimatorParameters",descNested) ;

  // Corrections
  desc.add<std::string>("superClusterErrorFunction","EcalClusterEnergyUncertainty") ;
 }

GsfElectronBaseProducer::GsfElectronBaseProducer( const edm::ParameterSet& cfg )
 : pfTranslatorParametersChecked_(false),
   ecalSeedingParametersChecked_(false)
 {
  produces<GsfElectronCollection>();

  inputCfg_.previousGsfElectrons = cfg.getParameter<edm::InputTag>("previousGsfElectronsTag");
  inputCfg_.gsfElectronCores = cfg.getParameter<edm::InputTag>("gsfElectronCoresTag");
  inputCfg_.hcalTowersTag = cfg.getParameter<edm::InputTag>("hcalTowers") ;
  //inputCfg_.tracks_ = cfg.getParameter<edm::InputTag>("tracks");
  inputCfg_.reducedBarrelRecHitCollection = cfg.getParameter<edm::InputTag>("reducedBarrelRecHitCollectionTag") ;
  inputCfg_.reducedEndcapRecHitCollection = cfg.getParameter<edm::InputTag>("reducedEndcapRecHitCollectionTag") ;
  inputCfg_.pfMVA = cfg.getParameter<edm::InputTag>("pfMvaTag") ;
  inputCfg_.ctfTracks = cfg.getParameter<edm::InputTag>("ctfTracksTag");
  inputCfg_.seedsTag = cfg.getParameter<edm::InputTag>("seedsTag"); // used to check config consistency with seeding
  inputCfg_.beamSpotTag = cfg.getParameter<edm::InputTag>("beamSpotTag") ;

  strategyCfg_.applyPreselection = cfg.getParameter<bool>("applyPreselection") ;
  strategyCfg_.applyEtaCorrection = cfg.getParameter<bool>("applyEtaCorrection") ;
  strategyCfg_.applyAmbResolution = cfg.getParameter<bool>("applyAmbResolution") ;
  strategyCfg_.ambSortingStrategy = cfg.getParameter<unsigned>("ambSortingStrategy") ;
  strategyCfg_.ambClustersOverlapStrategy = cfg.getParameter<unsigned>("ambClustersOverlapStrategy") ;
  strategyCfg_.addPflowElectrons = cfg.getParameter<bool>("addPflowElectrons") ;
  strategyCfg_.ctfTracksCheck = cfg.getParameter<bool>("ctfTracksCheck");

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
  cutsCfg_.maxDeltaPhiBarrel = cfg.getParameter<double>("maxDeltaPhiBarrel") ;
  cutsCfg_.maxDeltaPhiEndcaps = cfg.getParameter<double>("maxDeltaPhiEndcaps") ;
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
  cutsCfgPflow_.maxTIP = cfg.getParameter<double>("maxTIPPflow") ;
  cutsCfgPflow_.seedFromTEC = true ; // not applied for pflow

  // hcal helpers
  hcalCfg_.hOverEConeSize = cfg.getParameter<double>("hOverEConeSize") ;
  if (hcalCfg_.hOverEConeSize>0)
   {
    hcalCfg_.useTowers = true ;
    hcalCfg_.hcalTowers = cfg.getParameter<edm::InputTag>("hcalTowers") ;
    hcalCfg_.hOverEPtMin = cfg.getParameter<double>("hOverEPtMin") ;
   }
  hcalCfgPflow_.hOverEConeSize = cfg.getParameter<double>("hOverEConeSizePflow") ;
  if (hcalCfgPflow_.hOverEConeSize>0)
   {
    hcalCfgPflow_.useTowers = true ;
    hcalCfgPflow_.hcalTowers = cfg.getParameter<edm::InputTag>("hcalTowers") ;
    hcalCfgPflow_.hOverEPtMin = cfg.getParameter<double>("hOverEPtMinPflow") ;
   }

  // isolation
  GsfElectronAlgo::IsolationConfiguration isoCfg ;
  isoCfg.intRadiusBarrelTk = cfg.getParameter<double>("intRadiusBarrelTk") ;
  isoCfg.intRadiusEndcapTk = cfg.getParameter<double>("intRadiusEndcapTk") ;
  isoCfg.stripBarrelTk = cfg.getParameter<double>("stripBarrelTk") ;
  isoCfg.stripEndcapTk = cfg.getParameter<double>("stripEndcapTk") ;
  isoCfg.ptMinTk = cfg.getParameter<double>("ptMinTk") ;
  isoCfg.maxVtxDistTk = cfg.getParameter<double>("maxVtxDistTk") ;
  isoCfg.maxDrbTk = cfg.getParameter<double>("maxDrbTk") ;
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

  // spike removal configuration
  GsfElectronAlgo::SpikeConfiguration spikeCfg ;
  spikeCfg.severityLevelCut = cfg.getParameter<int>("severityLevelCut") ;
  spikeCfg.severityRecHitThreshold = cfg.getParameter<double>("severityRecHitThreshold") ;
  spikeCfg.spikeIdThreshold = cfg.getParameter<double>("spikeIdThreshold") ;
  std::string spikeIdString = cfg.getParameter<std::string>("spikeIdString") ;
  if     (!spikeIdString.compare("kE1OverE9"))   spikeCfg.spikeId = EcalSeverityLevelAlgo::kE1OverE9 ;
  else if(!spikeIdString.compare("kSwissCross")) spikeCfg.spikeId = EcalSeverityLevelAlgo::kSwissCross ;
  else if(!spikeIdString.compare("kSwissCrossBordersIncluded")) spikeCfg.spikeId = EcalSeverityLevelAlgo::kSwissCrossBordersIncluded ;
  else
   {
    spikeCfg.spikeId = EcalSeverityLevelAlgo::kSwissCrossBordersIncluded ;
    edm::LogWarning("GsfElectronAlgo|SpikeRemovalForIsolation")
      << "Cannot find the requested method. kSwissCross set instead." ;
   }
  spikeCfg.recHitFlagsToBeExcluded = cfg.getParameter<std::vector<int> >("recHitFlagsToBeExcluded") ;

  // function for corrector
  EcalClusterFunctionBaseClass * superClusterErrorFunction = 0 ;
  std::string superClusterErrorFunctionName
   = cfg.getParameter<std::string>("superClusterErrorFunction") ;
  if (superClusterErrorFunctionName!="")
   {
    superClusterErrorFunction
     = EcalClusterFunctionFactory::get()->create(superClusterErrorFunctionName,cfg) ;
   }

  // create algo
  algo_ = new GsfElectronAlgo
   ( inputCfg_, strategyCfg_,
     cutsCfg_,cutsCfgPflow_,
     hcalCfg_,hcalCfgPflow_,
     isoCfg,spikeCfg,
     superClusterErrorFunction ) ;
 }

GsfElectronBaseProducer::~GsfElectronBaseProducer()
 { delete algo_ ; }

void GsfElectronBaseProducer::beginEvent( edm::Event & event, const edm::EventSetup & setup )
 {
  // check configuration
  if (!pfTranslatorParametersChecked_)
   {
    pfTranslatorParametersChecked_ = true ;
    edm::Handle<edm::ValueMap<float> > pfMva ;
    event.getByLabel(inputCfg_.pfMVA,pfMva) ;
    checkPfTranslatorParameters(pfMva.provenance()->psetID()) ;
   }
  if (!ecalSeedingParametersChecked_)
   {
    ecalSeedingParametersChecked_ = true ;
    edm::Handle<reco::ElectronSeedCollection> seeds ;
    event.getByLabel(inputCfg_.seedsTag,seeds) ;
    if (!seeds.isValid())
     {
      edm::LogWarning("GsfElectronAlgo|UnreachableSeedsProvenance")
        <<"Cannot check consistency of parameters with ecal seeding ones,"
        <<" because the original collection of seeds is not any more available." ;
     }
    else
     {
      checkEcalSeedingParameters(seeds.provenance()->psetID()) ;
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
  std::auto_ptr<GsfElectronCollection> finalCollection( new GsfElectronCollection ) ;
  algo_->copyElectrons(*finalCollection) ;
  event.put(finalCollection) ;
 }

void GsfElectronBaseProducer::endEvent()
 {
  algo_->endEvent() ;
 }

void GsfElectronBaseProducer::checkPfTranslatorParameters( edm::ParameterSetID const & psetid )
 {
  edm::ParameterSet pset ;
  edm::pset::Registry::instance()->getMapped(psetid,pset) ;
  edm::ParameterSet mvaBlock = pset.getParameter<edm::ParameterSet>("MVACutBlock") ;
  double pfTranslatorMinMva = mvaBlock.getParameter<double>("MVACut") ;
  double pfTranslatorUndefined = -99. ;
  if (strategyCfg_.applyPreselection&&(cutsCfgPflow_.minMVA<pfTranslatorMinMva))
   {
    // For pure tracker seeded electrons, if MVA is under translatorMinMva, there is no supercluster
    // of any kind available, so GsfElectronCoreProducer has already discarded the electron.
    edm::LogWarning("GsfElectronAlgo|MvaCutTooLow")
      <<"Parameter minMVAPflow will have no effect on purely tracker seeded electrons."
      <<" It is inferior to the cut already applied by PFlow translator." ;
   }
  if (strategyCfg_.applyPreselection&&(cutsCfg_.minMVA<pfTranslatorMinMva))
   {
    // For ecal seeded electrons, there is a cluster and GsfElectronCoreProducer has kept all electrons,
    // but when MVA is under translatorMinMva, the translator has not stored the supercluster and
    // forced the MVA value to translatorUndefined
    if (cutsCfg_.minMVA>pfTranslatorUndefined)
     {
      edm::LogWarning("GsfElectronAlgo|IncompletePflowInformation")
        <<"Parameter minMVA is inferior to the cut applied by PFlow translator."
        <<" Some ecal (and eventually tracker) seeded electrons may lack their MVA value and PFlow supercluster." ;
     }
    else
     {
      // the MVA value has been forced to translatorUndefined, inferior minMVAPflow
      // so the cut actually applied is the PFlow one
      throw cms::Exception("GsfElectronAlgo|BadMvaCut")
        <<"Parameter minMVA is inferior to the lowest possible value."
        <<" Every electron will be blessed whatever other criteria." ;
     }
   }
 }

void GsfElectronBaseProducer::checkEcalSeedingParameters( edm::ParameterSetID const & psetid )
 {
  edm::ParameterSet pset ;
  edm::pset::Registry::instance()->getMapped(psetid,pset) ;
  edm::ParameterSet seedConfiguration = pset.getParameter<edm::ParameterSet>("SeedConfiguration") ;
  edm::ParameterSet orderedHitsFactoryPSet = seedConfiguration.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet") ;
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


