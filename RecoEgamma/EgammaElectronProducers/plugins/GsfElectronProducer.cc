#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRecHit/interface/EcalSeverityLevel.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/EgAmbiguityTools.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

using namespace reco;

namespace {

  void setMVAOutputs(reco::GsfElectronCollection& electrons,
                     const GsfElectronAlgo::HeavyObjectCache* hoc,
                     reco::VertexCollection const& vertices) {
    for (auto& el : electrons) {
      GsfElectron::MvaOutput mvaOutput;
      mvaOutput.mva_e_pi = hoc->sElectronMVAEstimator->mva(el, vertices);
      mvaOutput.mva_Isolated = hoc->iElectronMVAEstimator->mva(el, vertices.size());
      el.setMvaOutput(mvaOutput);
    }
  }

  // Something more clever has to be found. The collections are small, so the timing is not
  // an issue here; but it is clearly suboptimal

  auto matchWithPFCandidates(std::vector<reco::PFCandidate> const& pfCandidates) {
    std::map<reco::GsfTrackRef, reco::GsfElectron::MvaInput> gsfMVAInputs{};

    //Loop over the collection of PFFCandidates
    for (auto const& pfCand : pfCandidates) {
      reco::GsfElectronRef myRef;
      // First check that the GsfTrack is non null
      if (pfCand.gsfTrackRef().isNonnull()) {
        reco::GsfElectron::MvaInput input;
        input.earlyBrem = pfCand.egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_FirstBrem);
        input.lateBrem = pfCand.egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_LateBrem);
        input.deltaEta = pfCand.egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_DeltaEtaTrackCluster);
        input.sigmaEtaEta = pfCand.egammaExtraRef()->sigmaEtaEta();
        input.hadEnergy = pfCand.egammaExtraRef()->hadEnergy();
        gsfMVAInputs[pfCand.gsfTrackRef()] = input;
      }
    }
    return gsfMVAInputs;
  }

  void logElectrons(reco::GsfElectronCollection const& electrons, edm::Event const& event, const std::string& title) {
    LogTrace("GsfElectronAlgo") << "========== " << title << " ==========";
    LogTrace("GsfElectronAlgo") << "Event: " << event.id();
    LogTrace("GsfElectronAlgo") << "Number of electrons: " << electrons.size();
    for (auto const& ele : electrons) {
      LogTrace("GsfElectronAlgo") << "Electron with charge, pt, eta, phi: " << ele.charge() << " , " << ele.pt()
                                  << " , " << ele.eta() << " , " << ele.phi();
    }
    LogTrace("GsfElectronAlgo") << "=================================================";
  }

}  // namespace

class GsfElectronProducer : public edm::stream::EDProducer<edm::GlobalCache<GsfElectronAlgo::HeavyObjectCache>> {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  explicit GsfElectronProducer(const edm::ParameterSet&, const GsfElectronAlgo::HeavyObjectCache*);

  static std::unique_ptr<GsfElectronAlgo::HeavyObjectCache> initializeGlobalCache(const edm::ParameterSet& conf) {
    return std::make_unique<GsfElectronAlgo::HeavyObjectCache>(conf);
  }

  static void globalEndJob(GsfElectronAlgo::HeavyObjectCache const*) {}

  // ------------ method called to produce the data  ------------
  void produce(edm::Event& event, const edm::EventSetup& setup) override;

private:
  std::unique_ptr<GsfElectronAlgo> algo_;

  // configurables
  GsfElectronAlgo::Tokens inputCfg_;
  GsfElectronAlgo::StrategyConfiguration strategyCfg_;
  const GsfElectronAlgo::CutsConfiguration cutsCfg_;
  ElectronHcalHelper::Configuration hcalCfg_;

  bool isPreselected(reco::GsfElectron const& ele) const;
  void setAmbiguityData(reco::GsfElectronCollection& electrons,
                        edm::Event const& event,
                        bool ignoreNotPreselected = true) const;

  // check expected configuration of previous modules
  bool ecalSeedingParametersChecked_;
  void checkEcalSeedingParameters(edm::ParameterSet const&);

  const edm::EDPutTokenT<reco::GsfElectronCollection> electronPutToken_;
  const edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracksTag_;
  edm::EDGetTokenT<reco::PFCandidateCollection> egmPFCandidateCollection_;

  const bool useGsfPfRecTracks_;

  const bool resetMvaValuesUsingPFCandidates_;
};

void GsfElectronProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // input collections
  desc.add<edm::InputTag>("gsfElectronCoresTag", {"ecalDrivenGsfElectronCores"});
  desc.add<edm::InputTag>("hcalTowers", {"towerMaker"});
  desc.add<edm::InputTag>("vtxTag", {"offlinePrimaryVertices"});
  desc.add<edm::InputTag>("conversionsTag", {"allConversions"});
  desc.add<edm::InputTag>("gsfPfRecTracksTag", {"pfTrackElec"});
  desc.add<edm::InputTag>("barrelRecHitCollectionTag", {"ecalRecHit", "EcalRecHitsEB"});
  desc.add<edm::InputTag>("endcapRecHitCollectionTag", {"ecalRecHit", "EcalRecHitsEE"});
  desc.add<edm::InputTag>("seedsTag", {"ecalDrivenElectronSeeds"});
  desc.add<edm::InputTag>("beamSpotTag", {"offlineBeamSpot"});
  desc.add<edm::InputTag>("egmPFCandidatesTag", {"particleFlowEGamma"});
  desc.add<bool>("checkHcalStatus", true);

  // steering
  desc.add<bool>("useDefaultEnergyCorrection", true);
  desc.add<bool>("useCombinationRegression", false);
  desc.add<bool>("ecalDrivenEcalEnergyFromClassBasedParameterization", true);
  desc.add<bool>("ecalDrivenEcalErrorFromClassBasedParameterization", true);
  desc.add<bool>("applyPreselection", false);
  desc.add<bool>("useEcalRegression", false);
  desc.add<bool>("applyAmbResolution", false);
  desc.add<bool>("ignoreNotPreselected", true);
  desc.add<bool>("useGsfPfRecTracks", true);
  desc.add<bool>("pureTrackerDrivenEcalErrorFromSimpleParameterization", true);
  desc.add<unsigned int>("ambSortingStrategy", 1);
  desc.add<unsigned int>("ambClustersOverlapStrategy", 1);
  desc.add<bool>("fillConvVtxFitProb", true);
  desc.add<bool>("resetMvaValuesUsingPFCandidates", false);

  // Ecal rec hits configuration
  desc.add<std::vector<std::string>>("recHitFlagsToBeExcludedBarrel");
  desc.add<std::vector<std::string>>("recHitFlagsToBeExcludedEndcaps");
  desc.add<std::vector<std::string>>("recHitSeverityToBeExcludedBarrel");
  desc.add<std::vector<std::string>>("recHitSeverityToBeExcludedEndcaps");

  // Isolation algos configuration
  desc.add("trkIsol03Cfg", EleTkIsolFromCands::pSetDescript());
  desc.add("trkIsol04Cfg", EleTkIsolFromCands::pSetDescript());
  desc.add("trkIsolHEEP03Cfg", EleTkIsolFromCands::pSetDescript());
  desc.add("trkIsolHEEP04Cfg", EleTkIsolFromCands::pSetDescript());
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

  // backward compatibility mechanism for ctf tracks
  desc.add<bool>("ctfTracksCheck", true);
  desc.add<edm::InputTag>("ctfTracksTag", {"generalTracks"});

  desc.add<double>("MaxElePtForOnlyMVA", 50.0);
  desc.add<double>("PreSelectMVA", -0.1);

  {
    edm::ParameterSetDescription psd0;
    psd0.add<double>("minSCEtBarrel", 4.0);
    psd0.add<double>("minSCEtEndcaps", 4.0);
    psd0.add<double>("minEOverPBarrel", 0.0);
    psd0.add<double>("minEOverPEndcaps", 0.0);
    psd0.add<double>("maxEOverPBarrel", 999999999.0);
    psd0.add<double>("maxEOverPEndcaps", 999999999.0);
    psd0.add<double>("maxDeltaEtaBarrel", 0.02);
    psd0.add<double>("maxDeltaEtaEndcaps", 0.02);
    psd0.add<double>("maxDeltaPhiBarrel", 0.15);
    psd0.add<double>("maxDeltaPhiEndcaps", 0.15);
    psd0.add<double>("hOverEConeSize", 0.15);
    psd0.add<double>("hOverEPtMin", 0.0);
    psd0.add<double>("maxHOverEBarrelCone", 0.15);
    psd0.add<double>("maxHOverEEndcapsCone", 0.15);
    psd0.add<double>("maxHBarrelCone", 0.0);
    psd0.add<double>("maxHEndcapsCone", 0.0);
    psd0.add<double>("maxHOverEBarrelTower", 0.15);
    psd0.add<double>("maxHOverEEndcapsTower", 0.15);
    psd0.add<double>("maxHBarrelTower", 0.0);
    psd0.add<double>("maxHEndcapsTower", 0.0);
    psd0.add<double>("maxSigmaIetaIetaBarrel", 999999999.0);
    psd0.add<double>("maxSigmaIetaIetaEndcaps", 999999999.0);
    psd0.add<double>("maxFbremBarrel", 999999999.0);
    psd0.add<double>("maxFbremEndcaps", 999999999.0);
    psd0.add<bool>("isBarrel", false);
    psd0.add<bool>("isEndcaps", false);
    psd0.add<bool>("isFiducial", false);
    psd0.add<bool>("seedFromTEC", true);
    psd0.add<double>("maxTIP", 999999999.0);
    // preselection parameters
    desc.add<edm::ParameterSetDescription>("preselection", psd0);
  }

  // Corrections
  desc.add<std::string>("crackCorrectionFunction", "EcalClusterCrackCorrection");

  desc.add<bool>("ecalWeightsFromDB", true);
  desc.add<std::vector<std::string>>("ecalRefinedRegressionWeightFiles", {})
      ->setComment("if not from DB. Otherwise, keep empty");
  desc.add<bool>("combinationWeightsFromDB", true);
  desc.add<std::vector<std::string>>("combinationRegressionWeightFile", {})
      ->setComment("if not from DB. Otherwise, keep empty");

  // regression. The labels are needed in all cases.
  desc.add<std::vector<std::string>>("ecalRefinedRegressionWeightLabels", {});
  desc.add<std::vector<std::string>>("combinationRegressionWeightLabels", {});

  desc.add<std::vector<std::string>>(
      "ElecMVAFilesString",
      {
          "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_10_17Feb2011.weights.xml",
          "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_12_17Feb2011.weights.xml",
          "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_20_17Feb2011.weights.xml",
          "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_22_17Feb2011.weights.xml",
      });
  desc.add<std::vector<std::string>>(
      "SoftElecMVAFilesString",
      {
          "RecoEgamma/ElectronIdentification/data/TMVA_BDTSoftElectrons_7Feb2014.weights.xml",
      });

  descriptions.add("gsfElectronProducer", desc);
}

namespace {
  GsfElectronAlgo::CutsConfiguration makeCutsConfiguration(edm::ParameterSet const& pset) {
    return GsfElectronAlgo::CutsConfiguration{
        .minSCEtBarrel = pset.getParameter<double>("minSCEtBarrel"),
        .minSCEtEndcaps = pset.getParameter<double>("minSCEtEndcaps"),
        .maxEOverPBarrel = pset.getParameter<double>("maxEOverPBarrel"),
        .maxEOverPEndcaps = pset.getParameter<double>("maxEOverPEndcaps"),
        .minEOverPBarrel = pset.getParameter<double>("minEOverPBarrel"),
        .minEOverPEndcaps = pset.getParameter<double>("minEOverPEndcaps"),
        .maxHOverEBarrelCone = pset.getParameter<double>("maxHOverEBarrelCone"),
        .maxHOverEEndcapsCone = pset.getParameter<double>("maxHOverEEndcapsCone"),
        .maxHBarrelCone = pset.getParameter<double>("maxHBarrelCone"),
        .maxHEndcapsCone = pset.getParameter<double>("maxHEndcapsCone"),
        .maxHOverEBarrelTower = pset.getParameter<double>("maxHOverEBarrelTower"),
        .maxHOverEEndcapsTower = pset.getParameter<double>("maxHOverEEndcapsTower"),
        .maxHBarrelTower = pset.getParameter<double>("maxHBarrelTower"),
        .maxHEndcapsTower = pset.getParameter<double>("maxHEndcapsTower"),
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
        .maxTIP = pset.getParameter<double>("maxTIP"),
        .seedFromTEC = pset.getParameter<bool>("seedFromTEC"),
    };
  }
};  // namespace

GsfElectronProducer::GsfElectronProducer(const edm::ParameterSet& cfg, const GsfElectronAlgo::HeavyObjectCache*)
    : cutsCfg_{makeCutsConfiguration(cfg.getParameter<edm::ParameterSet>("preselection"))},
      ecalSeedingParametersChecked_(false),
      electronPutToken_(produces<GsfElectronCollection>()),
      gsfPfRecTracksTag_(consumes(cfg.getParameter<edm::InputTag>("gsfPfRecTracksTag"))),
      useGsfPfRecTracks_(cfg.getParameter<bool>("useGsfPfRecTracks")),
      resetMvaValuesUsingPFCandidates_(cfg.getParameter<bool>("resetMvaValuesUsingPFCandidates")) {
  if (resetMvaValuesUsingPFCandidates_) {
    egmPFCandidateCollection_ = consumes(cfg.getParameter<edm::InputTag>("egmPFCandidatesTag"));
  }

  inputCfg_.gsfElectronCores = consumes(cfg.getParameter<edm::InputTag>("gsfElectronCoresTag"));
  inputCfg_.hcalTowersTag = consumes(cfg.getParameter<edm::InputTag>("hcalTowers"));
  inputCfg_.barrelRecHitCollection = consumes(cfg.getParameter<edm::InputTag>("barrelRecHitCollectionTag"));
  inputCfg_.endcapRecHitCollection = consumes(cfg.getParameter<edm::InputTag>("endcapRecHitCollectionTag"));
  inputCfg_.ctfTracks = consumes(cfg.getParameter<edm::InputTag>("ctfTracksTag"));
  // used to check config consistency with seeding
  inputCfg_.seedsTag = consumes(cfg.getParameter<edm::InputTag>("seedsTag"));
  inputCfg_.beamSpotTag = consumes(cfg.getParameter<edm::InputTag>("beamSpotTag"));
  inputCfg_.vtxCollectionTag = consumes(cfg.getParameter<edm::InputTag>("vtxTag"));
  if (cfg.getParameter<bool>("fillConvVtxFitProb"))
    inputCfg_.conversions = consumes(cfg.getParameter<edm::InputTag>("conversionsTag"));

  strategyCfg_.useDefaultEnergyCorrection = cfg.getParameter<bool>("useDefaultEnergyCorrection");

  strategyCfg_.applyPreselection = cfg.getParameter<bool>("applyPreselection");
  strategyCfg_.ecalDrivenEcalEnergyFromClassBasedParameterization =
      cfg.getParameter<bool>("ecalDrivenEcalEnergyFromClassBasedParameterization");
  strategyCfg_.ecalDrivenEcalErrorFromClassBasedParameterization =
      cfg.getParameter<bool>("ecalDrivenEcalErrorFromClassBasedParameterization");
  strategyCfg_.pureTrackerDrivenEcalErrorFromSimpleParameterization =
      cfg.getParameter<bool>("pureTrackerDrivenEcalErrorFromSimpleParameterization");
  strategyCfg_.applyAmbResolution = cfg.getParameter<bool>("applyAmbResolution");
  strategyCfg_.ignoreNotPreselected = cfg.getParameter<bool>("ignoreNotPreselected");
  strategyCfg_.ambSortingStrategy = cfg.getParameter<unsigned>("ambSortingStrategy");
  strategyCfg_.ambClustersOverlapStrategy = cfg.getParameter<unsigned>("ambClustersOverlapStrategy");
  strategyCfg_.ctfTracksCheck = cfg.getParameter<bool>("ctfTracksCheck");
  strategyCfg_.PreSelectMVA = cfg.getParameter<double>("PreSelectMVA");
  strategyCfg_.MaxElePtForOnlyMVA = cfg.getParameter<double>("MaxElePtForOnlyMVA");
  strategyCfg_.useEcalRegression = cfg.getParameter<bool>("useEcalRegression");
  strategyCfg_.useCombinationRegression = cfg.getParameter<bool>("useCombinationRegression");
  strategyCfg_.fillConvVtxFitProb = cfg.getParameter<bool>("fillConvVtxFitProb");

  // hcal helpers
  auto const& psetPreselection = cfg.getParameter<edm::ParameterSet>("preselection");
  hcalCfg_.hOverEConeSize = psetPreselection.getParameter<double>("hOverEConeSize");
  if (hcalCfg_.hOverEConeSize > 0) {
    hcalCfg_.useTowers = true;
    hcalCfg_.checkHcalStatus = cfg.getParameter<bool>("checkHcalStatus");
    hcalCfg_.hcalTowers = consumes(cfg.getParameter<edm::InputTag>("hcalTowers"));
    hcalCfg_.hOverEPtMin = psetPreselection.getParameter<double>("hOverEPtMin");
  }

  // Ecal rec hits configuration
  GsfElectronAlgo::EcalRecHitsConfiguration recHitsCfg;
  auto const& flagnamesbarrel = cfg.getParameter<std::vector<std::string>>("recHitFlagsToBeExcludedBarrel");
  recHitsCfg.recHitFlagsToBeExcludedBarrel = StringToEnumValue<EcalRecHit::Flags>(flagnamesbarrel);
  auto const& flagnamesendcaps = cfg.getParameter<std::vector<std::string>>("recHitFlagsToBeExcludedEndcaps");
  recHitsCfg.recHitFlagsToBeExcludedEndcaps = StringToEnumValue<EcalRecHit::Flags>(flagnamesendcaps);
  auto const& severitynamesbarrel = cfg.getParameter<std::vector<std::string>>("recHitSeverityToBeExcludedBarrel");
  recHitsCfg.recHitSeverityToBeExcludedBarrel =
      StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesbarrel);
  auto const& severitynamesendcaps = cfg.getParameter<std::vector<std::string>>("recHitSeverityToBeExcludedEndcaps");
  recHitsCfg.recHitSeverityToBeExcludedEndcaps =
      StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesendcaps);

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
      .useNumCrystals = cfg.getParameter<bool>("useNumCrystals")};

  const RegressionHelper::Configuration regressionCfg{
      .ecalRegressionWeightLabels = cfg.getParameter<std::vector<std::string>>("ecalRefinedRegressionWeightLabels"),
      .ecalWeightsFromDB = cfg.getParameter<bool>("ecalWeightsFromDB"),
      .ecalRegressionWeightFiles = cfg.getParameter<std::vector<std::string>>("ecalRefinedRegressionWeightFiles"),
      .combinationRegressionWeightLabels =
          cfg.getParameter<std::vector<std::string>>("combinationRegressionWeightLabels"),
      .combinationWeightsFromDB = cfg.getParameter<bool>("combinationWeightsFromDB"),
      .combinationRegressionWeightFiles =
          cfg.getParameter<std::vector<std::string>>("combinationRegressionWeightFile")};

  // create algo
  algo_ = std::make_unique<GsfElectronAlgo>(
      inputCfg_,
      strategyCfg_,
      cutsCfg_,
      hcalCfg_,
      isoCfg,
      recHitsCfg,
      EcalClusterFunctionFactory::get()->create(cfg.getParameter<std::string>("crackCorrectionFunction"), cfg),
      regressionCfg,
      cfg.getParameter<edm::ParameterSet>("trkIsol03Cfg"),
      cfg.getParameter<edm::ParameterSet>("trkIsol04Cfg"),
      cfg.getParameter<edm::ParameterSet>("trkIsolHEEP03Cfg"),
      cfg.getParameter<edm::ParameterSet>("trkIsolHEEP04Cfg"),
      consumesCollector());
}

void GsfElectronProducer::checkEcalSeedingParameters(edm::ParameterSet const& pset) {
  if (!pset.exists("SeedConfiguration")) {
    return;
  }
  edm::ParameterSet seedConfiguration = pset.getParameter<edm::ParameterSet>("SeedConfiguration");

  if (seedConfiguration.getParameter<bool>("applyHOverECut")) {
    if ((hcalCfg_.hOverEConeSize != 0) &&
        (hcalCfg_.hOverEConeSize != seedConfiguration.getParameter<double>("hOverEConeSize"))) {
      edm::LogWarning("GsfElectronAlgo|InconsistentParameters")
          << "The H/E cone size (" << hcalCfg_.hOverEConeSize << ") is different from ecal seeding ("
          << seedConfiguration.getParameter<double>("hOverEConeSize") << ").";
    }
    if (cutsCfg_.maxHOverEBarrelCone < seedConfiguration.getParameter<double>("maxHOverEBarrel")) {
      edm::LogWarning("GsfElectronAlgo|InconsistentParameters")
          << "The max barrel cone H/E is lower than during ecal seeding.";
    }
    if (cutsCfg_.maxHOverEEndcapsCone < seedConfiguration.getParameter<double>("maxHOverEEndcaps")) {
      edm::LogWarning("GsfElectronAlgo|InconsistentParameters")
          << "The max endcaps cone H/E is lower than during ecal seeding.";
    }
  }

  if (cutsCfg_.minSCEtBarrel < seedConfiguration.getParameter<double>("SCEtCut")) {
    edm::LogWarning("GsfElectronAlgo|InconsistentParameters")
        << "The minimum super-cluster Et in barrel is lower than during ecal seeding.";
  }
  if (cutsCfg_.minSCEtEndcaps < seedConfiguration.getParameter<double>("SCEtCut")) {
    edm::LogWarning("GsfElectronAlgo|InconsistentParameters")
        << "The minimum super-cluster Et in endcaps is lower than during ecal seeding.";
  }
}

//=======================================================================================
// Ambiguity solving
//=======================================================================================

void GsfElectronProducer::setAmbiguityData(reco::GsfElectronCollection& electrons,
                                           edm::Event const& event,
                                           bool ignoreNotPreselected) const {
  // Getting required event data
  auto const& beamspot = event.get(inputCfg_.beamSpotTag);
  auto gsfPfRecTracks =
      useGsfPfRecTracks_ ? event.getHandle(gsfPfRecTracksTag_) : edm::Handle<reco::GsfPFRecTrackCollection>{};
  auto const& barrelRecHits = event.get(inputCfg_.barrelRecHitCollection);
  auto const& endcapRecHits = event.get(inputCfg_.endcapRecHitCollection);

  if (strategyCfg_.ambSortingStrategy == 0) {
    std::sort(electrons.begin(), electrons.end(), egamma::isBetterElectron);
  } else if (strategyCfg_.ambSortingStrategy == 1) {
    std::sort(electrons.begin(), electrons.end(), egamma::isInnermostElectron);
  } else {
    throw cms::Exception("GsfElectronAlgo|UnknownAmbiguitySortingStrategy")
        << "value of strategyCfg_.ambSortingStrategy is : " << strategyCfg_.ambSortingStrategy;
  }

  // init
  for (auto& electron : electrons) {
    electron.clearAmbiguousGsfTracks();
    electron.setAmbiguous(false);
  }

  // get ambiguous from GsfPfRecTracks
  if (useGsfPfRecTracks_) {
    for (auto& e1 : electrons) {
      bool found = false;
      for (auto const& gsfPfRecTrack : *gsfPfRecTracks) {
        if (gsfPfRecTrack.gsfTrackRef() == e1.gsfTrack()) {
          if (found) {
            edm::LogWarning("GsfElectronAlgo") << "associated gsfPfRecTrack already found";
          } else {
            found = true;
            for (auto const& duplicate : gsfPfRecTrack.convBremGsfPFRecTrackRef()) {
              e1.addAmbiguousGsfTrack(duplicate->gsfTrackRef());
            }
          }
        }
      }
    }
  }
  // or search overlapping clusters
  else {
    for (auto e1 = electrons.begin(); e1 != electrons.end(); ++e1) {
      if (e1->ambiguous())
        continue;
      if (ignoreNotPreselected && !isPreselected(*e1))
        continue;

      SuperClusterRef scRef1 = e1->superCluster();
      CaloClusterPtr eleClu1 = e1->electronCluster();
      LogDebug("GsfElectronAlgo") << "Blessing electron with E/P " << e1->eSuperClusterOverP() << ", cluster "
                                  << scRef1.get() << " & track " << e1->gsfTrack().get();

      for (auto e2 = e1 + 1; e2 != electrons.end(); ++e2) {
        if (e2->ambiguous())
          continue;
        if (ignoreNotPreselected && !isPreselected(*e2))
          continue;

        SuperClusterRef scRef2 = e2->superCluster();
        CaloClusterPtr eleClu2 = e2->electronCluster();

        // search if same cluster
        bool sameCluster = false;
        if (strategyCfg_.ambClustersOverlapStrategy == 0) {
          sameCluster = (scRef1 == scRef2);
        } else if (strategyCfg_.ambClustersOverlapStrategy == 1) {
          float eMin = 1.;
          float threshold = eMin * cosh(EleRelPoint(scRef1->position(), beamspot.position()).eta());
          using egamma::sharedEnergy;
          sameCluster = ((sharedEnergy(*eleClu1, *eleClu2, barrelRecHits, endcapRecHits) >= threshold) ||
                         (sharedEnergy(*scRef1->seed(), *eleClu2, barrelRecHits, endcapRecHits) >= threshold) ||
                         (sharedEnergy(*eleClu1, *scRef2->seed(), barrelRecHits, endcapRecHits) >= threshold) ||
                         (sharedEnergy(*scRef1->seed(), *scRef2->seed(), barrelRecHits, endcapRecHits) >= threshold));
        } else {
          throw cms::Exception("GsfElectronAlgo|UnknownAmbiguityClustersOverlapStrategy")
              << "value of strategyCfg_.ambClustersOverlapStrategy is : " << strategyCfg_.ambClustersOverlapStrategy;
        }

        // main instructions
        if (sameCluster) {
          LogDebug("GsfElectronAlgo") << "Discarding electron with E/P " << e2->eSuperClusterOverP() << ", cluster "
                                      << scRef2.get() << " and track " << e2->gsfTrack().get();
          e1->addAmbiguousGsfTrack(e2->gsfTrack());
          e2->setAmbiguous(true);
        } else if (e1->gsfTrack() == e2->gsfTrack()) {
          edm::LogWarning("GsfElectronAlgo") << "Forgetting electron with E/P " << e2->eSuperClusterOverP()
                                             << ", cluster " << scRef2.get() << " and track " << e2->gsfTrack().get();
          e2->setAmbiguous(true);
        }
      }
    }
  }
}

bool GsfElectronProducer::isPreselected(GsfElectron const& ele) const {
  bool passCutBased = ele.passingCutBasedPreselection();
  bool passPF = ele.passingPflowPreselection();
  // it is worth nothing for gedGsfElectrons, this does nothing as its not set
  // till GedGsfElectron finaliser, this is always false
  bool passmva = ele.passingMvaPreselection();
  if (!ele.ecalDrivenSeed()) {
    if (ele.pt() > strategyCfg_.MaxElePtForOnlyMVA)
      return passmva && passCutBased;
    else
      return passmva;
  } else {
    return passCutBased || passPF || passmva;
  }
}

// ------------ method called to produce the data  ------------
void GsfElectronProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  // check configuration
  if (!ecalSeedingParametersChecked_) {
    ecalSeedingParametersChecked_ = true;
    auto seeds = event.getHandle(inputCfg_.seedsTag);
    if (!seeds.isValid()) {
      edm::LogWarning("GsfElectronAlgo|UnreachableSeedsProvenance")
          << "Cannot check consistency of parameters with ecal seeding ones,"
          << " because the original collection of seeds is not any more available.";
    } else {
      checkEcalSeedingParameters(edm::parameterSet(seeds.provenance()->stable(), event.processHistory()));
    }
  }

  auto electrons = algo_->completeElectrons(event, setup, globalCache());
  if (resetMvaValuesUsingPFCandidates_) {
    const auto gsfMVAInputMap = matchWithPFCandidates(event.get(egmPFCandidateCollection_));
    setMVAOutputs(electrons, globalCache(), event.get(inputCfg_.vtxCollectionTag));
    for (auto& el : electrons)
      el.setMvaInput(gsfMVAInputMap.find(el.gsfTrack())->second);  // set MVA inputs
  }

  // all electrons
  logElectrons(electrons, event, "GsfElectronAlgo Info (before preselection)");
  // preselection
  if (strategyCfg_.applyPreselection) {
    electrons.erase(
        std::remove_if(electrons.begin(), electrons.end(), [this](auto const& ele) { return !isPreselected(ele); }),
        electrons.end());
    logElectrons(electrons, event, "GsfElectronAlgo Info (after preselection)");
  }
  // ambiguity
  setAmbiguityData(electrons, event, strategyCfg_.ignoreNotPreselected);
  if (strategyCfg_.applyAmbResolution) {
    electrons.erase(std::remove_if(electrons.begin(), electrons.end(), std::mem_fn(&reco::GsfElectron::ambiguous)),
                    electrons.end());
    logElectrons(electrons, event, "GsfElectronAlgo Info (after amb. solving)");
  }
  // final filling
  event.emplace(electronPutToken_, std::move(electrons));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GsfElectronProducer);
