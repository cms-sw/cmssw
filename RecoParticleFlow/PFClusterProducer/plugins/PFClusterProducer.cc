#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"
#include <memory>

class PFClusterProducer : public edm::stream::EDProducer<> {
  typedef RecHitTopologicalCleanerBase RHCB;
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFClusterProducer(const edm::ParameterSet&);
  ~PFClusterProducer() override = default;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // inputs
  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;
  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  bool cutsFromDB;
  HcalPFCuts const* paramPF = nullptr;

  // options
  const bool _prodInitClusters;
  // the actual algorithm
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase>> _cleaners;
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase>> _seedcleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<InitialClusteringStepBase> _initialClustering;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFClusterProducer);

void PFClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsSource", {});
  desc.add<bool>("usePFThresholdsFromDB", false);
  {
    edm::ParameterSetDescription psd;
    psd.add<std::string>("algoName", "");
    desc.addVPSet("recHitCleaners", psd, {});
  }
  {
    edm::ParameterSetDescription psd;
    psd.add<std::string>("algoName", "");
    psd.add<std::vector<std::string>>("RecHitFlagsToBeExcluded", {});
    desc.addVPSet("seedCleaners", psd, {});
  }
  {
    edm::ParameterSetDescription pset;
    pset.add<std::string>("algoName", "");
    pset.add<int>("nNeighbours", 0);
    {
      edm::ParameterSetDescription psd;
      psd.add<std::string>("detector", "");
      psd.addNode((edm::ParameterDescription<double>("seedingThreshold", 0, true) and
                   edm::ParameterDescription<double>("seedingThresholdPt", 0, true)) xor
                  (edm::ParameterDescription<std::vector<int>>("depths", {}, true) and
                   edm::ParameterDescription<std::vector<double>>("seedingThreshold", {}, true) and
                   edm::ParameterDescription<std::vector<double>>("seedingThresholdPt", {}, true)));
      pset.addVPSet("thresholdsByDetector", psd, {});
    }
    desc.add<edm::ParameterSetDescription>("seedFinder", pset);
  }
  {
    edm::ParameterSetDescription pset;
    pset.add<std::string>("algoName", "");
    {
      edm::ParameterSetDescription psd;
      psd.add<std::string>("detector", "");
      psd.addNode((edm::ParameterDescription<double>("gatheringThreshold", 0, true) and
                   edm::ParameterDescription<double>("gatheringThresholdPt", 0, true)) xor
                  (edm::ParameterDescription<std::vector<int>>("depths", {}, true) and
                   edm::ParameterDescription<std::vector<double>>("gatheringThreshold", {}, true) and
                   edm::ParameterDescription<std::vector<double>>("gatheringThresholdPt", {}, true)));
      pset.addVPSet("thresholdsByDetector", psd, {});
    }
    pset.add<bool>("useCornerCells", false);
    pset.add<edm::InputTag>("clusterSrc", {});
    pset.add<bool>("filterByTracksterIteration", false);
    pset.add<bool>("filterByTracksterPID", false);
    pset.add<std::vector<int>>("filter_on_categories", {});
    pset.add<std::vector<int>>("filter_on_iterations", {});
    pset.add<double>("pid_threshold", 0);
    pset.add<edm::InputTag>("tracksterSrc", {});
    pset.add<double>("exclusiveFraction", 0);
    pset.add<double>("invisibleFraction", 0);
    pset.add<bool>("maxDistanceFilter", false);
    pset.add<double>("maxDistance", 0);
    pset.add<double>("maxDforTimingSquared", 0);
    pset.add<double>("timeOffset", 0);
    pset.add<uint32_t>("minNHitsforTiming", 0);
    pset.add<bool>("useMCFractionsForExclEnergy", false);
    pset.add<std::vector<double>>("hadronCalib", {});
    pset.add<std::vector<double>>("egammaCalib", {});
    pset.add<double>("calibMinEta", 0);
    pset.add<double>("calibMaxEta", 0);
    pset.add<edm::InputTag>("simClusterSrc", {});
    desc.add<edm::ParameterSetDescription>("initialClusteringStep", pset);
  }
  {
    edm::ParameterSetDescription pset;
    pset.add<std::string>("algoName", "");
    {
      edm::ParameterSetDescription pset2;
      pset2.add<std::string>("algoName", "");
      pset2.add<double>("minFractionInCalc", 0);
      pset2.add<int>("posCalcNCrystals", -1);
      {
        edm::ParameterSetDescription psd;
        psd.add<std::string>("detector", "");
        psd.add<std::vector<int>>("depths", {});
        psd.add<std::vector<double>>("logWeightDenominator", {});
        pset2.addVPSet("logWeightDenominatorByDetector", psd, {});
      }
      pset2.add<double>("logWeightDenominator", 0);
      pset2.add<double>("minAllowedNormalization", 0);
      {
        edm::ParameterSetDescription pset3;
        pset3.add<double>("constantTerm", 0);
        pset3.add<double>("constantTermLowE", 0);
        pset3.add<double>("corrTermLowE", 0);
        pset3.add<double>("noiseTerm", 0);
        pset3.add<double>("noiseTermLowE", 0);
        pset3.add<double>("threshHighE", -1.);
        pset3.add<double>("threshLowE", -1.);
        pset2.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", pset3);
        pset2.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", pset3);
      }
      pset.add<edm::ParameterSetDescription>("allCellsPositionCalc", pset2);
      pset.add<edm::ParameterSetDescription>("positionCalc", pset2);
    }
    pset.add<double>("minFractionToKeep", 0);
    pset.add<double>("nSigmaEta", 0);
    pset.add<double>("nSigmaPhi", 0);
    pset.add<bool>("excludeOtherSeeds", false);
    pset.add<uint32_t>("maxIterations", 0);
    pset.add<double>("minFracTot", 0);
    {
      edm::ParameterSetDescription pset2;
      pset2.add<std::string>("algoName", "");
      pset2.add<double>("minFractionInCalc", 0);
      pset2.add<double>("T0_EB", 0);
      pset2.add<double>("T0_EE", 0);
      pset2.add<double>("T0_ES", 0);
      pset2.add<double>("W0", 0);
      pset2.add<double>("X0", 0);
      pset2.add<double>("minAllowedNormalization", 0);
      pset2.add<edm::ParameterSetDescription>("timeResolutionCalc", {});
      pset.add<edm::ParameterSetDescription>("positionCalcForConvergence", pset2);
    }
    {
      edm::ParameterSetDescription psd;
      psd.add<std::string>("detector", "");
      psd.addNode(edm::ParameterDescription<double>("recHitEnergyNorm", 0, true) xor
                  (edm::ParameterDescription<std::vector<int>>("depths", {}, true) and
                   edm::ParameterDescription<std::vector<double>>("recHitEnergyNorm", {}, true)));
      pset.addVPSet("recHitEnergyNorms", psd, {});
    }
    pset.add<double>("showerSigma", 1.5);
    pset.add<double>("stoppingTolerance", 1e-08);
    pset.add<bool>("clusterTimeResFromSeed", false);
    pset.add<double>("maxNSigmaTime", 10.0);
    pset.add<double>("minChi2Prob", 0);
    {
      edm::ParameterSetDescription pset2;
      pset2.add<double>("constantTerm", 0);
      pset2.add<double>("constantTermLowE", 0);
      pset2.add<double>("corrTermLowE", 0);
      pset2.add<double>("noiseTerm", 0);
      pset2.add<double>("noiseTermLowE", 0);
      pset2.add<double>("threshHighE", -1.);
      pset2.add<double>("threshLowE", -1.);
      pset.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", pset2);
      pset.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", pset2);
    }
    pset.add<double>("timeSigmaEB", 10.0);
    pset.add<double>("timeSigmaEE", 10.0);
    desc.add<edm::ParameterSetDescription>("pfClusterBuilder", pset);
  }
  {
    edm::ParameterSetDescription pset;
    pset.add<std::string>("algoName", "");
    pset.add<double>("minFractionInCalc", 0);
    pset.add<bool>("updateTiming", false);
    pset.add<double>("T0_EB", 0);
    pset.add<double>("T0_EE", 0);
    pset.add<double>("T0_ES", 0);
    pset.add<double>("W0", 0);
    pset.add<double>("X0", 0);
    pset.add<double>("minAllowedNormalization", 0);
    pset.add<edm::ParameterSetDescription>("timeResolutionCalc", {});
    desc.add<edm::ParameterSetDescription>("positionReCalc", pset);
  }
  desc.add<edm::ParameterSetDescription>("energyCorrector", {});
  descriptions.addWithDefaultLabel(desc);
}

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

PFClusterProducer::PFClusterProducer(const edm::ParameterSet& conf)
    : _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)) {
  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"));
  cutsFromDB = conf.getParameter<bool>("usePFThresholdsFromDB");
  edm::ConsumesCollector cc = consumesCollector();

  if (cutsFromDB) {
    hcalCutsToken_ = esConsumes<HcalPFCuts, HcalPFCutsRcd, edm::Transition::BeginRun>(edm::ESInputTag("", "withTopo"));
  }

  //setup rechit cleaners
  const edm::VParameterSet& cleanerConfs = conf.getParameterSetVector("recHitCleaners");
  for (const auto& conf : cleanerConfs) {
    const std::string& cleanerName = conf.getParameter<std::string>("algoName");
    _cleaners.emplace_back(RecHitTopologicalCleanerFactory::get()->create(cleanerName, conf, cc));
  }

  const auto& seedcleanerConfs = conf.getParameterSetVector("seedCleaners");
  if (!seedcleanerConfs.empty()) {
    for (const auto& conf : seedcleanerConfs) {
      const std::string& seedcleanerName = conf.getParameter<std::string>("algoName");
      _seedcleaners.emplace_back(RecHitTopologicalCleanerFactory::get()->create(seedcleanerName, conf, cc));
    }
  }

  // setup seed finding
  const edm::ParameterSet& sfConf = conf.getParameterSet("seedFinder");
  const std::string& sfName = sfConf.getParameter<std::string>("algoName");
  _seedFinder = SeedFinderFactory::get()->create(sfName, sfConf);
  //setup topo cluster builder
  const edm::ParameterSet& initConf = conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, cc);
  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    if (!pfcName.empty())
      _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
  }
  //setup (possible) recalcuation of positions
  const edm::ParameterSet& pConf = conf.getParameterSet("positionReCalc");
  if (!pConf.empty()) {
    const std::string& pName = pConf.getParameter<std::string>("algoName");
    if (!pName.empty())
      _positionReCalc = PFCPositionCalculatorFactory::get()->create(pName, pConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    if (!cName.empty())
      _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();
}

void PFClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  if (cutsFromDB) {
    paramPF = &es.getData(hcalCutsToken_);
  }
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
  for (const auto& cleaner : _cleaners)
    cleaner->update(es);
  for (const auto& cleaner : _seedcleaners)
    cleaner->update(es);
}

void PFClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel, rechits);

  _initialClustering->updateEvent(e);

  std::vector<bool> mask(rechits->size(), true);
  for (const auto& cleaner : _cleaners) {
    cleaner->clean(rechits, mask);
  }

  // no seeding on these hits
  std::vector<bool> seedmask = mask;
  for (const auto& cleaner : _seedcleaners) {
    cleaner->clean(rechits, seedmask);
  }

  std::vector<bool> seedable(rechits->size(), false);
  _seedFinder->findSeeds(rechits, seedmask, seedable, paramPF);

  auto initialClusters = std::make_unique<reco::PFClusterCollection>();
  _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters, paramPF);
  LOGVERB("PFClusterProducer::produce()") << *_initialClustering;

  auto pfClusters = std::make_unique<reco::PFClusterCollection>();
  pfClusters = std::make_unique<reco::PFClusterCollection>();
  if (_pfClusterBuilder) {  // if we've defined a re-clustering step execute it
    _pfClusterBuilder->buildClusters(*initialClusters, seedable, *pfClusters, paramPF);
    LOGVERB("PFClusterProducer::produce()") << *_pfClusterBuilder;
  } else {
    pfClusters->insert(pfClusters->end(), initialClusters->begin(), initialClusters->end());
  }

  if (_positionReCalc) {
    _positionReCalc->calculateAndSetPositions(*pfClusters, paramPF);
  }

  if (_energyCorrector) {
    _energyCorrector->correctEnergies(*pfClusters);
  }

  if (_prodInitClusters)
    e.put(std::move(initialClusters), "initialClusters");
  e.put(std::move(pfClusters));
}
