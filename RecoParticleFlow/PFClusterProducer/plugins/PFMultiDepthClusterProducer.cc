#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

class PFMultiDepthClusterProducer : public edm::stream::EDProducer<> {
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFMultiDepthClusterProducer(const edm::ParameterSet&);
  ~PFMultiDepthClusterProducer() override = default;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // inputs
  edm::EDGetTokenT<reco::PFClusterCollection> _clustersLabel;
  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  // options
  // the actual algorithm
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;
  bool cutsFromDB;
  HcalPFCuts const* paramPF = nullptr;
};

DEFINE_FWK_MODULE(PFMultiDepthClusterProducer);

void PFMultiDepthClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("clustersSource", {});
  desc.add<edm::ParameterSetDescription>("energyCorrector", {});
  {
    edm::ParameterSetDescription pset0;
    pset0.add<std::string>("algoName", "PFMultiDepthClusterizer");
    {
      edm::ParameterSetDescription pset1;
      pset1.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
      {
        edm::ParameterSetDescription psd;
        psd.add<std::vector<int>>("depths", {});
        psd.add<std::string>("detector", "");
        psd.add<std::vector<double>>("logWeightDenominator", {});
        pset1.addVPSet("logWeightDenominatorByDetector", psd, {});
      }
      pset1.add<double>("minAllowedNormalization", 1e-09);
      pset1.add<double>("minFractionInCalc", 1e-09);
      pset1.add<int>("posCalcNCrystals", -1);
      pset1.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", {});
      pset1.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", {});
      pset0.add<edm::ParameterSetDescription>("allCellsPositionCalc", pset1);
    }
    pset0.add<edm::ParameterSetDescription>("positionCalc", {});
    pset0.add<double>("minFractionToKeep", 1e-07);
    pset0.add<double>("nSigmaEta", 2.0);
    pset0.add<double>("nSigmaPhi", 2.0);
    desc.add<edm::ParameterSetDescription>("pfClusterBuilder", pset0);
  }
  desc.add<edm::ParameterSetDescription>("positionReCalc", {});
  desc.add<bool>("usePFThresholdsFromDB", false);
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

PFMultiDepthClusterProducer::PFMultiDepthClusterProducer(const edm::ParameterSet& conf) {
  _clustersLabel = consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("clustersSource"));
  cutsFromDB = conf.getParameter<bool>("usePFThresholdsFromDB");
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");

  edm::ConsumesCollector&& cc = consumesCollector();

  if (cutsFromDB) {
    hcalCutsToken_ = esConsumes<HcalPFCuts, HcalPFCutsRcd, edm::Transition::BeginRun>(edm::ESInputTag("", "withTopo"));
  }

  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    if (!pfcName.empty())
      _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    if (!cName.empty())
      _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  produces<reco::PFClusterCollection>();
}

void PFMultiDepthClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  if (cutsFromDB) {
    paramPF = &es.getData(hcalCutsToken_);
  }
  _pfClusterBuilder->update(es);
}

void PFMultiDepthClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  _pfClusterBuilder->reset();

  edm::Handle<reco::PFClusterCollection> inputClusters;
  e.getByToken(_clustersLabel, inputClusters);

  std::vector<bool> seedable;

  auto pfClusters = std::make_unique<reco::PFClusterCollection>();
  _pfClusterBuilder->buildClusters(*inputClusters, seedable, *pfClusters, paramPF);
  LOGVERB("PFMultiDepthClusterProducer::produce()") << *_pfClusterBuilder;

  if (_energyCorrector) {
    _energyCorrector->correctEnergies(*pfClusters);
  }
  e.put(std::move(pfClusters));
}
