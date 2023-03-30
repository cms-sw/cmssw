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

private:
  // inputs
  edm::EDGetTokenT<reco::PFClusterCollection> _clustersLabel;
  // options
  // the actual algorithm
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;
};

DEFINE_FWK_MODULE(PFMultiDepthClusterProducer);

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
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");

  edm::ConsumesCollector&& cc = consumesCollector();
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  produces<reco::PFClusterCollection>();
}

void PFMultiDepthClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  _pfClusterBuilder->update(es);
}

void PFMultiDepthClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  _pfClusterBuilder->reset();

  edm::Handle<reco::PFClusterCollection> inputClusters;
  e.getByToken(_clustersLabel, inputClusters);

  std::vector<bool> seedable;

  auto pfClusters = std::make_unique<reco::PFClusterCollection>();
  _pfClusterBuilder->buildClusters(*inputClusters, seedable, *pfClusters);
  LOGVERB("PFMultiDepthClusterProducer::produce()") << *_pfClusterBuilder;

  if (_energyCorrector) {
    _energyCorrector->correctEnergies(*pfClusters);
  }
  e.put(std::move(pfClusters));
}
