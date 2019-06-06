#include "PFMultiDepthClusterProducer.h"

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
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = std::unique_ptr<PFCBB>{PFClusterBuilderFactory::get()->create(pfcName, pfcConf)};
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    _energyCorrector =
        std::unique_ptr<PFClusterEnergyCorrectorBase>{PFClusterEnergyCorrectorFactory::get()->create(cName, cConf)};
  }

  produces<reco::PFClusterCollection>();
}

void PFMultiDepthClusterProducer::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
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
