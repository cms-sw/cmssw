#include "PFClusterProducer.h"

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

PFClusterProducer::PFClusterProducer(const edm::ParameterSet& conf) :
  _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters",false))
{
  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource")); 
  //setup rechit cleaners
  const edm::VParameterSet& cleanerConfs = 
    conf.getParameterSetVector("recHitCleaners");
  for( const auto& conf : cleanerConfs ) {
    const std::string& cleanerName = 
      conf.getParameter<std::string>("algoName");
    RHCB* cleaner = 
      RecHitTopologicalCleanerFactory::get()->create(cleanerName,conf);
    _cleaners.push_back(std::unique_ptr<RHCB>(cleaner));
  }
  // setup seed finding
  const edm::ParameterSet& sfConf = 
    conf.getParameterSet("seedFinder");
  const std::string& sfName = sfConf.getParameter<std::string>("algoName");
  SeedFinderBase* sfb = SeedFinderFactory::get()->create(sfName,sfConf);
  _seedFinder.reset(sfb);
  //setup topo cluster builder
  const edm::ParameterSet& initConf = 
    conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  ICSB* initb = InitialClusteringStepFactory::get()->create(initName,initConf);
  _initialClustering.reset(initb);
  //setup pf cluster builder if requested
  _pfClusterBuilder.reset(NULL);
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if( !pfcConf.empty() ) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    PFCBB* pfcb = PFClusterBuilderFactory::get()->create(pfcName,pfcConf);
    _pfClusterBuilder.reset(pfcb);
  }
  //setup (possible) recalcuation of positions
  _positionReCalc.reset(NULL);
  const edm::ParameterSet& pConf = conf.getParameterSet("positionReCalc");
  if( !pConf.empty() ) {
    const std::string& pName = pConf.getParameter<std::string>("algoName");
    PosCalc* pcalc = PFCPositionCalculatorFactory::get()->create(pName,pConf);
    _positionReCalc.reset(pcalc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf =  conf.getParameterSet("energyCorrector");
  if( !cConf.empty() ) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    PFClusterEnergyCorrectorBase* eCorr =
      PFClusterEnergyCorrectorFactory::get()->create(cName,cConf);
    _energyCorrector.reset(eCorr);
  }
  

  if( _prodInitClusters ) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();
}

void PFClusterProducer::beginLuminosityBlock(const edm::LuminosityBlock& lumi, 
					     const edm::EventSetup& es) {
  _initialClustering->update(es);
  _pfClusterBuilder->update(es);
  if( _positionReCalc ) _positionReCalc->update(es);
  
}

void PFClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  _initialClustering->reset();
  _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel,rechits);  
  
  std::vector<bool> mask(rechits->size(),true);
  for( const auto& cleaner : _cleaners ) {
    cleaner->clean(rechits, mask);
  }

  std::vector<bool> seedable(rechits->size(),false);
  _seedFinder->findSeeds(rechits,mask,seedable);

  std::auto_ptr<reco::PFClusterCollection> initialClusters;
  initialClusters.reset(new reco::PFClusterCollection);
  _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters);
  LOGVERB("PFClusterProducer::produce()") << *_initialClustering;

  std::auto_ptr<reco::PFClusterCollection> pfClusters;
  pfClusters.reset(new reco::PFClusterCollection);
  if( _pfClusterBuilder ) { // if we've defined a re-clustering step execute it
    _pfClusterBuilder->buildClusters(*initialClusters, seedable, *pfClusters);
    LOGVERB("PFClusterProducer::produce()") << *_pfClusterBuilder;
  } else {
    pfClusters->insert(pfClusters->end(),
		       initialClusters->begin(),initialClusters->end());
  }
  
  if( _positionReCalc ) {
    _positionReCalc->calculateAndSetPositions(*pfClusters);
  }

  if( _energyCorrector ) {
    _energyCorrector->correctEnergies(*pfClusters);
  }

  if( _prodInitClusters ) e.put(initialClusters,"initialClusters");
  e.put(pfClusters);
}
