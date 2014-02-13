#include "PFClusterProducerNew.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/TopoClusterBuilderFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorFactory.h"

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

using namespace newpf;

PFClusterProducer::PFClusterProducer(const edm::ParameterSet& conf) :
  _prodTopoClusters(conf.getUntrackedParameter<bool>("prodTopoClusters",false))
{
  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource")); 
  //setup rechit cleaners
  const edm::VParameterSet& cleanerConfs = 
    conf.getParameterSetVector("recHitCleaners");
  for( const auto& conf : cleanerConfs ) {
    const std::string& cleanerName = 
      conf.getParameter<std::string>("algoName");
    RHCB* cleaner = RecHitCleanerFactory::get()->create(cleanerName,conf);
    _cleaners.push_back(std::unique_ptr<RHCB>(cleaner));
  }
  // setup seed finding
  const edm::ParameterSet& sfConf = 
    conf.getParameterSet("seedFinder");
  const std::string& sfName = sfConf.getParameter<std::string>("algoName");
  SeedFinderBase* sfb = SeedFinderFactory::get()->create(sfName,sfConf);
  _seedFinder.reset(sfb);
  //setup topo cluster builder
  const edm::ParameterSet& topoConf = 
    conf.getParameterSet("topoClusterBuilder");
  const std::string& topoName = topoConf.getParameter<std::string>("algoName");
  TCBB* topob = TopoClusterBuilderFactory::get()->create(topoName,topoConf);
  _topoBuilder.reset(topob);
  //setup pf cluster builder
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
  PFCBB* pfcb = PFClusterBuilderFactory::get()->create(pfcName,pfcConf);
  _pfClusterBuilder.reset(pfcb);
  //setup (possible) recalcuation of positions
  _positionReCalc.reset(NULL);
  if( conf.exists("positionReCalc") ) {
    const edm::ParameterSet& pConf = conf.getParameterSet("positionReCalc");
    const std::string& pName = pConf.getParameter<std::string>("algoName");
    PosCalc* pcalc = PFCPositionCalculatorFactory::get()->create(pName,pConf);
    _positionReCalc.reset(pcalc);
  }
  
  if( _prodTopoClusters ) {
    produces<reco::PFClusterCollection>("topoClusters");
  }
  produces<reco::PFClusterCollection>();
}

void PFClusterProducer::beginLuminosityBlock(const edm::LuminosityBlock& lumi, 
					     const edm::EventSetup& es) {
  _topoBuilder->update(es);
  _pfClusterBuilder->update(es);
  if( _positionReCalc ) _positionReCalc->update(es);
  
}

void PFClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  _topoBuilder->reset();
  _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel,rechits);  
  reco::PFRecHitRefVector refhits;
  for( unsigned i = 0; i < rechits->size(); ++i ) {
    refhits.push_back(reco::PFRecHitRef(rechits,i));
  }
  
  std::vector<bool> mask(true, refhits.size());
  for( const std::unique_ptr<RecHitCleanerBase>& cleaner : _cleaners ) {
    cleaner->clean(rechits, mask);
  }
  
  std::vector<bool> seedable(false, refhits.size());
  _seedFinder->findSeeds(rechits,mask,seedable);

  std::auto_ptr<reco::PFClusterCollection> topoClusters;
  topoClusters.reset(new reco::PFClusterCollection);
  _topoBuilder->buildTopoClusters(rechits, mask, seedable, *topoClusters);
  LOGVERB("PFClusterProducer::produce()") << *_topoBuilder;

  std::auto_ptr<reco::PFClusterCollection> pfClusters;
  pfClusters.reset(new reco::PFClusterCollection);
  _pfClusterBuilder->buildPFClusters(*topoClusters, seedable, *pfClusters);
  LOGVERB("PFClusterProducer::produce()") << *_pfClusterBuilder;
  
  if( _positionReCalc ) {
    _positionReCalc->calculateAndSetPositions(*pfClusters);
  }
  if( _prodTopoClusters ) e.put(topoClusters,"topo");
  e.put(pfClusters);
}
