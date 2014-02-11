#include "PFClusterProducerNew.h"

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
  /*
  for( const std::unique_ptr<RecHitCleanerBase>& cleaner : _cleaners ) {
    cleaner->clean(refhits, mask);
  }
  */
  
  std::auto_ptr<reco::PFClusterCollection> topoClusters;
  _topoBuilder->buildTopoClusters(refhits, mask, *topoClusters);
  LOGVERB("PFClusterProducer::produce()") << *_topoBuilder;

  std::auto_ptr<reco::PFClusterCollection> pfClusters;
  _pfClusterBuilder->buildPFClusters(*topoClusters, *pfClusters);
  LOGVERB("PFClusterProducer::produce()") << *_pfClusterBuilder;
  if( _positionReCalc ) {
    _positionReCalc->calculateAndSetPositions(*pfClusters);
  }

  if( _prodTopoClusters ) e.put(topoClusters,"topo");
  e.put(pfClusters);
}
