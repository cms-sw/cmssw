#include "RecoCTPPS/PixelLocal/interface/CTPPSPixelRecHitProducer.h"


CTPPSPixelRecHitProducer::CTPPSPixelRecHitProducer(const edm::ParameterSet& conf) :
  param_(conf), cluster2hit_(conf)
{
  src_ = conf.getParameter<edm::InputTag>("RPixClusterTag");
  verbosity_ = conf.getUntrackedParameter<int> ("RPixVerbosity");	 
  tokenCTPPSPixelCluster_ = consumes<edm::DetSetVector<CTPPSPixelCluster> >(src_);
  produces<edm::DetSetVector<CTPPSPixelRecHit> > ();
}

CTPPSPixelRecHitProducer::~CTPPSPixelRecHitProducer() {}

void CTPPSPixelRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions){
  edm::ParameterSetDescription desc;
  desc.addUntracked<int>("RPixVerbosity",0);
  desc.add<edm::InputTag>("RPixClusterTag",edm::InputTag("ctppsPixelClusters"));
  descriptions.add("ctppsPixelRecHits", desc);
}

void CTPPSPixelRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
   
	edm::Handle<edm::DetSetVector<CTPPSPixelCluster> > rpCl;
	iEvent.getByToken(tokenCTPPSPixelCluster_, rpCl);

	edm::DetSetVector<CTPPSPixelRecHit>  output;

// run reconstruction
	if (!rpCl->empty())
	  run(*rpCl, output);

	iEvent.put(std::make_unique<edm::DetSetVector<CTPPSPixelRecHit> >(output));

}

void CTPPSPixelRecHitProducer::run(const edm::DetSetVector<CTPPSPixelCluster> &input, edm::DetSetVector<CTPPSPixelRecHit> &output){

  for (const auto &ds_cluster : input)
    {
      edm::DetSet<CTPPSPixelRecHit> &ds_rechit = output.find_or_insert(ds_cluster.id);

//calculate the cluster parameters and convert it into a rechit 
      cluster2hit_.buildHits(ds_cluster.id, ds_cluster.data, ds_rechit.data);
      
    }
}

DEFINE_FWK_MODULE( CTPPSPixelRecHitProducer);
