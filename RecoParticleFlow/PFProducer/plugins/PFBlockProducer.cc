#include "RecoParticleFlow/PFProducer/plugins/PFBlockProducer.h"

using namespace std;
using namespace edm;

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig) :
  verbose_{ iConfig.getUntrackedParameter<bool>("verbose",false)},
  putToken_{produces<reco::PFBlockCollection>()}
{
  bool debug_ = 
    iConfig.getUntrackedParameter<bool>("debug",false);  
  pfBlockAlgo_.setDebug(debug_);  
      
  edm::ConsumesCollector coll = consumesCollector();
  const std::vector<edm::ParameterSet>& importers
    = iConfig.getParameterSetVector("elementImporters");      
  pfBlockAlgo_.setImporters(importers,coll);

  const std::vector<edm::ParameterSet>& linkdefs 
    = iConfig.getParameterSetVector("linkDefinitions");
  pfBlockAlgo_.setLinkers(linkdefs);  
  
}



PFBlockProducer::~PFBlockProducer() { }


void PFBlockProducer::
beginLuminosityBlock(edm::LuminosityBlock const& lb, 
		     edm::EventSetup const& es) {
  pfBlockAlgo_.updateEventSetup(es);
}

void 
PFBlockProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) {
    
  pfBlockAlgo_.buildElements(iEvent);
  
  auto blocks = pfBlockAlgo_.findBlocks();
  
  if(verbose_) {
    ostringstream  str;
    str<<pfBlockAlgo_<<endl;
    str<<"number of blocks : "<<blocks.size()<<endl;
    str<<endl;
    
    for(PFBlockAlgo::IBC ib=blocks.begin(); 
	ib != blocks.end(); ++ib) {
      str<<(*ib)<<endl;
    }

    LogInfo("PFBlockProducer") << str.str()<<endl;
  }    

  iEvent.emplace(putToken_,blocks);
    
}
