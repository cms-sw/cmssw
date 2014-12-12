#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducer.h"

namespace {
  bool sortByDetId(const reco::PFRecHit& a,
		   const reco::PFRecHit& b) {
    return a.detId() < b.detId();
  }
}

PFRecHitProducer:: PFRecHitProducer(const edm::ParameterSet& iConfig):
  _useHitMap(iConfig.getUntrackedParameter<bool>("useHitMap",false))
{

  produces<reco::PFRecHitCollection>();
  produces<reco::PFRecHitCollection>("Cleaned");


  edm::ConsumesCollector iC = consumesCollector();
  std::vector<edm::ParameterSet> creators = iConfig.getParameter<std::vector<edm::ParameterSet> >("producers");
  for (unsigned int i=0;i<creators.size();++i) {
      std::string name = creators.at(i).getParameter<std::string>("name");
      creators_.push_back(std::unique_ptr<PFRecHitCreatorBase>(PFRecHitFactory::get()->create(name,creators.at(i),iC)));
  }


  edm::ParameterSet navSet = iConfig.getParameter<edm::ParameterSet>("navigator");

  navigator_ = std::unique_ptr<PFRecHitNavigatorBase>(PFRecHitNavigationFactory::get()->create(navSet.getParameter<std::string>("name"),navSet));
    
}


 PFRecHitProducer::~ PFRecHitProducer()
{
 }


//
// member functions
//

// ------------ method called to produce the data  ------------
void
 PFRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   std::auto_ptr<reco::PFRecHitCollection> out(new reco::PFRecHitCollection );
   std::auto_ptr<reco::PFRecHitCollection> cleaned(new reco::PFRecHitCollection );

   navigator_->beginEvent(iSetup);

   for (unsigned int i=0;i<creators_.size();++i) {
     creators_.at(i)->importRecHits(out,cleaned,iEvent,iSetup);
   }

   if(!_useHitMap) std::sort(out->begin(),out->end(),sortByDetId);

   reco::PFRecHitCollection& outprod = *out;
   PFRecHitNavigatorBase::DetIdToHitIdx hitmap(outprod.size());
   if( _useHitMap ) {
     for( unsigned i = 0 ; i < outprod.size(); ++i ) {
       hitmap[outprod[i].detId()] = i;
     }
   }

   //create a refprod here
   edm::RefProd<reco::PFRecHitCollection> refProd = 
     iEvent.getRefBeforePut<reco::PFRecHitCollection>();

   for( unsigned int i=0;i<outprod.size();++i) {
     if( _useHitMap ) {
       navigator_->associateNeighbours(outprod[i],out,hitmap,refProd);
     } else {
       navigator_->associateNeighbours(outprod[i],out,refProd);
     }
   }

   iEvent.put(out,"");
   iEvent.put(cleaned,"Cleaned");
   hitmap.clear();
}



void
 PFRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

