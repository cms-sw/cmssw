#include "Calibration/HcalAlCaRecoProducers/interface/AlCaEcalHcalReadoutsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


AlCaEcalHcalReadoutsProducer::AlCaEcalHcalReadoutsProducer(const edm::ParameterSet& iConfig)
{
EcalHcalReadoutsProducer_ = iConfig.getParameter< edm::InputTag > ("EcalHcalReadoutsProducer");

LogDebug("") << "producer: " << EcalHcalReadoutsProducer_.encode() ;

   //register your products
   produces<HBHERecHitCollection>();
}


AlCaEcalHcalReadoutsProducer::~AlCaEcalHcalReadoutsProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaEcalHcalReadoutsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   edm::Handle<HBHERecHitCollection> hbhe;

   try {
   iEvent.getByLabel(hbheLabel_,hbhe);
   } catch ( std::exception& ex ) {
     LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get product!" << std::endl;
   }

  //Create empty output collections

  std::auto_ptr<HBHERecHitCollection> miniHBHERecHitCollection(new HBHERecHitCollection);

  for(HBHERecHitCollection::const_iterator hbheItr = (*hbhe).begin();
      hbheItr != (*hbhe).end(); ++hbheItr)
      {
       miniHBHERecHitCollection->push_back(*hbheItr);
      }


  //Put selected information in the event
  iEvent.put( miniHBHERecHitCollection, "HBHERecHitCollection");
  
  
}
