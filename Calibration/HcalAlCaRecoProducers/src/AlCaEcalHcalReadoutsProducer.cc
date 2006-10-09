#include "Calibration/HcalAlCaRecoProducers/interface/AlCaEcalHcalReadoutsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


AlCaEcalHcalReadoutsProducer::AlCaEcalHcalReadoutsProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<HBHERecHitCollection>("HBHERecHitCollection");
   produces<HORecHitCollection>("HORecHitCollection");
   produces<HFRecHitCollection>("HFRecHitCollection");
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
   edm::Handle<HORecHitCollection> ho;
   edm::Handle<HFRecHitCollection> hf;

   try {
     iEvent.getByType(hbhe);
     } catch ( std::exception& ex ) {
       LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get hbhe product!" << std::endl;
     }

   try {
   iEvent.getByType(ho);
   } catch ( std::exception& ex ) {
     LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get ho product!" << std::endl;
   }

   try {
   iEvent.getByType(hf);
   } catch ( std::exception& ex ) {
     LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get hf product!" << std::endl;
   }
  //Create empty output collections

  std::auto_ptr<HBHERecHitCollection> miniHBHERecHitCollection(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> miniHORecHitCollection(new HORecHitCollection);
  std::auto_ptr<HFRecHitCollection> miniHFRecHitCollection(new HFRecHitCollection);


  const HBHERecHitCollection Hithbhe = *(hbhe.product());
  for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
        {
         miniHBHERecHitCollection->push_back(*hbheItr);
        }
  const HORecHitCollection Hitho = *(ho.product());
  for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
        {
         miniHORecHitCollection->push_back(*hoItr);
        }

  const HFRecHitCollection Hithf = *(hf.product());
  for(HFRecHitCollection::const_iterator hfItr=Hithf.begin(); hfItr!=Hithf.end(); hfItr++)
      {
         miniHFRecHitCollection->push_back(*hfItr);
      }



  //Put selected information in the event
  iEvent.put( miniHBHERecHitCollection, "HBHERecHitCollection");
  iEvent.put( miniHORecHitCollection, "HORecHitCollection");
  iEvent.put( miniHFRecHitCollection, "HFRecHitCollection");
  
  
}
