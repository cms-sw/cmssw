#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersProducer.h"

ScalersProducer::ScalersProducer(const edm::ParameterSet & iConfig)
{
verbose_ = iConfig.getUntrackedParameter < bool > ("verbose", false);
//register product
produces <L1TriggerScalersCollection> ();
produces <L1TriggerRatesCollection> ();
produces <LumiScalersCollection> ();
previousTrig = NULL;

} 


ScalersProducer::~ScalersProducer()
{
}

void ScalersProducer::produce(edm::Event& iEvent, edm::EventSetup const&)
{
  //create empty collection
  
  std::auto_ptr<L1TriggerScalersCollection> l1ScalersCollection( new L1TriggerScalersCollection);
  std::auto_ptr<L1TriggerRatesCollection> l1RatesCollection( new L1TriggerRatesCollection);
  std::auto_ptr<LumiScalersCollection> lumiCollection( new LumiScalersCollection);
  

  L1TriggerScalers *trig = new L1TriggerScalers(buffer);
  if(verbose_) std::cout << *trig;
  l1ScalersCollection->push_back(*trig);
  
  if( previousTrig->orbitNumber() < trig->orbitNumber() ) 
    { 
      L1TriggerRates rates(*previousTrig,*trig); 
      std::cout << std::endl;
      std::cout << rates;
      delete(previousTrig); 
      previousTrig = trig;

      l1RatesCollection->push_back(rates);

    }

  LumiScalers lum(buffer);
  if(verbose_) std::cout << lum;
  lumiCollection->push_back(lum);


//  put into event  
  iEvent.put( l1ScalersCollection);
  iEvent.put( l1RatesCollection);
  iEvent.put( lumiCollection);
  
    
}
