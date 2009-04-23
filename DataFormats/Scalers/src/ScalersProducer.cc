/*
 *  File: DataFormats/Scalers/src/ScalersProducer.cc
 */

#include "DataFormats/Scalers/interface/ScalersProducer.h"

#include <iostream>
#include <time.h>
#include <fcntl.h>

using namespace edm;


ScalersProducer::ScalersProducer(const edm::ParameterSet & iConfig)
{
verbose_ = iConfig.getUntrackedParameter < bool > ("verbose", false);
//register product
produces <L1AcceptBunchCrossingCollection> ();
produces <L1TriggerScalersCollection> ();
produces <L1TriggerRatesCollection> ();
produces <LumiScalersCollection> ();
previousTrig = NULL;

} 

void ScalersProducer::beginJob(const EventSetup & c)
{
  fileName = "scalers.dat";
  bytes = 1;
  fd = open(fileName, O_RDONLY);
  ev = 0;
  previousTrig = NULL;
}

void ScalersProducer::endJob()
{
  close(fd);
}


ScalersProducer::~ScalersProducer() { }

void ScalersProducer::produce(edm::Event& iEvent, edm::EventSetup const&)
{
  //create empty collection
  
  std::auto_ptr<L1TriggerScalersCollection> l1ScalersCollection( new L1TriggerScalersCollection);
  std::auto_ptr<L1TriggerRatesCollection> l1RatesCollection( new L1TriggerRatesCollection);
  std::auto_ptr<LumiScalersCollection> lumiCollection( new LumiScalersCollection);
  
  if(fd<=0) 
  {
    std::cout << "Problem opening file..." << std::endl;
    return;
  }

  bytes = read(fd,buffer,sizeof(struct ScalersEventRecordRaw_v1));
  ev++;
 
  if(bytes<=0)
  {
    std::cout << "Finished reading file." << std::endl;
    close(fd);
    fd = open(fileName, O_RDONLY);
  } 
  else 
  {
    std::cout << " " << std::endl;
    std::cout << "Reading event " << ev << std::endl;
      
    const L1TriggerScalers *trig = new L1TriggerScalers(buffer);
    if(verbose_) std::cout << *trig;
    l1ScalersCollection->push_back(*trig);
  
    if ( ev > 1 )
    {
      if( previousTrig->orbitNumber() < trig->orbitNumber() ) 
      { 
	L1TriggerRates rates(*previousTrig,*trig); 
	std::cout << std::endl;
	std::cout << rates;
	delete(previousTrig); 
	l1RatesCollection->push_back(rates);
	previousTrig = trig;
      }
    } 
    else 
    {
      previousTrig = trig;
    }
    LumiScalers lum(buffer);
    if(verbose_) std::cout << lum;
    lumiCollection->push_back(lum);
  }     

  //  put into event  
  iEvent.put( l1ScalersCollection);
  iEvent.put( l1RatesCollection);
  iEvent.put( lumiCollection);
}
