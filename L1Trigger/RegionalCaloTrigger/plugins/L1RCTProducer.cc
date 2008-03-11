#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h" 

#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) : 
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables)),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigisLabel(conf.getParameter<edm::InputTag>("ecalDigisLabel")),
  hcalDigisLabel(conf.getParameter<edm::InputTag>("hcalDigisLabel")),
  preSamples(conf.getParameter<unsigned>("preSamples")),
  postSamples(conf.getParameter<unsigned>("postSamples"))
{
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();
}

L1RCTProducer::~L1RCTProducer()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
}

void L1RCTProducer::beginJob(const edm::EventSetup& eventSetup)
{
}

void L1RCTProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{

  // Refresh configuration information every event
  // Hopefully, this does not take too much time
  // There should be a call back function in future to
  // handle changes in configuration

  edm::ESHandle<L1RCTParameters> rctParameters;
  eventSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* r = rctParameters.product();
  edm::ESHandle<CaloTPGTranscoder> transcoder;
  eventSetup.get<CaloTPGRecord>().get(transcoder);
  const CaloTPGTranscoder* t = transcoder.product();
  edm::ESHandle<L1CaloEtScale> emScale;
  eventSetup.get<L1EmEtScaleRcd>().get(emScale);
  const L1CaloEtScale* s = emScale.product();
  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setTranscoder(t);
  rctLookupTables->setL1CaloEtScale(s);
  
  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;
  
  EcalTrigPrimDigiCollection::const_iterator ecal_it;
  HcalTrigPrimDigiCollection::const_iterator hcal_it;

  if (useEcal) { event.getByLabel(ecalDigisLabel, ecal); }
  if (useHcal) { event.getByLabel(hcalDigisLabel, hcal); }

  unsigned nSamples = preSamples + postSamples + 1;
  //std::cout << "pre " << preSamples << " post " << postSamples
  //	    << " total " << nSamples;


  std::vector<EcalTrigPrimDigiCollection> ecalColl(nSamples);
  std::vector<HcalTrigPrimDigiCollection> hcalColl(nSamples);
  if (ecal.isValid()) 
    { 
      //ecalColl = *ecal; 
      // loop through all ecal digis
      //std::cout << "size of ecal digi coll is " << ecal->size();
      //for (int i = 0; i = ecal->size(); i++)
      for (ecal_it = ecal->begin(); ecal_it != ecal->end(); ecal_it++)
	{
	  // loop through time samples for each digi
	  //short digiSize = (*ecal)[i].size();
	  unsigned short digiSize = ecal_it->size();
	  // (size of each digi must be no less than nSamples)
	  //std::cout << " size of this digi is " << digiSize;
	  if (digiSize < nSamples)
	    {
	      //throw exception
	      throw cms::Exception("EventCorruption") 
		<< "ECAL data should have at least " << nSamples
		<< " time samples per digi, current digi only has " 
		<< digiSize;
	    }
	  for (unsigned short sample = 0; sample < nSamples; sample++)
	    {
	      // put each time sample into its own digi
	      //short zside = (*ecal)[i].id().zside();
	      //short ietaAbs = (*ecal)[i].id().ietaAbs();
	      //short iphi = (*ecal)[i].id().iphi();
	      short zside = ecal_it->id().zside();
	      unsigned short ietaAbs = ecal_it->id().ietaAbs();
	      short iphi = ecal_it->id().iphi();
	      /*std::cout << " [producer] sample " << sample << " zside "
			<< zside << " ietaAbs " << ietaAbs
			<< " iphi " << iphi << std::endl;
	      */
	      /*EcalTriggerPrimitiveDigi
		ecalDigi(EcalTrigTowerDetId((*ecal)[i].id().zside(), 
					    EcalTriggerTower, 
					    (*ecal)[i].id().ietaAbs(),
					    (*ecal)[i].id().iphi()));
	      */
	      EcalTriggerPrimitiveDigi
		ecalDigi(EcalTrigTowerDetId((int) zside, EcalTriggerTower,
					    (int) ietaAbs, (int) iphi));
	      ecalDigi.setSize(1);
	      //ecalDigi.setSample(0, EcalTriggerPrimitiveSample((*ecal)[i].sample(sample).raw()));
	      ecalDigi.setSample(0, EcalTriggerPrimitiveSample(ecal_it->sample(sample).raw()));
	      // push back each digi into correct "time sample" of coll
	      ecalColl[sample].push_back(ecalDigi);
	    }
	}
    }
  if (hcal.isValid()) 
    { 
      //hcalColl = *hcal; 
      // loop through all hcal digis
      //for (int i = 0; i = (*hcal).size(); i++)
      for (hcal_it = hcal->begin(); hcal_it != hcal->end(); hcal_it++)
	{
	  // loop through time samples for each digi
	  //unsigned short digiSize = (*hcal)[i].size();
	  unsigned short digiSize = hcal_it->size();
	  // (size of each digi must be no less than nSamples)
	  if (digiSize < nSamples)
	    {
	      // throw exception
	      throw cms::Exception("EventCorruption") 
		<< "HCAL data should have at least " << nSamples
		<< " time samples per digi, current digi only has " 
		<< digiSize;
	    }
	  for (unsigned short sample = 0; sample < nSamples; sample++)
	    {
	      // put each (relevant) time sample into its own digi
	      HcalTriggerPrimitiveDigi hcalDigi(HcalTrigTowerDetId
						((int) hcal_it->id().ieta(),
						 (int) hcal_it->id().iphi()));
	      hcalDigi.setSize(1);
	      hcalDigi.setPresamples(0);
	      hcalDigi.setSample(0, HcalTriggerPrimitiveSample
				 (hcal_it->sample(hcal_it->presamples() - preSamples + sample).raw()));
	      hcalColl[sample].push_back(hcalDigi);  
	    }
	}
    }

  //rct->digiInput(ecalColl, hcalColl);

  std::auto_ptr<L1CaloEmCollection> rctEmCands (new L1CaloEmCollection);
  std::auto_ptr<L1CaloRegionCollection> rctRegions (new L1CaloRegionCollection);

  // loop through and process each bx
  for (unsigned short sample = 0; sample < nSamples; sample++)
    {
      rct->digiInput(ecalColl[sample], hcalColl[sample]);
      rct->processEvent();

      // Stuff to create
      
      for (int j = 0; j<18; j++)
	{
	  L1CaloEmCollection isolatedEGObjects = rct->getIsolatedEGObjects(j);
	  L1CaloEmCollection nonisolatedEGObjects = rct->getNonisolatedEGObjects(j);
	  for (int i = 0; i<4; i++) 
	    {
	      isolatedEGObjects.at(i).setBx(sample - preSamples);
	      nonisolatedEGObjects.at(i).setBx(sample - preSamples);
	      rctEmCands->push_back(isolatedEGObjects.at(i));
	      rctEmCands->push_back(nonisolatedEGObjects.at(i));
	    }
	}
      
      
      for (int i = 0; i < 18; i++)
	{
	  vector<L1CaloRegion> regions = rct->getRegions(i);
	  for (int j = 0; j < 22; j++)
	    {
	      regions.at(j).setBx(sample - preSamples);
	      rctRegions->push_back(regions.at(j));
	    }
	}
    }
  
  //putting stuff back into event
  event.put(rctEmCands);
  event.put(rctRegions);

}
