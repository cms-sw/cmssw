#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTInputProducer.h" 

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

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

L1RCTInputProducer::L1RCTInputProducer(const edm::ParameterSet& conf) : 
  rctLookupTables(new L1RCTLookupTables),
  rct(new L1RCT(rctLookupTables)),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigisLabel(conf.getParameter<edm::InputTag>("ecalDigisLabel")),
  hcalDigisLabel(conf.getParameter<edm::InputTag>("hcalDigisLabel"))
{
  produces<vector<int> >();
}

L1RCTInputProducer::~L1RCTInputProducer()
{
  if(rct != 0) delete rct;
  if(rctLookupTables != 0) delete rctLookupTables;
}

void L1RCTInputProducer::beginJob(const edm::EventSetup& eventSetup)
{
}

void L1RCTInputProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
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
  
  if (useEcal) { event.getByLabel(ecalDigisLabel, ecal); }
  if (useHcal) { event.getByLabel(hcalDigisLabel, hcal); }

  EcalTrigPrimDigiCollection ecalColl;
  HcalTrigPrimDigiCollection hcalColl;
  if (ecal.isValid()) { ecalColl = *ecal; }
  if (hcal.isValid()) { hcalColl = *hcal; }

  rct->digiInput(ecalColl, hcalColl);

  // Stuff to create

  std::auto_ptr<std::vector<int> > rctLUTOutput(new vector<int>);
  for(int crate = 0; crate < 18; crate++) {
    for(int card = 0; card < 7; card++) {
      for(int tower = 0; tower < 32; tower++) {
	unsigned short ecalCompressedET = rct->ecalCompressedET(crate, card, tower);
	unsigned short ecalFineGrainBit = rct->ecalFineGrainBit(crate, card, tower);
	unsigned short hcalCompressedET = rct->hcalCompressedET(crate, card, tower);
	int lutBits = rctLookupTables->lookup(ecalCompressedET, hcalCompressedET, ecalFineGrainBit, crate, card, tower);
	if(lutBits > 0) {
	  lutBits += ( (crate >> 18) + (card >> 23) + (tower >> 26) );
	  rctLUTOutput->push_back(lutBits);
	}
      }
    }
  }

  std::auto_ptr<std::vector<int> > hfLUTOutput(new vector<int>);
  for(int crate = 0; crate < 18; crate++) {
    for(int hfRegion = 0; hfRegion < 8; hfRegion++) {
      unsigned short hfCompressedET = rct->hfCompressedET(crate, hfRegion);
      int lutBits = rctLookupTables->lookup(hfCompressedET, crate, 999, hfRegion);
      if(lutBits > 0) {
	lutBits += ( (crate >> 16) + (hfRegion >> 24) );
	hfLUTOutput->push_back(lutBits);
      }
    }
  }

  //putting stuff back into event
  event.put(rctLUTOutput);
  event.put(hfLUTOutput);

}
