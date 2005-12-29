using namespace std;
#include "EventFilter/HcalRawToDigi/interface/HcalHistogramRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/EDProduct/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include <iostream>


HcalHistogramRawToDigi::HcalHistogramRawToDigi(edm::ParameterSet const& conf):
  unpacker_(conf.getParameter<int>("HcalFirstFED")),
  fedUnpackList_(conf.getParameter<std::vector<int> >("FEDs")),
  firstFED_(conf.getParameter<int>("HcalFirstFED"))
{
  std::cout << "HcalHistogramRawToDigi will unpack FEDs ";
  for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
    std::cout << fedUnpackList_[i] << " ";
  std::cout << std::endl;
    
  // products produced...
  produces<HcalHistogramDigiCollection>();
}

// Virtual destructor needed.
HcalHistogramRawToDigi::~HcalHistogramRawToDigi() { }  

// Functions that gets called by framework every event
void HcalHistogramRawToDigi::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;  
  // edm::ProcessNameSelector s("PROD"); 
  e.getByType(rawraw);           // HACK!
  // get the mapping
  edm::ESHandle<HcalDbService> pSetup;
  es.get<HcalDbRecord>().get( pSetup );
  const HcalElectronicsMap* readoutMap=pSetup->getHcalMapping();

  // Step B: Create empty output
  std::auto_ptr<HcalHistogramDigiCollection> prod(new HcalHistogramDigiCollection());
  std::vector<HcalHistogramDigi> digis;
 
  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw->FEDData(*i);

    unpacker_.unpack(fed,*readoutMap,digis);
  }

  // Step B2: encapsulate vectors in actual collections
  prod->swap_contents(digis);

  // Step D: Put outputs into event
  prod->sort();
  e.put(prod);
}


