using namespace std;
#include "EventFilter/HcalRawToDigi/interface/HcalRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/EDProduct/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "CondFormats/HcalMapping/interface/HcalMappingTextFileReader.h"
#include <iostream>


HcalRawToDigi::HcalRawToDigi(edm::ParameterSet const& conf):
  unpacker_(conf.getParameter<int>("HcalFirstFED"),conf.getParameter<int>("firstSample"),conf.getParameter<int>("lastSample")),
  filter_(conf.getParameter<bool>("FilterDataQuality"),conf.getParameter<bool>("FilterDataQuality"),
	  conf.getParameter<bool>("FilterAmplitude"),
	  conf.getParameter<int>("FilterAmpBegin"),conf.getParameter<int>("FilterAmpEnd"),
	  conf.getParameter<double>("FilterAmpLevel")),
  readoutMapSource_(conf.getParameter<std::string>("readoutMapSource")),
  fedUnpackList_(conf.getParameter<std::vector<int> >("FEDs")),
  firstFED_(conf.getParameter<int>("HcalFirstFED"))
{
  // load the readout map from a file, if desired.
  const std::string filePrefix("file://");
  if (readoutMapSource_.find(filePrefix)==0) {
    std::string theFile=readoutMapSource_;
    theFile.erase(0,filePrefix.length());
    std::cout << "Reading HcalMapping from '" << theFile << "'\n";
    readoutMap_=HcalMappingTextFileReader::readFromFile(theFile.c_str(),true); // maintain L2E for no real reason
  }
  std::cout << "HcalRawToDigi will unpack FEDs ";
  for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
    std::cout << fedUnpackList_[i] << " ";
  std::cout << std::endl;
    
  // products produced...
  produces<HBHEDigiCollection>();
  produces<HFDigiCollection>();
  produces<HODigiCollection>();
  produces<HcalTrigPrimDigiCollection>();
}

// Virtual destructor needed.
HcalRawToDigi::~HcalRawToDigi() { }  

// Functions that gets called by framework every event
void HcalRawToDigi::produce(edm::Event& e, const edm::EventSetup&)
{
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;  
  // edm::ProcessNameSelector s("PROD"); 
  e.getByType(rawraw);           // HACK!
  
  // Step B: Create empty output  : three vectors for three classes...
  std::vector<HBHEDataFrame> hbhe;
  std::vector<HODataFrame> ho;
  std::vector<HFDataFrame> hf;
  std::vector<HcalTriggerPrimitiveDigi> htp;
 
  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw->FEDData(*i);
    //      std::cout << "Processing FED " << *i << std::endl;
    // look only at the potential ones, to save time.
    if (readoutMap_->subdetectorPresent(HcalBarrel,*i-firstFED_) || readoutMap_->subdetectorPresent(HcalEndcap,*i-firstFED_)) 
      unpacker_.unpack(fed,*readoutMap_,hbhe, htp);   
    if (readoutMap_->subdetectorPresent(HcalOuter,*i-firstFED_)) 
      unpacker_.unpack(fed,*readoutMap_,ho, htp);
    if (readoutMap_->subdetectorPresent(HcalForward,*i-firstFED_)) 
      unpacker_.unpack(fed,*readoutMap_,hf, htp);
  }

  // Step B: encapsulate vectors in actual collections
  std::auto_ptr<HBHEDigiCollection> hbhe_prod(new HBHEDigiCollection()); 
  std::auto_ptr<HFDigiCollection> hf_prod(new HFDigiCollection());
  std::auto_ptr<HODigiCollection> ho_prod(new HODigiCollection());
  std::auto_ptr<HcalTrigPrimDigiCollection> htp_prod(new HcalTrigPrimDigiCollection());  

  hbhe_prod->swap_contents(hbhe);
  hf_prod->swap_contents(hf);
  ho_prod->swap_contents(ho);
  htp_prod->swap_contents(htp);

  // Step C2: filter FEDs, if required
  if (filter_.active()) {
    HBHEDigiCollection filtered_hbhe=filter_.filter(*hbhe_prod);
    HODigiCollection filtered_ho=filter_.filter(*ho_prod);
    HFDigiCollection filtered_hf=filter_.filter(*hf_prod);
    
    hbhe_prod->swap(filtered_hbhe);
    ho_prod->swap(filtered_ho);
    hf_prod->swap(filtered_hf);    
  }


  // Step D: Put outputs into event
  // just until the sorting is proven
  hbhe_prod->sort();
  ho_prod->sort();
  hf_prod->sort();
  htp_prod->sort();

  e.put(hbhe_prod);
  e.put(ho_prod);
  e.put(hf_prod);
  e.put(htp_prod);
}


