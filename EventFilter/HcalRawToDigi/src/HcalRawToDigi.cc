using namespace std;
#include "EventFilter/HcalRawToDigi/interface/HcalRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/EDProduct/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "CondFormats/HcalMapping/interface/HcalMappingTextFileReader.h"
#include <iostream>
using namespace raw;
namespace cms
{

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
      readoutMap_=cms::hcal::HcalMappingTextFileReader::readFromFile(theFile.c_str(),true); // maintain L2E for no real reason
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
    edm::ProcessNameSelector s("DaqProd"); // HACK!
    e.get(s, rawraw);           

    // Step B: Create empty output  : three vectors for three classes...
    std::auto_ptr<cms::HBHEDigiCollection> hbhe(new cms::HBHEDigiCollection);
    std::auto_ptr<cms::HFDigiCollection> hf(new cms::HFDigiCollection);
    std::auto_ptr<cms::HODigiCollection> ho(new cms::HODigiCollection);
    std::auto_ptr<cms::HcalTrigPrimDigiCollection> htp(new cms::HcalTrigPrimDigiCollection);

    // since the filter might drop all the data frames...
    bool lookedAtHBHE=false, lookedAtHF=false, lookedAtHO=false;

    // Step C: unpack all requested FEDs
    for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {
      const raw::FEDRawData& fed = rawraw->FEDData(*i);
      cms::HcalSubdetector subdet=readoutMap_.majorityDetector(*i-firstFED_);
      //      std::cout << "Processing FED " << *i << std::endl;

      if (subdet==cms::HcalBarrel || subdet==cms::HcalEndcap) {
	unpacker_.unpack(fed,readoutMap_,*hbhe, *htp);
	lookedAtHBHE=true;
      } else if (subdet==cms::HcalOuter) {
	unpacker_.unpack(fed,readoutMap_,*ho, *htp);
	lookedAtHO=true;
      } else if (subdet==cms::HcalForward) {
	unpacker_.unpack(fed,readoutMap_,*hf, *htp);
	lookedAtHF=true;
      }
      // TODO: else complain!
    }

    // Step C2: filter FEDs, if required
    if (filter_.active()) {
      cms::HBHEDigiCollection filtered_hbhe=filter_.filter(*hbhe);
      cms::HODigiCollection filtered_ho=filter_.filter(*ho);
      cms::HFDigiCollection filtered_hf=filter_.filter(*hf);
      
      hbhe->swap(filtered_hbhe);
      ho->swap(filtered_ho);
      hf->swap(filtered_hf);
 
    }

    // Step D: Put outputs into event
    /*
    if (lookedAtHBHE) e.put(hbhe);
    if (lookedAtHO) e.put(ho);
    if (lookedAtHF)
    */
      e.put(hbhe);
      e.put(ho);
      e.put(hf);
    e.put(htp);
  }

}
