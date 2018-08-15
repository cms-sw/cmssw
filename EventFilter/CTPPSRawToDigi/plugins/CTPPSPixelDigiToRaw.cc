#include "CTPPSPixelDigiToRaw.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelDataFormatter.h"

//raw test
//#include "DataFormats/DetId/interface/DetIdCollection.h"
//#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
//#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelRawToDigi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
using namespace std;

CTPPSPixelDigiToRaw::CTPPSPixelDigiToRaw( const edm::ParameterSet& pset ) :
  config_(pset)
{

  tCTPPSPixelDigi = consumes<edm::DetSetVector<CTPPSPixelDigi> >(config_.getParameter<edm::InputTag>("InputLabel")); 

  // Define EDProduct type
  produces<FEDRawDataCollection>();
  mappingLabel_ = config_.getParameter<std::string> ("mappingLabel"); 
  // start the counters
  eventCounter = 0;
  allDigiCounter = 0;
  allWordCounter = 0;

}

// -----------------------------------------------------------------------------
CTPPSPixelDigiToRaw::~CTPPSPixelDigiToRaw() {
  edm::LogInfo("CTPPSPixelDigiToRaw")  << " CTPPSPixelDigiToRaw destructor!";

}

// -----------------------------------------------------------------------------
/*
void CTPPSPixelDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("RPixDetDigitizer"));
  desc.add<std::string>("mappingLabel","RPix");
  descriptions.add("ctppsPixelRawData", desc);
  //descriptions.add("rawDataCollector", desc);
}
*/
// -----------------------------------------------------------------------------
void CTPPSPixelDigiToRaw::produce( edm::Event& ev,
                              const edm::EventSetup& es)
{
  eventCounter++;
  edm::LogInfo("CTPPSPixelDigiToRaw") << "[CTPPSPixelDigiToRaw::produce] "
                                   << "event number: " << eventCounter;

  edm::Handle< edm::DetSetVector<CTPPSPixelDigi> > digiCollection;
  label_ = config_.getParameter<edm::InputTag>("InputLabel");
  ev.getByToken( tCTPPSPixelDigi, digiCollection);

  CTPPSPixelDataFormatter::RawData rawdata;
  CTPPSPixelDataFormatter::Digis digis;
  typedef vector< edm::DetSet<CTPPSPixelDigi> >::const_iterator DI;

  int digiCounter = 0; 
  for (DI di=digiCollection->begin(); di != digiCollection->end(); di++) {
    digiCounter += (di->data).size(); 
    digis[ di->id] = di->data;
  }
  allDigiCounter += digiCounter;
   edm::ESHandle<CTPPSPixelDAQMapping> mapping;
  if (recordWatcher.check( es )) {
    es.get<CTPPSPixelDAQMappingRcd>().get(mapping);
    for (const auto &p : mapping->ROCMapping)    {

        const uint32_t piD = p.second.iD;  
        short unsigned int pROC   = p.second.roc;
        short unsigned int pFediD = p.first.getFEDId();
        short unsigned int pFedcH = p.first.getChannelIdx();
        short unsigned int pROCcH = p.first.getROC();

	
        std::map<short unsigned int,short unsigned int> mapROCIdCh;
        mapROCIdCh.insert(std::pair<short unsigned int,short unsigned int>(pROC,pROCcH));

        std::map<const uint32_t, std::map<short unsigned int, short unsigned int> > mapDetRoc;
        mapDetRoc.insert(std::pair<const uint32_t, std::map<short unsigned int,short unsigned int> >(piD,mapROCIdCh));
	std::map<short unsigned int,short unsigned int> mapFedIdCh; 
        mapFedIdCh.insert(std::pair<short unsigned int,short unsigned int>(pFediD,pFedcH)); 

        iDdet2fed_.insert(std::pair<std::map<const uint32_t, std::map<short unsigned int, short unsigned int> >, std::map<short unsigned int,short unsigned int> > (mapDetRoc,mapFedIdCh)); 

    }
    fedIds_ = mapping->fedIds();
  }

  CTPPSPixelDataFormatter formatter(mapping->ROCMapping);

  // create product (raw data)
  auto buffers = std::make_unique<FEDRawDataCollection>();

  // convert data to raw
  formatter.formatRawData( ev.id().event(), rawdata, digis, iDdet2fed_);
  
  // pack raw data into collection
  for (auto it = fedIds_.begin(); it != fedIds_.end(); it++) { 
    FEDRawData& fedRawData = buffers->FEDData( *it );
    CTPPSPixelDataFormatter::RawData::iterator fedbuffer = rawdata.find( *it );
    if( fedbuffer != rawdata.end() ) fedRawData = fedbuffer->second;
  }
	allWordCounter += formatter.nWords();

	if (debug) LogDebug("CTPPSPixelDigiToRaw") 
	        << "Words/Digis this ev: "<<digiCounter<<"(fm:"<<formatter.nDigis()<<")/"
        	<<formatter.nWords()
	        <<"  all: "<< allDigiCounter <<"/"<<allWordCounter;

	ev.put(std::move(buffers));
}
