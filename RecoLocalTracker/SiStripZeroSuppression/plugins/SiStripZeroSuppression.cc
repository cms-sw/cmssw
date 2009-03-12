#include "RecoLocalTracker/SiStripZeroSuppression/plugins/SiStripZeroSuppression.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include <memory>

SiStripZeroSuppression::
SiStripZeroSuppression(edm::ParameterSet const& conf)
  : inputTags(conf.getParameter<std::vector<edm::InputTag> >("RawDigiProducersList")),
    suppressor(SiStripRawProcessingFactory::create_Suppressor(conf)),
    subtractorCMN(SiStripRawProcessingFactory::create_SubtractorCMN(conf)),
    subtractorPed(SiStripRawProcessingFactory::create_SubtractorPed(conf))  {

  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag )
    produces< edm::DetSetVector<SiStripDigi> > (inputTag->instance());
}

void SiStripZeroSuppression::
produce(edm::Event& e, const edm::EventSetup& es) {

  subtractorPed->init(es);
  subtractorCMN->init(es);
  suppressor->init(es);

  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ) {

    std::vector<edm::DetSet<SiStripDigi> > output_base; 
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > input;
    e.getByLabel(*inputTag,input);

    if (input->size()) 
      processRaw(*inputTag, *input, output_base);

    std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(output_base) );
    e.put( output, inputTag->instance() );
  }
}


inline
void SiStripZeroSuppression::
processRaw(const edm::InputTag& inputTag, const edm::DetSetVector<SiStripRawDigi>& input, std::vector<edm::DetSet<SiStripDigi> >& output) {
  output.reserve(10000);    
  for ( edm::DetSetVector<SiStripRawDigi>::const_iterator 
	  rawDigis = input.begin(); rawDigis != input.end(); rawDigis++) {
    

    edm::DetSet<SiStripDigi> suppressedDigis(rawDigis->id);

    if ( "ProcessedRaw" == inputTag.instance()) 
      suppressor->suppress( *rawDigis, suppressedDigis ); else 

    if ( "VirginRaw" == inputTag.instance()) {
      std::vector<int16_t> processedRawDigis(rawDigis->size());
      subtractorPed->subtract( *rawDigis, processedRawDigis);
      subtractorCMN->subtract( rawDigis->id, processedRawDigis);
      
      suppressor->suppress( processedRawDigis, suppressedDigis );
    }

    else throw cms::Exception("Unknown input type") 
      << inputTag.instance() << " unknown.  SiStripZeroZuppression can only process types \"VirginRaw\" and \"ProcessedRaw\" ";
    
    
    if (suppressedDigis.size()) 
      output.push_back(suppressedDigis); 


  }
}
