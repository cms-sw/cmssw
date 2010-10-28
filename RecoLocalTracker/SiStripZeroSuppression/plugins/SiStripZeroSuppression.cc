#include "RecoLocalTracker/SiStripZeroSuppression/plugins/SiStripZeroSuppression.h"

#include "FWCore/Framework/interface/Event.h"
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
    algorithms(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
    storeCM(conf.getParameter<bool>("storeCM")),
	doAPVRestore(conf.getParameter<bool>("doAPVRestore")),
    produceRawDigis(conf.getParameter<bool>("produceRawDigis")),
    mergeCollections(conf.getParameter<bool>("mergeCollections")),
    fixCM(conf.getParameter<bool>("fixCM"))
	{

  if(mergeCollections){
    storeCM = false;
    produceRawDigis = false;
  }
  
  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ){
    produces< edm::DetSetVector<SiStripDigi> > (inputTag->instance());
    if(produceRawDigis&&!mergeCollections)
      produces< edm::DetSetVector<SiStripRawDigi> > (inputTag->instance());		
  }	

  if(storeCM)
    produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("APVCM");
  

}

void SiStripZeroSuppression::
produce(edm::Event& e, const edm::EventSetup& es) {
  
  algorithms->initialize(es);
  if( doAPVRestore ) algorithms->restorer->LoadMeanCMMap( e );
  
  if(mergeCollections)
    this->CollectionMergedZeroSuppression(e);
  else 
    this->StandardZeroSuppression(e);
  
}

inline void SiStripZeroSuppression::StandardZeroSuppression(edm::Event& e){
	
  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ) {

    edm::Handle< edm::DetSetVector<SiStripRawDigi> > input;
    e.getByLabel(*inputTag,input);

    std::vector<edm::DetSet<SiStripDigi> > output_base; 
    std::vector<edm::DetSet<SiStripRawDigi> > output_base_raw; 
    
    if (input->size()) 
      processRaw(*inputTag, *input, output_base, output_base_raw );
    
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(output_base) );
    e.put( output, inputTag->instance() );
    
	
    if(produceRawDigis){
      std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > outputraw(new edm::DetSetVector<SiStripRawDigi>(output_base_raw) );
      e.put(outputraw, inputTag->instance() );
	   
    }
  }

  if(storeCM){
    std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputAPVCM(new edm::DetSetVector<SiStripProcessedRawDigi>(output_apvcm) );
    e.put( outputAPVCM,"APVCM");
  }  
}


inline void SiStripZeroSuppression::CollectionMergedZeroSuppression(edm::Event& e){

  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ) {

    edm::Handle< edm::DetSetVector<SiStripDigi> > inputdigi;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > inputraw;
    e.getByLabel(*inputTag,inputdigi);
    e.getByLabel(*inputTag,inputraw);
	
    std::vector<edm::DetSet<SiStripDigi> > outputdigi; 
    std::vector<edm::DetSet<SiStripRawDigi> > outputraw;     
	
    if (inputraw->size())	
      processRaw(*inputTag, *inputraw, outputdigi, outputraw );
    
	
    for ( std::vector<edm::DetSet<SiStripDigi> >::const_iterator itinputdigi = inputdigi->begin(); itinputdigi !=inputdigi->end(); ++itinputdigi) {
      outputdigi.push_back(*itinputdigi);	
    }
	
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(outputdigi) );
    e.put( output, inputTag->instance() );
  	
  }

}

inline
void SiStripZeroSuppression::
processRaw(const edm::InputTag& inputTag, const edm::DetSetVector<SiStripRawDigi>& input, std::vector<edm::DetSet<SiStripDigi> >& output, std::vector<edm::DetSet<SiStripRawDigi> >& outputraw) {

  if(storeCM){
    output_apvcm.clear();
    output_apvcm.reserve(16000);
  }
  
  output.reserve(10000);    
  outputraw.reserve(10000);
  for ( edm::DetSetVector<SiStripRawDigi>::const_iterator 
	  rawDigis = input.begin(); rawDigis != input.end(); rawDigis++) {
    
    edm::DetSet<SiStripDigi> suppressedDigis(rawDigis->id);
    
    int16_t nAPVflagged = 0;
	
    if ( "ProcessedRaw" == inputTag.instance()) {
      std::vector<int16_t> processedRawDigis;
      transform(rawDigis->begin(), rawDigis->end(), back_inserter(processedRawDigis), boost::bind(&SiStripRawDigi::adc , _1));
      if( doAPVRestore ) nAPVflagged = algorithms->restorer->inspect( rawDigis->id, processedRawDigis );
      algorithms->subtractorCMN->subtract( rawDigis->id, processedRawDigis);
      if( doAPVRestore ) algorithms->restorer->restore( processedRawDigis, algorithms->subtractorCMN->getAPVsCM() );
      algorithms->suppressor->suppress( processedRawDigis, suppressedDigis );
      
      if(storeCM) this->storeCMN(rawDigis->id);
       
	  
    } else

    if ( "VirginRaw" == inputTag.instance()) {
      std::vector<int16_t> processedRawDigis(rawDigis->size());
      algorithms->subtractorPed->subtract( *rawDigis, processedRawDigis);
      if( doAPVRestore ) nAPVflagged = algorithms->restorer->inspect( rawDigis->id, processedRawDigis );
	  algorithms->subtractorCMN->subtract( rawDigis->id, processedRawDigis);
      if( doAPVRestore ) algorithms->restorer->restore( processedRawDigis, algorithms->subtractorCMN->getAPVsCM() );
      algorithms->suppressor->suppress( processedRawDigis, suppressedDigis );
	  
      if(storeCM) this->storeCMN(rawDigis->id);
	 
    } else 
      
      throw cms::Exception("Unknown input type") 
	<< inputTag.instance() << " unknown.  SiStripZeroZuppression can only process types \"VirginRaw\" and \"ProcessedRaw\" ";
    

    if (suppressedDigis.size() && nAPVflagged==0) 
      output.push_back(suppressedDigis); 
    
    if(produceRawDigis && nAPVflagged > 0) 
	outputraw.push_back(*rawDigis);
      
    }
  
}


inline 
void SiStripZeroSuppression::storeCMN(uint32_t id){
	const std::vector< std::pair<short,float> >& vmedians = algorithms->subtractorCMN->getAPVsCM();
	edm::DetSet<SiStripProcessedRawDigi> apvDetSet(id);
	short apvNb=0;
	for(size_t i=0;i<vmedians.size();++i){
	  if(vmedians[i].first>apvNb){
	    for(int i=0;i<vmedians[i].first-apvNb;++i){
	      apvDetSet.push_back(SiStripProcessedRawDigi(0.));
	      apvNb++;
	    }
	  }
	  apvDetSet.push_back(SiStripProcessedRawDigi(vmedians[i].second));
	  //std::cout << "CM patch in VR " << rawDigis->id << " " << vmedians[i].first << " " << vmedians[i].second << " " << apvNb<< std::endl;
	  apvNb++;
	}
        if(fixCM) algorithms->restorer->fixAPVsCM( apvDetSet ); 
	if(apvDetSet.size())
	  output_apvcm.push_back(apvDetSet);
	
}
