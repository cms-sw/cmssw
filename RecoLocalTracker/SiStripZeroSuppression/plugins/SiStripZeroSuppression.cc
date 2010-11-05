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
    mergeCollections(conf.getParameter<bool>("mergeCollections"))
{
  
  if(doAPVRestore){
    produceRawDigis=conf.getParameter<bool>("produceRawDigis");
    produceCalculatedBaseline = conf.getParameter<bool>("produceCalculatedBaseline");
    produceBaselinePoints = conf.getParameter<bool>("produceBaselinePoints");
    storeInZScollBadAPV = conf.getParameter<bool>("storeInZScollBadAPV");
    useCMMeanMap =conf.getParameter<bool>("useCMMeanMap");
    fixCM= conf.getParameter<bool>("fixCM"); 
  }
  
  if(mergeCollections){
    storeCM = false;
    produceRawDigis = false;
    mergeCollections =false;
  }
  
  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ){
    produces< edm::DetSetVector<SiStripDigi> > (inputTag->instance());
    if(produceRawDigis)
      produces< edm::DetSetVector<SiStripRawDigi> > (inputTag->instance());
  } 
  
  if(produceCalculatedBaseline) 
    produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("BADAPVBASELINE");
	
  if(produceBaselinePoints) 
    produces< edm::DetSetVector<SiStripDigi> > ("BADAPVBASELINEPOINTS");
	  
  if(storeCM)
    produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("APVCM");
  
  
}

void SiStripZeroSuppression::
produce(edm::Event& e, const edm::EventSetup& es) {
  
  //std::cout << "SiStripZeroSuppression EventN: " <<e.id() << std::endl;
  algorithms->initialize(es);
  if( doAPVRestore && useCMMeanMap) algorithms->restorer->LoadMeanCMMap( e );
  
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

  
  if(produceCalculatedBaseline){
    std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputbaseline(new edm::DetSetVector<SiStripProcessedRawDigi>(output_baseline) );
    e.put(outputbaseline, "BADAPVBASELINE" );
  }
  
  if(produceBaselinePoints){
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > outputbaselinepoints(new edm::DetSetVector<SiStripDigi>(output_baseline_points) );
    e.put(outputbaselinepoints, "BADAPVBASELINEPOINTS" );
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
  
  if(produceCalculatedBaseline){
    output_baseline.clear();
    output_baseline.reserve(16000);
  }
  
  if(produceBaselinePoints){
    output_baseline_points.clear();
    output_baseline_points.reserve(16000);
  }
  
  output.reserve(10000);    
  outputraw.reserve(10000);
  for ( edm::DetSetVector<SiStripRawDigi>::const_iterator 
	  rawDigis = input.begin(); rawDigis != input.end(); ++rawDigis) {
    	
    edm::DetSet<SiStripDigi> suppressedDigis(rawDigis->id);
    
    int16_t nAPVflagged = 0;
	
    if ( "ProcessedRaw" == inputTag.instance()) {
      std::vector<int16_t> processedRawDigis;
      transform(rawDigis->begin(), rawDigis->end(), back_inserter(processedRawDigis), boost::bind(&SiStripRawDigi::adc , _1));
      if( doAPVRestore ) nAPVflagged = algorithms->restorer->inspect( rawDigis->id, processedRawDigis );
      algorithms->subtractorCMN->subtract( rawDigis->id, processedRawDigis);
      if( doAPVRestore ) algorithms->restorer->restore( processedRawDigis, algorithms->subtractorCMN->getAPVsCM() );
      algorithms->suppressor->suppress( processedRawDigis, suppressedDigis );
      
      const std::vector< std::pair<short,float> >& vmedians= algorithms->subtractorCMN->getAPVsCM();
      if(storeCM) this->storeCMN(rawDigis->id, vmedians);
      if(produceCalculatedBaseline&& nAPVflagged > 0){
	std::map< uint16_t, std::vector < int16_t> >& baselinemap = algorithms->restorer->GetBaselineMap();
	this->storeBaseline(rawDigis->id, vmedians, baselinemap);
      }
      // if(produceBaselinePoints&& nAPVflagged > 0){
      //	std::vector< std::map< uint16_t, int16_t> >&  baselinpoints = algorithms->restorer->GetSmoothedPoints();
      //	this->storeBaselinePoints(rawDigis->id, baselinpoints);
      // }
    } else if ( "VirginRaw" == inputTag.instance()) {
      std::vector<int16_t> processedRawDigis(rawDigis->size());
      algorithms->subtractorPed->subtract( *rawDigis, processedRawDigis);
      if( doAPVRestore ) nAPVflagged = algorithms->restorer->inspect( rawDigis->id, processedRawDigis );
      algorithms->subtractorCMN->subtract( rawDigis->id, processedRawDigis);
      if( doAPVRestore ) algorithms->restorer->restore( processedRawDigis, algorithms->subtractorCMN->getAPVsCM() );
      algorithms->suppressor->suppress( processedRawDigis, suppressedDigis );
      
      const std::vector< std::pair<short,float> >& vmedians = algorithms->subtractorCMN->getAPVsCM();
      if(storeCM) this->storeCMN(rawDigis->id, vmedians);
      if(produceCalculatedBaseline&& nAPVflagged > 0){
	std::map< uint16_t, std::vector < int16_t> >& baselinemap = algorithms->restorer->GetBaselineMap();
	this->storeBaseline(rawDigis->id, vmedians, baselinemap);
      }
      // if(produceBaselinePoints&& nAPVflagged > 0){
      //	std::vector< std::map< uint16_t, int16_t> >&  baselinpoints = algorithms->restorer->GetSmoothedPoints();
      //	this->storeBaselinePoints(rawDigis->id, baselinpoints);
      // }
      
    } else 
      
      throw cms::Exception("Unknown input type") 
	<< inputTag.instance() << " unknown.  SiStripZeroZuppression can only process types \"VirginRaw\" and \"ProcessedRaw\" ";
    
    if (suppressedDigis.size() && (storeInZScollBadAPV || nAPVflagged ==0)) 
      output.push_back(suppressedDigis); 
    
    if (produceRawDigis && nAPVflagged > 0) 
      outputraw.push_back(*rawDigis);
    
  }
  
}


inline 
void SiStripZeroSuppression::storeBaseline(uint32_t id, const std::vector< std::pair<short,float> >& vmedians, std::map< uint16_t, std::vector < int16_t> >& baselinemap){
	
  edm::DetSet<SiStripProcessedRawDigi> baselineDetSet(id);
  std::map< uint16_t, std::vector < int16_t> >::iterator itBaselineMap;
  
  for(size_t i=0;i<vmedians.size();++i){
    uint16_t APVn = vmedians[i].first;
    float median = vmedians[i].second;
    itBaselineMap = baselinemap.find(APVn);
    if(itBaselineMap==baselinemap.end()){
      for(size_t strip=0; strip < 128; ++strip)  baselineDetSet.push_back(SiStripProcessedRawDigi(median));
    } else {
      for(size_t strip=0; strip < 128; ++strip) baselineDetSet.push_back(SiStripProcessedRawDigi((itBaselineMap->second)[strip]));
    }
    
  }
  
  if(baselineDetSet.size())
    output_baseline.push_back(baselineDetSet);
  
}

inline
void storeBaselinePoints(uint32_t id, std::vector< std::map< uint16_t, int16_t> >& BasPointVec){

  /*dm::DetSet<SiStripDigi> baspointDetSet(id);
    std::vector< std::map< uint16_t, int16_t> >::iterator itBasPointVec;
    
    for(size_t i=0;i<vmedians.size();++i){
    uint16_t APVn = vmedians[i].first;
    float median = vmedians[i].second;
    itBaselineMap = baselinemap.find(APVn);
    if(itBaselineMap==baselinemap.end()){
    for(size_t strip=0; strip < 128; ++strip)  baselineDetSet.push_back(SiStripProcessedRawDigi(median));
    } else {
    for(size_t strip=0; strip < 128; ++strip) baselineDetSet.push_back(SiStripProcessedRawDigi((itBaselineMap->second)[strip]));
    }
    
    }
    
    if(baselineDetSet.size())
    output_baseline_points.push_back(baselineDetSet);
  */
}

inline 
void SiStripZeroSuppression::storeCMN(uint32_t id, const std::vector< std::pair<short,float> >& vmedians){
	
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
