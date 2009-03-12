#include "AnalysisAlgos/SiStripClusterInfoProducer/plugins/SiStripProcessedRawDigiProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripProcessedRawDigi.h"

#include <functional>

SiStripProcessedRawDigiProducer::SiStripProcessedRawDigiProducer(edm::ParameterSet const& conf) 
  : inputTags(conf.getParameter<std::vector<edm::InputTag> >("DigiProducersList")),
    subtractorPed(SiStripRawProcessingFactory::create_SubtractorPed(conf)),
    subtractorCMN(SiStripRawProcessingFactory::create_SubtractorCMN(conf)){

  produces< edm::DetSetVector<SiStripProcessedRawDigi> >("");
}

void SiStripProcessedRawDigiProducer::
produce(edm::Event& e, const edm::EventSetup& es) {

  std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > output(new edm::DetSetVector<SiStripProcessedRawDigi>());
  edm::Handle< edm::DetSetVector<SiStripDigi> > inputDigis; 
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > inputRawdigis; 

  subtractorPed->init(es);
  subtractorCMN->init(es);
  es.get<SiStripGainRcd>().get(gainHandle);
  
  for( std::vector<edm::InputTag>::const_iterator 
	 inputTag = inputTags.begin(); inputTag < inputTags.end(); ++inputTag ) {

    if( "ZeroSuppressed" == inputTag->instance() ) {
      e.getByLabel(*inputTag, inputDigis);
      if(inputDigis->size()) zs_process(*inputDigis, *output);
    }

    else if( "ProcessedRaw" == inputTag->instance() ) {
      e.getByLabel(*inputTag, inputRawdigis); 
      if(inputRawdigis->size()) pr_process(*inputRawdigis, *output);
    }

    else if( "VirginRaw" == inputTag->instance() || "ScopeMode" == inputTag->instance() ) {
      e.getByLabel(*inputTag, inputRawdigis); 
      if(inputRawdigis->size()) {
	vr_process(*inputRawdigis, *output); 
      }
    }

    else 
      edm::LogError("Unknown DigiProducer") << *inputTag;
  }

  e.put(output);
}


void SiStripProcessedRawDigiProducer::
zs_process(const edm::DetSetVector<SiStripDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripDigi>::const_iterator DSV_it=input.begin(); DSV_it!=input.end(); DSV_it++)  {
    std::vector<float> digis;
    for(edm::DetSet<SiStripDigi>::const_iterator it=DSV_it->begin(); it!=DSV_it->end(); it++) {
      if(it->strip() + unsigned(1) > digis.size() ) { digis.resize(it->strip()+1, float(0.0)); }
      digis.at(it->strip())= static_cast<float>(it->adc());
    }
    common_process( DSV_it->id, digis, output);
  }
}

void SiStripProcessedRawDigiProducer::
pr_process(const edm::DetSetVector<SiStripRawDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripRawDigi>::const_iterator DSV_it=input.begin(); DSV_it!=input.end(); DSV_it++) {
    std::vector<float> digis;
    transform(DSV_it->begin(), DSV_it->end(), back_inserter(digis), boost::bind(&SiStripRawDigi::adc , _1));
    common_process( DSV_it->id, digis, output);
  }
}

void SiStripProcessedRawDigiProducer::
vr_process(const edm::DetSetVector<SiStripRawDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripRawDigi>::const_iterator DSV_it=input.begin(); DSV_it!=input.end(); DSV_it++) {
    std::vector<int16_t> int_digis(DSV_it->size());
    subtractorPed->subtract(*DSV_it,int_digis);
    subtractorCMN->subtract(DSV_it->id,int_digis);
    std::vector<float> digis(int_digis.begin(), int_digis.end());
    common_process( DSV_it->id, digis, output);
  }
}

void SiStripProcessedRawDigiProducer::
common_process(const uint32_t detId, std::vector<float> & digis, edm::DetSetVector<SiStripProcessedRawDigi>& output) {

  //Apply Gains
  SiStripApvGain::Range detGainRange =  gainHandle->getRange(detId);   
  for(std::vector<float>::iterator it=digis.begin(); it<digis.end(); it++) 
    (*it)/= (gainHandle->getStripGain(it-digis.begin(), detGainRange));

  //Insert as DetSet
  edm::DetSet<SiStripProcessedRawDigi> ds(detId);
  copy(digis.begin(), digis.end(), back_inserter(ds.data) ); 
  output.insert(ds);
}
