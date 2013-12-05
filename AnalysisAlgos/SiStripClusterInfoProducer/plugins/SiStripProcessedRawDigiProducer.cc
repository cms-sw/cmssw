#include "AnalysisAlgos/SiStripClusterInfoProducer/plugins/SiStripProcessedRawDigiProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/transform.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"

#include <functional>

SiStripProcessedRawDigiProducer::SiStripProcessedRawDigiProducer(edm::ParameterSet const& conf)
  : inputTags(conf.getParameter<std::vector<edm::InputTag> >("DigiProducersList")),
    inputTokensDigi(edm::vector_transform(inputTags, [this](edm::InputTag const & tag){return consumes<edm::DetSetVector<SiStripDigi> >(tag);})),
    inputTokensRawDigi(edm::vector_transform(inputTags, [this](edm::InputTag const & tag){return consumes<edm::DetSetVector<SiStripRawDigi> >(tag);})),
    subtractorPed(SiStripRawProcessingFactory::create_SubtractorPed(conf)),
    subtractorCMN(SiStripRawProcessingFactory::create_SubtractorCMN(conf)){

  produces< edm::DetSetVector<SiStripProcessedRawDigi> >("");
}

void SiStripProcessedRawDigiProducer::
produce(edm::Event& e, const edm::EventSetup& es) {

  std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > output(new edm::DetSetVector<SiStripProcessedRawDigi>());
  edm::Handle< edm::DetSetVector<SiStripDigi> > inputDigis;
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > inputRawdigis;

  es.get<SiStripGainRcd>().get(gainHandle);
  subtractorPed->init(es);
  subtractorCMN->init(es);

  std::string label = findInput(inputRawdigis, inputTokensRawDigi, e);
  if(      "VirginRaw"  == label )  vr_process(*inputRawdigis, *output);
  else if( "ProcessedRaw" == label )  pr_process(*inputRawdigis, *output);
  else if( "ZeroSuppressed" == findInput(inputDigis, inputTokensDigi, e) ) zs_process(*inputDigis, *output);
  else
    edm::LogError("Input Not Found");

  e.put(output);
}

template<class T>
inline
std::string SiStripProcessedRawDigiProducer::
findInput(edm::Handle<T>& handle, const std::vector<edm::EDGetTokenT<T> >& tokens, const edm::Event& e ) {

  for( typename std::vector<edm::EDGetTokenT<T> >::const_iterator
	 token = tokens.begin(); token != tokens.end(); ++token ) {
    unsigned index(token - tokens.begin());
    e.getByToken(*token, handle);
    if( handle.isValid() && !handle->empty() ) {
      edm::LogInfo("Input") << inputTags.at(index);
      return inputTags.at(index).instance();
    }
  }
  return "Input Not Found";
}

void SiStripProcessedRawDigiProducer::
zs_process(const edm::DetSetVector<SiStripDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  std::vector<float> digis;
  for(edm::DetSetVector<SiStripDigi>::const_iterator detset = input.begin(); detset != input.end(); detset++ )  {
    digis.clear();
    for(edm::DetSet<SiStripDigi>::const_iterator digi = detset->begin();  digi != detset->end();  digi++) {
      digis.resize( digi->strip(), 0);
      digis.push_back( digi->adc() );
    }
    common_process( detset->id, digis, output);
  }
}

void SiStripProcessedRawDigiProducer::
pr_process(const edm::DetSetVector<SiStripRawDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripRawDigi>::const_iterator detset=input.begin(); detset!=input.end(); detset++) {
    std::vector<float> digis;
    transform(detset->begin(), detset->end(), back_inserter(digis), boost::bind(&SiStripRawDigi::adc , _1));
    subtractorCMN->subtract(detset->id, 0, digis);
    common_process( detset->id, digis, output);
  }
}

void SiStripProcessedRawDigiProducer::
vr_process(const edm::DetSetVector<SiStripRawDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripRawDigi>::const_iterator detset=input.begin(); detset!=input.end(); detset++) {
    std::vector<int16_t> int_digis(detset->size());
    subtractorPed->subtract(*detset,int_digis);
    std::vector<float> digis(int_digis.begin(), int_digis.end());
    subtractorCMN->subtract(detset->id, 0, digis);
    common_process( detset->id, digis, output);
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
