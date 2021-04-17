#include "AnalysisAlgos/SiStripClusterInfoProducer/plugins/SiStripProcessedRawDigiProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"

#include <functional>

SiStripProcessedRawDigiProducer::SiStripProcessedRawDigiProducer(edm::ParameterSet const& conf)
    : inputTags_(conf.getParameter<std::vector<edm::InputTag> >("DigiProducersList")),
      inputTokensDigi_(edm::vector_transform(
          inputTags_, [this](edm::InputTag const& tag) { return consumes<edm::DetSetVector<SiStripDigi> >(tag); })),
      inputTokensRawDigi_(edm::vector_transform(
          inputTags_, [this](edm::InputTag const& tag) { return consumes<edm::DetSetVector<SiStripRawDigi> >(tag); })),
      gainToken_(esConsumes()),
      subtractorPed_(SiStripRawProcessingFactory::create_SubtractorPed(conf, consumesCollector())),
      subtractorCMN_(SiStripRawProcessingFactory::create_SubtractorCMN(conf, consumesCollector())) {
  produces<edm::DetSetVector<SiStripProcessedRawDigi> >("");
}

void SiStripProcessedRawDigiProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  std::unique_ptr<edm::DetSetVector<SiStripProcessedRawDigi> > output(new edm::DetSetVector<SiStripProcessedRawDigi>());
  edm::Handle<edm::DetSetVector<SiStripDigi> > inputDigis;
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > inputRawdigis;

  const auto& gain = es.getData(gainToken_);
  subtractorPed_->init(es);
  subtractorCMN_->init(es);

  std::string label = findInput(inputRawdigis, inputTokensRawDigi_, e);
  if ("VirginRaw" == label)
    vr_process(*inputRawdigis, *output, gain);
  else if ("ProcessedRaw" == label)
    pr_process(*inputRawdigis, *output, gain);
  else if ("ZeroSuppressed" == findInput(inputDigis, inputTokensDigi_, e))
    zs_process(*inputDigis, *output, gain);
  else
    edm::LogError("Input Not Found");

  e.put(std::move(output));
}

template <class T>
inline std::string SiStripProcessedRawDigiProducer::findInput(edm::Handle<T>& handle,
                                                              const std::vector<edm::EDGetTokenT<T> >& tokens,
                                                              const edm::Event& e) {
  for (typename std::vector<edm::EDGetTokenT<T> >::const_iterator token = tokens.begin(); token != tokens.end();
       ++token) {
    unsigned index(token - tokens.begin());
    e.getByToken(*token, handle);
    if (handle.isValid() && !handle->empty()) {
      edm::LogInfo("Input") << inputTags_.at(index);
      return inputTags_.at(index).instance();
    }
  }
  return "Input Not Found";
}

void SiStripProcessedRawDigiProducer::zs_process(const edm::DetSetVector<SiStripDigi>& input,
                                                 edm::DetSetVector<SiStripProcessedRawDigi>& output,
                                                 const SiStripGain& gain) {
  std::vector<float> digis;
  for (edm::DetSetVector<SiStripDigi>::const_iterator detset = input.begin(); detset != input.end(); ++detset) {
    digis.clear();
    for (edm::DetSet<SiStripDigi>::const_iterator digi = detset->begin(); digi != detset->end(); ++digi) {
      digis.resize(digi->strip(), 0);
      digis.push_back(digi->adc());
    }
    common_process(detset->id, digis, output, gain);
  }
}

void SiStripProcessedRawDigiProducer::pr_process(const edm::DetSetVector<SiStripRawDigi>& input,
                                                 edm::DetSetVector<SiStripProcessedRawDigi>& output,
                                                 const SiStripGain& gain) {
  for (edm::DetSetVector<SiStripRawDigi>::const_iterator detset = input.begin(); detset != input.end(); ++detset) {
    std::vector<float> digis;
    transform(
        detset->begin(), detset->end(), back_inserter(digis), std::bind(&SiStripRawDigi::adc, std::placeholders::_1));
    subtractorCMN_->subtract(detset->id, 0, digis);
    common_process(detset->id, digis, output, gain);
  }
}

void SiStripProcessedRawDigiProducer::vr_process(const edm::DetSetVector<SiStripRawDigi>& input,
                                                 edm::DetSetVector<SiStripProcessedRawDigi>& output,
                                                 const SiStripGain& gain) {
  for (edm::DetSetVector<SiStripRawDigi>::const_iterator detset = input.begin(); detset != input.end(); ++detset) {
    std::vector<int16_t> int_digis(detset->size());
    subtractorPed_->subtract(*detset, int_digis);
    std::vector<float> digis(int_digis.begin(), int_digis.end());
    subtractorCMN_->subtract(detset->id, 0, digis);
    common_process(detset->id, digis, output, gain);
  }
}

void SiStripProcessedRawDigiProducer::common_process(const uint32_t detId,
                                                     std::vector<float>& digis,
                                                     edm::DetSetVector<SiStripProcessedRawDigi>& output,
                                                     const SiStripGain& gain) {
  //Apply Gains
  SiStripApvGain::Range detGainRange = gain.getRange(detId);
  for (std::vector<float>::iterator it = digis.begin(); it < digis.end(); ++it)
    (*it) /= (gain.getStripGain(it - digis.begin(), detGainRange));

  //Insert as DetSet
  edm::DetSet<SiStripProcessedRawDigi> ds(detId);
  copy(digis.begin(), digis.end(), back_inserter(ds.data));
  output.insert(ds);
}
