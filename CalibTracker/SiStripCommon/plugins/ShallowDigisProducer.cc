#include "ShallowDigisProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

ShallowDigisProducer::ShallowDigisProducer(const edm::ParameterSet& conf)
    : inputTags(conf.getParameter<std::vector<edm::InputTag> >("DigiProducersList")), noisesToken_(esConsumes()) {
  produces<std::vector<unsigned> >("id");
  produces<std::vector<unsigned> >("subdet");
  produces<std::vector<unsigned> >("strip");
  produces<std::vector<unsigned> >("adc");
  produces<std::vector<float> >("noise");
}

void ShallowDigisProducer::insert(products& p, edm::Event& e) {
  e.put(std::move(p.id), "id");
  e.put(std::move(p.subdet), "subdet");
  e.put(std::move(p.strip), "strip");
  e.put(std::move(p.adc), "adc");
  e.put(std::move(p.noise), "noise");
}

template <class T>
inline void ShallowDigisProducer::recordDigis(const T& digiCollection, products& p, const SiStripNoises& noises) {
  for (auto const& set : digiCollection) {
    SiStripNoises::Range detNoiseRange = noises.getRange(set.detId());
    for (auto const& digi : set) {
      p.id->push_back(set.detId());
      p.subdet->push_back((set.detId() >> 25) & 0x7);
      p.strip->push_back(digi.strip());
      p.adc->push_back(digi.adc());
      p.noise->push_back(noises.getNoise(digi.strip(), detNoiseRange));
    }
  }
}

void ShallowDigisProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  products p;
  edm::Handle<edm::DetSetVector<SiStripDigi> > inputOld;
  edm::Handle<edmNew::DetSetVector<SiStripDigi> > inputNew;
  const auto& noises = es.getData(noisesToken_);
  if (findInput(inputOld, e))
    recordDigis(*inputOld, p, noises);
  else if (findInput(inputNew, e))
    recordDigis(*inputNew, p, noises);
  else
    edm::LogWarning("Input Not Found");
  insert(p, e);
}

template <class T>
inline bool ShallowDigisProducer::findInput(edm::Handle<T>& handle, const edm::Event& e) {
  for (auto const& inputTag : inputTags) {
    e.getByLabel(inputTag, handle);
    if (handle.isValid() && !handle->empty()) {
      LogDebug("Input") << inputTag;
      return true;
    }
  }
  return false;
}
