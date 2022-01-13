#ifndef EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H
#define EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

/**
    @file EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialDigiSource.h
    @class SiStripTrivialDigiSource

    @brief Creates a DetSetVector of SiStripDigis created using random
    number generators and attaches the collection to the Event. Allows
    to test the final DigiToRaw and RawToDigi converters.  
*/
class SiStripTrivialDigiSource : public edm::global::EDProducer<> {
public:
  SiStripTrivialDigiSource(const edm::ParameterSet&);
  ~SiStripTrivialDigiSource();

  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> esTokenCabling_;
  const float meanOcc_;
  const float rmsOcc_;
  const int ped_;
  const bool raw_;
  const bool useFedKey_;
};

#endif  // EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H
