
#ifndef EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H
#define EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H

#include "FWCore/Framework/interface/EDProducer.h"

/**
    @file EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialDigiSource.h
    @class SiStripTrivialDigiSource

    @brief Creates a DetSetVector of SiStripDigis created using random
    number generators and attaches the collection to the Event. Allows
    to test the final DigiToRaw and RawToDigi converters.  
*/
class SiStripTrivialDigiSource : public edm::EDProducer {
public:
  SiStripTrivialDigiSource(const edm::ParameterSet&);
  ~SiStripTrivialDigiSource();

  virtual void beginJob() { ; }
  virtual void endJob() { ; }

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  float meanOcc_;

  float rmsOcc_;

  int ped_;

  bool raw_;

  bool useFedKey_;
};

#endif  // EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H
