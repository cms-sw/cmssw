#ifndef EventFilter_SiStripRawToDigi_TrivialSiStripDigiSource_H
#define EventFilter_SiStripRawToDigi_TrivialSiStripDigiSource_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "boost/cstdint.hpp"

 /**
    @file EventFilter/SiStripRawToDigi/interface/TrivialSiStripDigiSource.h
    @class TrivialSiStripDigiSource

    @brief Creates a DetSetVector of SiStripDigis created using random
    number generators and attaches the collection to the Event. Allows
    to test the final DigiToRaw and RawToDigi converters.  
*/
class TrivialSiStripDigiSource : public edm::EDProducer {
  
 public:
  
  TrivialSiStripDigiSource( const edm::ParameterSet& );
  ~TrivialSiStripDigiSource();
  
  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 
  
  uint32_t eventCounter_;
  float meanOcc_;
  float rmsOcc_;
  
};

#endif // EventFilter_SiStripRawToDigi_TrivialSiStripDigiSource_H
