// Last commit: $Id: SiStripTrivialDigiSource.h,v 1.2 2008/03/31 16:51:50 bainbrid Exp $

#ifndef EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H
#define EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H

#include "EventFilter/SiStripRawToDigi/test/stubs/SiStripTrivialDigiAnalysis.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "boost/cstdint.hpp"

 /**
    @file EventFilter/SiStripRawToDigi/test/stubs/SiStripTrivialDigiSource.h
    @class SiStripTrivialDigiSource

    @brief Creates a DetSetVector of SiStripDigis created using random
    number generators and attaches the collection to the Event. Allows
    to test the final DigiToRaw and RawToDigi converters.  
*/
class SiStripTrivialDigiSource : public edm::EDProducer {
  
 public:
  
  SiStripTrivialDigiSource( const edm::ParameterSet& );
  ~SiStripTrivialDigiSource();
  
  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  float meanOcc_;

  float rmsOcc_;

  int ped_;

  bool  raw_;

  bool  useFedKey_;
  
};

#endif // EventFilter_SiStripRawToDigi_SiStripTrivialDigiSource_H
