#ifndef EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H
#define EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include <string>

class SiStripRawToDigiUnpacker;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h
   @class SiStripRawToDigiModule 
   
   @brief A plug-in module that takes a FEDRawDataCollection as input
   from the Event and creates EDProducts containing StripDigis.
*/
class SiStripRawToDigiModule : public edm::EDProducer {
  
 public:
  
  SiStripRawToDigiModule( const edm::ParameterSet& );
  ~SiStripRawToDigiModule();

  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 
  
  SiStripRawToDigiUnpacker* rawToDigi_;

  bool createDigis_;

  std::string label_;
  std::string instance_;
  
};

#endif // EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

