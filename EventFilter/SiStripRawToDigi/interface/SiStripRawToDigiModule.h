#ifndef EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H
#define EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "boost/cstdint.hpp"
#include <string>

class SiStripRawToDigi;

using namespace std;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h
   @class SiStripRawToDigiModule 
   
   @brief A plug-in module that takes a FEDRawDataCollection as input
   from the Event and creates an EDProduct comprising StripDigis.
*/
class SiStripRawToDigiModule : public edm::EDProducer {
  
 public:

  SiStripRawToDigiModule( const edm::ParameterSet& );
  ~SiStripRawToDigiModule();

  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 
  
  string inputModuleLabel_;
  SiStripRawToDigi* rawToDigi_;
  uint32_t event_;

};

#endif // EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

