#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "boost/cstdint.hpp"
#include <string>

class SiStripDigiToRaw;

using namespace std;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h
   @class SiStripDigiToRawModule 
   
   @brief A plug-in module that takes StripDigis as input from the
   Event and creates an EDProduct comprising a FEDRawDataCollection.
*/
class SiStripDigiToRawModule : public edm::EDProducer {
  
 public:
  
  SiStripDigiToRawModule( const edm::ParameterSet& );
  ~SiStripDigiToRawModule();
  
  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private:

  string inputModuleLabel_;
  SiStripDigiToRaw* digiToRaw_;
  uint32_t eventCounter_;

};

#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

