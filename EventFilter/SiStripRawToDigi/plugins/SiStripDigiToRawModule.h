// Last commit: $Id: OldSiStripDigiToRawModule.h,v 1.2 2008/01/17 11:54:44 giordano Exp $

#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "boost/cstdint.hpp"
#include <string>

class OldSiStripDigiToRaw;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h
   @class OldSiStripDigiToRawModule 
   
   @brief A plug-in module that takes StripDigis as input from the
   Event and creates an EDProduct comprising a FEDRawDataCollection.
*/
class OldSiStripDigiToRawModule : public edm::EDProducer {
  
 public:
  
  OldSiStripDigiToRawModule( const edm::ParameterSet& );
  ~OldSiStripDigiToRawModule();
  
  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private:

  std::string inputModuleLabel_;
  std::string inputDigiLabel_;
  OldSiStripDigiToRaw* digiToRaw_;
  uint32_t eventCounter_;

};

#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

