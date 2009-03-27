// Last commit: $Id: SiStripDigiToRawModule.h,v 1.3 2009/03/26 18:54:49 bainbrid Exp $

#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "boost/cstdint.hpp"
#include <string>

namespace sistrip {

  class DigiToRaw;

  /**
     @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h
     @class DigiToRawModule 
   
     @brief A plug-in module that takes StripDigis as input from the
     Event and creates an EDProduct comprising a FEDRawDataCollection.
  */
  class DigiToRawModule : public edm::EDProducer {
  
  public:
  
    DigiToRawModule( const edm::ParameterSet& );
    ~DigiToRawModule();
  
    virtual void beginJob( const edm::EventSetup& ) {;}
    virtual void endJob() {;}
  
    virtual void produce( edm::Event&, const edm::EventSetup& );
  
  private:

    std::string inputModuleLabel_;
    std::string inputDigiLabel_;
    DigiToRaw* digiToRaw_;
    uint32_t eventCounter_;

  };

}


////////////////////////////////////////////////////////////////////////////////
//@@@ TO BE DEPRECATED BELOW!!!


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

