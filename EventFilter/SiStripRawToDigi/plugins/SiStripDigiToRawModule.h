// Last commit: $Id: SiStripDigiToRawModule.h,v 1.7 2010/01/13 22:42:01 wmtan Exp $

#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
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
  
    virtual void beginJob() {}
    virtual void endJob() {}
  
    virtual void produce( edm::Event&, const edm::EventSetup& );
  
  private:

    std::string inputModuleLabel_;
    std::string inputDigiLabel_;
    FEDReadoutMode mode_;
    bool rawdigi_;
    DigiToRaw* digiToRaw_;
    uint32_t eventCounter_;

  };

}

#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

