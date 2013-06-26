#ifndef EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H
#define EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <string>

namespace sistrip { class RawToDigiModule; }
namespace sistrip { class RawToDigiUnpacker; }
class SiStripFedCabling;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h
   @class SiStripRawToDigiModule 
   
   @brief A plug-in module that takes a FEDRawDataCollection as input
   from the Event and creates EDProducts containing StripDigis.
*/

namespace sistrip {
  
  class RawToDigiModule : public edm::EDProducer {
    
  public:
    
    RawToDigiModule( const edm::ParameterSet& );
    ~RawToDigiModule();
    
    virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    
  private: 
    
    void updateCabling( const edm::EventSetup& );
    
    RawToDigiUnpacker* rawToDigi_;
    edm::InputTag productLabel_;
    const SiStripFedCabling* cabling_;
    uint32_t cacheId_;
    bool extractCm_;    
    bool doFullCorruptBufferChecks_;

    //March 2012: add flag for disabling APVe check in configuration
    bool doAPVEmulatorCheck_; 

  };
  
}

#endif // EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

