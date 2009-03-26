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
class OldSiStripRawToDigiUnpacker;
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
    
    virtual void beginJob( const edm::EventSetup& );
    virtual void beginRun( edm::Run&, const edm::EventSetup& );
    virtual void produce( edm::Event&, const edm::EventSetup& );
    
  private: 
    
    void updateCabling( const edm::EventSetup& );
    
    RawToDigiUnpacker* rawToDigi_;
    std::string label_;
    std::string instance_;
    const SiStripFedCabling* cabling_;
    uint32_t cacheId_;
    
  };
  
}

class OldSiStripRawToDigiModule : public edm::EDProducer {
  
 public:
  
  OldSiStripRawToDigiModule( const edm::ParameterSet& );
  ~OldSiStripRawToDigiModule();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void beginRun( edm::Run&, const edm::EventSetup& );
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  void updateCabling( const edm::EventSetup& );
  
  OldSiStripRawToDigiUnpacker* rawToDigi_;
  std::string label_;
  std::string instance_;
  const SiStripFedCabling* cabling_;
  uint32_t cacheId_;

};



#endif // EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

