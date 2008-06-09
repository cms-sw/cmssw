// Last commit: $Id: SiStripRawToDigiModule.h,v 1.3 2008/06/09 11:30:18 bainbrid Exp $

#ifndef EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H
#define EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <string>

class SiStripRawToDigiUnpacker;
class SiStripFedCabling;

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

  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();

  virtual void beginRun( edm::Run&, const edm::EventSetup& );
  virtual void endRun( edm::Run&, const edm::EventSetup& ) {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  void updateCabling( const edm::EventSetup& );
  void deleteCabling();
  
  SiStripRawToDigiUnpacker* rawToDigi_;

  std::string label_;
  std::string instance_;

  SiStripFedCabling* cabling_;
  
  uint32_t cacheId_;

};

#endif // EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

