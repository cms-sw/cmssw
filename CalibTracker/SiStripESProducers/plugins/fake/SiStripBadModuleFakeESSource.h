#ifndef CalibTracker_SiStripESProducers_SiStripBadModuleFakeESSource
#define CalibTracker_SiStripESProducers_SiStripBadModuleFakeESSource

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/DataRecord/interface/SiStripBadModuleRcd.h"

//
// class declaration
//


class SiStripBadModuleFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripBadModuleFakeESSource(const edm::ParameterSet&);
  ~SiStripBadModuleFakeESSource(){};
  
  
  std::auto_ptr<SiStripBadStrip> produce(const SiStripBadModuleRcd&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripBadModuleFakeESSource( const SiStripBadModuleFakeESSource& );
  const SiStripBadModuleFakeESSource& operator=( const SiStripBadModuleFakeESSource& );
};

#endif
