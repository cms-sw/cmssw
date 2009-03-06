#ifndef CalibTracker_SiStripESProducers_SiStripBadFiberFakeESSource
#define CalibTracker_SiStripESProducers_SiStripBadFiberFakeESSource

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
#include "CondFormats/DataRecord/interface/SiStripBadFiberRcd.h"

//
// class declaration
//


class SiStripBadFiberFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripBadFiberFakeESSource(const edm::ParameterSet&);
  ~SiStripBadFiberFakeESSource(){};
  
  
  std::auto_ptr<SiStripBadStrip> produce(const SiStripBadFiberRcd&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripBadFiberFakeESSource( const SiStripBadFiberFakeESSource& );
  const SiStripBadFiberFakeESSource& operator=( const SiStripBadFiberFakeESSource& );
};

#endif
