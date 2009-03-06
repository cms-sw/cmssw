#ifndef CalibTracker_SiStripESProducers_SiStripBadStripFakeESSource
#define CalibTracker_SiStripESProducers_SiStripBadStripFakeESSource

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
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"

//
// class declaration
//


class SiStripBadStripFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripBadStripFakeESSource(const edm::ParameterSet&);
  ~SiStripBadStripFakeESSource(){};
  
  
  std::auto_ptr<SiStripBadStrip> produce(const SiStripBadStripRcd&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripBadStripFakeESSource( const SiStripBadStripFakeESSource& );
  const SiStripBadStripFakeESSource& operator=( const SiStripBadStripFakeESSource& );
};

#endif
