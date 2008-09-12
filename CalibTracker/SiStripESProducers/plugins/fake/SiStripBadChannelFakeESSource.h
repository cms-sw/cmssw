#ifndef CalibTracker_SiStripESProducers_SiStripBadChannelFakeESSource
#define CalibTracker_SiStripESProducers_SiStripBadChannelFakeESSource

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
#include "CondFormats/DataRecord/interface/SiStripBadChannelRcd.h"

//
// class declaration
//


class SiStripBadChannelFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripBadChannelFakeESSource(const edm::ParameterSet&);
  ~SiStripBadChannelFakeESSource(){};
  
  
  std::auto_ptr<SiStripBadStrip> produce(const SiStripBadChannelRcd&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripBadChannelFakeESSource( const SiStripBadChannelFakeESSource& );
  const SiStripBadChannelFakeESSource& operator=( const SiStripBadChannelFakeESSource& );
};

#endif
