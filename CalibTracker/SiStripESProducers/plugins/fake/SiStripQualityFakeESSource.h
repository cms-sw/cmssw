#ifndef CalibTracker_SiStripESProducers_SiStripQualityFakeESSource
#define CalibTracker_SiStripESProducers_SiStripQualityFakeESSource

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

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

//
// class declaration
//


class SiStripQualityFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripQualityFakeESSource(const edm::ParameterSet&);
  ~SiStripQualityFakeESSource(){};
  
  
  std::auto_ptr<SiStripQuality> produce(const SiStripQualityRcd&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripQualityFakeESSource( const SiStripQualityFakeESSource& );
  const SiStripQualityFakeESSource& operator=( const SiStripQualityFakeESSource& );
};

#endif
