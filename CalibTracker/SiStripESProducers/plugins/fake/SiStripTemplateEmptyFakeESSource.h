#ifndef CalibTracker_SiStripESProducers_SiStripTemplateEmptyFakeESSource
#define CalibTracker_SiStripESProducers_SiStripTemplateEmptyFakeESSource

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

//
// class declaration
//

template< typename TObject , typename TRecord>
class SiStripTemplateEmptyFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripTemplateEmptyFakeESSource(const edm::ParameterSet&);
  ~SiStripTemplateEmptyFakeESSource(){};
  
  
  std::auto_ptr<TObject> produce(const TRecord&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripTemplateEmptyFakeESSource( const SiStripTemplateEmptyFakeESSource& );
  const SiStripTemplateEmptyFakeESSource& operator=( const SiStripTemplateEmptyFakeESSource& );
};

template< typename TObject , typename TRecord>
SiStripTemplateEmptyFakeESSource<TObject,TRecord>::SiStripTemplateEmptyFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<TRecord>();
}

template< typename TObject , typename TRecord>
std::auto_ptr<TObject> SiStripTemplateEmptyFakeESSource<TObject,TRecord>::produce(const TRecord& iRecord)
{
  std::auto_ptr<TObject> ptr(new TObject);
  return ptr;
}

template< typename TObject , typename TRecord>
void SiStripTemplateEmptyFakeESSource<TObject,TRecord>::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
							 const edm::IOVSyncValue& iov,
							 edm::ValidityInterval& iValidity){
  edm::ValidityInterval infinity( iov.beginOfTime(), iov.endOfTime() );
  iValidity = infinity;
}


#endif
