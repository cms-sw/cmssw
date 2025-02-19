#ifndef CalibTracker_SiStripESProducers_SiStripTemplateFakeESSource
#define CalibTracker_SiStripESProducers_SiStripTemplateFakeESSource

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

template< typename TObject , typename TRecord, typename TService>
class SiStripTemplateFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripTemplateFakeESSource(const edm::ParameterSet&);
  ~SiStripTemplateFakeESSource(){};
  
  
  std::auto_ptr<TObject> produce(const TRecord&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripTemplateFakeESSource( const SiStripTemplateFakeESSource& );
  const SiStripTemplateFakeESSource& operator=( const SiStripTemplateFakeESSource& );
};

template< typename TObject , typename TRecord, typename TService>
SiStripTemplateFakeESSource<TObject,TRecord,TService>::SiStripTemplateFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<TRecord>();
}

template< typename TObject , typename TRecord, typename TService>
std::auto_ptr<TObject> SiStripTemplateFakeESSource<TObject,TRecord,TService>::produce(const TRecord& iRecord)
{
  edm::Service<TService> condObjBuilder;
  TObject *obj=0; 
  condObjBuilder->getObj(obj);
  std::auto_ptr<TObject> ptr(obj);
  return ptr;
}

template< typename TObject , typename TRecord, typename TService>
void SiStripTemplateFakeESSource<TObject,TRecord,TService>::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
							 const edm::IOVSyncValue& iov,
							 edm::ValidityInterval& iValidity){
  edm::ValidityInterval infinity( iov.beginOfTime(), iov.endOfTime() );
  iValidity = infinity;
}


#endif
