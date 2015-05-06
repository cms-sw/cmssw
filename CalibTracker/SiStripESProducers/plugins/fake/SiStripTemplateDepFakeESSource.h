#ifndef CalibTracker_SiStripESProducers_SiStripTemplateDepFakeESSource
#define CalibTracker_SiStripESProducers_SiStripTemplateDepFakeESSource

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

template< typename TObject , typename TRecord, typename TService, typename DepTRecord, typename DepTObj>
class SiStripTemplateDepFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  SiStripTemplateDepFakeESSource(const edm::ParameterSet&);
  ~SiStripTemplateDepFakeESSource(){};
  
  
  std::auto_ptr<TObject> produce(const TRecord&);
  
private:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
		       const edm::IOVSyncValue& iov,
		       edm::ValidityInterval& iValidity);
  
  SiStripTemplateDepFakeESSource( const SiStripTemplateDepFakeESSource& );
  const SiStripTemplateDepFakeESSource& operator=( const SiStripTemplateDepFakeESSource& );
};

template< typename TObject , typename TRecord, typename TService, typename DepTRecord, typename DepTObj>
SiStripTemplateDepFakeESSource<TObject,TRecord,TService,DepTRecord,DepTObj>::SiStripTemplateDepFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<TRecord>();
}

template< typename TObject , typename TRecord, typename TService, typename DepTRecord, typename DepTObj>
std::auto_ptr<TObject> SiStripTemplateDepFakeESSource<TObject,TRecord,TService,DepTRecord,DepTObj>::produce(const TRecord& iRecord)
{
  edm::ESHandle<DepTObj> depObjHandle;
  iRecord.template getRecord<DepTRecord>().get(depObjHandle);
  const DepTObj* const depObject = depObjHandle.product();

  edm::Service<TService> condObjBuilder;
  TObject *obj=0; 
  condObjBuilder->getObj(obj,depObject);
  std::auto_ptr<TObject> ptr(obj);
  return ptr;
}

template< typename TObject , typename TRecord, typename TService, typename DepTRecord, typename DepTObj>
void SiStripTemplateDepFakeESSource<TObject,TRecord,TService,DepTRecord,DepTObj>::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
							 const edm::IOVSyncValue& iov,
							 edm::ValidityInterval& iValidity){
  edm::ValidityInterval infinity( iov.beginOfTime(), iov.endOfTime() );
  iValidity = infinity;
}


#endif
