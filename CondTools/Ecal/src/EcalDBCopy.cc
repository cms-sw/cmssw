#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/Ecal/interface/EcalDBCopy.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"


#include <map>
#include <vector>



EcalDBCopy::EcalDBCopy(const edm::ParameterSet& iConfig) :
  m_timetype(iConfig.getParameter<std::string>("timetype")),
  m_cacheIDs(),
  m_records()
{

  std::string container;
  std::string tag;
  std::string record;
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for(Parameters::iterator i = toCopy.begin(); i != toCopy.end(); ++i) {
    container = i->getParameter<std::string>("container");
    record = i->getParameter<std::string>("record");
    m_cacheIDs.insert( std::make_pair(container, 0) );
    m_records.insert( std::make_pair(container, record) );
  }
  
}


EcalDBCopy::~EcalDBCopy()
{
  
}

void EcalDBCopy::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
{

  std::string container;
  std::string record;
  typedef std::map<std::string, std::string>::const_iterator recordIter;
  for (recordIter i = m_records.begin(); i != m_records.end(); ++i) {
    container = (*i).first;
    record = (*i).second;
    if ( shouldCopy(evtSetup, container) ) {
      copyToDB(evtSetup, container);
    }
  }
  
}



bool EcalDBCopy::shouldCopy(const edm::EventSetup& evtSetup, std::string container)
{

  unsigned long long cacheID = 0;
  if (container == "EcalPedestals") {
    cacheID = evtSetup.get<EcalPedestalsRcd>().cacheIdentifier();
  } else if (container == "EcalADCToGeVConstant") {
    cacheID = evtSetup.get<EcalADCToGeVConstantRcd>().cacheIdentifier();
  } else if (container == "EcalIntercalibConstants") {
    cacheID = evtSetup.get<EcalIntercalibConstantsRcd>().cacheIdentifier();
  } else if (container == "EcalGainRatios") {
    cacheID = evtSetup.get<EcalGainRatiosRcd>().cacheIdentifier();
  } else if (container == "EcalTBWeights") {
    cacheID = evtSetup.get<EcalTBWeightsRcd>().cacheIdentifier();
  } else {
    throw cms::Exception("Unknown container");
  }
  
  if (m_cacheIDs[container] == cacheID) {
    return 0;
  } else {
    m_cacheIDs[container] = cacheID;
    return 1;
  }

}



void EcalDBCopy::copyToDB(const edm::EventSetup& evtSetup, std::string container)
{
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if ( !dbOutput.isAvailable() ) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string recordName = m_records[container];
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType(recordName));
  const edm::eventsetup::EventSetupRecord* record = evtSetup.find(recordKey);
  
  unsigned long long sinceTime;
  unsigned long long tillTime;
  if (m_timetype == "timestamp") {
    sinceTime = record->validityInterval().first().time().value();  
    tillTime = record->validityInterval().last().time().value(); 
  } else {
    sinceTime = record->validityInterval().first().eventID().run();
    tillTime = record->validityInterval().last().eventID().run();
  }

  size_t callbackToken = dbOutput->callbackToken(container);

  if (container == "EcalPedestals") {
    edm::ESHandle<EcalPedestals> handle;
    evtSetup.get<EcalPedestalsRcd>().get(handle);
    const EcalPedestals* obj = handle.product();
    dbOutput->newValidityForNewPayload<EcalPedestals>(new EcalPedestals(*obj), tillTime, callbackToken);
  } else if (container == "EcalADCToGeVConstant") {
    edm::ESHandle<EcalADCToGeVConstant> handle;
    evtSetup.get<EcalADCToGeVConstantRcd>().get(handle);
    const EcalADCToGeVConstant* obj = handle.product();
    dbOutput->newValidityForNewPayload<EcalADCToGeVConstant>(new EcalADCToGeVConstant(*obj), tillTime, callbackToken);
  } else if (container == "EcalIntercalibConstants") {
    edm::ESHandle<EcalIntercalibConstants> handle;
    evtSetup.get<EcalIntercalibConstantsRcd>().get(handle);
    const EcalIntercalibConstants* obj = handle.product();
    dbOutput->newValidityForNewPayload<EcalIntercalibConstants>(new EcalIntercalibConstants(*obj), tillTime, callbackToken);
  } else if (container == "EcalGainRatios") {
    edm::ESHandle<EcalGainRatios> handle;
    evtSetup.get<EcalGainRatiosRcd>().get(handle);
    const EcalGainRatios* obj = handle.product();
    dbOutput->newValidityForNewPayload<EcalGainRatios>(new EcalGainRatios(*obj), tillTime, callbackToken);
  } else if (container == "EcalTBWeights") {
    edm::ESHandle<EcalTBWeights> handle;
    evtSetup.get<EcalTBWeightsRcd>().get(handle);
    const EcalTBWeights* obj = handle.product();
    dbOutput->newValidityForNewPayload<EcalTBWeights>(new EcalTBWeights(*obj), tillTime, callbackToken);
  } else {
    throw cms::Exception("Unknown container");
  }

  edm::LogInfo("EcalDBCopy") << "Wrote " << container << " for " << m_timetype << " " << tillTime
			     << " (" << sinceTime << ")";

}
