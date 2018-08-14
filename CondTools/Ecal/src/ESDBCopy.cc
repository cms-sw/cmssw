#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "CondTools/Ecal/interface/ESDBCopy.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESADCToGeVConstantRcd.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/ESObjects/interface/ESWeightStripGroups.h"
#include "CondFormats/DataRecord/interface/ESWeightStripGroupsRcd.h"
#include "CondFormats/ESObjects/interface/ESTBWeights.h"
#include "CondFormats/DataRecord/interface/ESTBWeightsRcd.h"



#include <vector>

ESDBCopy::ESDBCopy(const edm::ParameterSet& iConfig) :
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


ESDBCopy::~ESDBCopy()
{
  
}

void ESDBCopy::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
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



bool ESDBCopy::shouldCopy(const edm::EventSetup& evtSetup, std::string container)
{

  unsigned long long cacheID = 0;
  if (container == "ESPedestals") {
    cacheID = evtSetup.get<ESPedestalsRcd>().cacheIdentifier();
  } else if (container == "ESADCToGeVConstant") {
    cacheID = evtSetup.get<ESADCToGeVConstantRcd>().cacheIdentifier();
  } else if (container == "ESIntercalibConstants") {
    cacheID = evtSetup.get<ESIntercalibConstantsRcd>().cacheIdentifier();
  } else if (container == "ESWeightStripGroups") {
    cacheID = evtSetup.get<ESWeightStripGroupsRcd>().cacheIdentifier();
  } else if (container == "ESTBWeights") {
    cacheID = evtSetup.get<ESTBWeightsRcd>().cacheIdentifier();
  } else if (container == "ESChannelStatus") {
    cacheID = evtSetup.get<ESChannelStatusRcd>().cacheIdentifier();
  } 

  else {
    throw cms::Exception("Unknown container");
  }
  
  if (m_cacheIDs[container] == cacheID) {
    return false;
  } else {
    m_cacheIDs[container] = cacheID;
    return true;
  }

}



void ESDBCopy::copyToDB(const edm::EventSetup& evtSetup, std::string container)
{
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if ( !dbOutput.isAvailable() ) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string recordName = m_records[container];

  if (container == "ESPedestals") {
    edm::ESHandle<ESPedestals> handle;
    evtSetup.get<ESPedestalsRcd>().get(handle);
    const ESPedestals* obj = handle.product();
    std::cout << "ped pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const ESPedestals>( new ESPedestals(*obj), dbOutput->beginOfTime(),dbOutput->endOfTime(),recordName);

  }  else if (container == "ESADCToGeVConstant") {
    edm::ESHandle<ESADCToGeVConstant> handle;
    evtSetup.get<ESADCToGeVConstantRcd>().get(handle);
    const ESADCToGeVConstant* obj = handle.product();
    std::cout << "adc pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const ESADCToGeVConstant>( new ESADCToGeVConstant(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  }  else if (container == "ESChannelStatus") {
    edm::ESHandle<ESChannelStatus> handle;
    evtSetup.get<ESChannelStatusRcd>().get(handle);
    const ESChannelStatus* obj = handle.product();
    std::cout << "channel status pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const ESChannelStatus>( new ESChannelStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  }
else if (container == "ESIntercalibConstants") {
    edm::ESHandle<ESIntercalibConstants> handle;
    evtSetup.get<ESIntercalibConstantsRcd>().get(handle);
    const ESIntercalibConstants* obj = handle.product();
    std::cout << "inter pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const ESIntercalibConstants>( new ESIntercalibConstants(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  } else if (container == "ESWeightStripGroups") {
    edm::ESHandle<ESWeightStripGroups> handle;
    evtSetup.get<ESWeightStripGroupsRcd>().get(handle);
    const ESWeightStripGroups* obj = handle.product();
    std::cout << "weight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const ESWeightStripGroups>( new ESWeightStripGroups(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "ESTBWeights") {
    edm::ESHandle<ESTBWeights> handle;
    evtSetup.get<ESTBWeightsRcd>().get(handle);
    const ESTBWeights* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const ESTBWeights>( new ESTBWeights(*obj), dbOutput->beginOfTime(),dbOutput->endOfTime(),recordName);


  } else {
    throw cms::Exception("Unknown container");
  }

  std::cout<< "ESDBCopy wrote " << recordName << std::endl;
}
