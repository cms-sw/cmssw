#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"

#include "EcalTPGDBCopy.h"

#include <vector>



EcalTPGDBCopy::EcalTPGDBCopy(const edm::ParameterSet& iConfig) :
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


EcalTPGDBCopy::~EcalTPGDBCopy()
{
  
}

void EcalTPGDBCopy::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
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



bool EcalTPGDBCopy::shouldCopy(const edm::EventSetup& evtSetup, std::string container)
{

  unsigned long long cacheID = 0;

  if (container == "EcalTPGPedestals") {
    cacheID = evtSetup.get<EcalTPGPedestalsRcd>().cacheIdentifier();
  } else if (container == "EcalTPGLinearizationConst") {
    cacheID = evtSetup.get<EcalTPGLinearizationConstRcd>().cacheIdentifier();
  } else if (container == "EcalTPGSlidingWindow") {
    cacheID = evtSetup.get<EcalTPGSlidingWindowRcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainEBIdMap") {
    cacheID = evtSetup.get<EcalTPGFineGrainEBIdMapRcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainStripEE") {
    cacheID = evtSetup.get<EcalTPGFineGrainStripEERcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainTowerEE") {
    cacheID = evtSetup.get<EcalTPGFineGrainTowerEERcd>().cacheIdentifier();
  } else if (container == "EcalTPGLutIdMap") {
    cacheID = evtSetup.get<EcalTPGLutIdMapRcd>().cacheIdentifier();
  } else if (container == "EcalTPGWeightIdMap") {
    cacheID = evtSetup.get<EcalTPGWeightIdMapRcd>().cacheIdentifier();
  } else if (container == "EcalTPGWeightGroup") {
    cacheID = evtSetup.get<EcalTPGWeightGroupRcd>().cacheIdentifier();
  } else if (container == "EcalTPGLutGroup") {
    cacheID = evtSetup.get<EcalTPGLutGroupRcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainEBGroup") {
    cacheID = evtSetup.get<EcalTPGFineGrainEBGroupRcd>().cacheIdentifier();
  } else if (container == "EcalTPGPhysicsConst") {
    cacheID = evtSetup.get<EcalTPGPhysicsConstRcd>().cacheIdentifier();
  } else if (container == "EcalTPGCrystalStatus") {
    cacheID = evtSetup.get<EcalTPGCrystalStatusRcd>().cacheIdentifier();
  } else if (container == "EcalTPGTowerStatus") {
    cacheID = evtSetup.get<EcalTPGTowerStatusRcd>().cacheIdentifier();
  }
   

  else {
    throw cms::Exception("Unknown container");
  }
  
  if (m_cacheIDs[container] == cacheID) {
    return 0;
  } else {
    m_cacheIDs[container] = cacheID;
    return 1;
  }

}



void EcalTPGDBCopy::copyToDB(const edm::EventSetup& evtSetup, std::string container)
{
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if ( !dbOutput.isAvailable() ) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string recordName = m_records[container];

  if (container == "EcalTPGPedestals") {
    edm::ESHandle<EcalTPGPedestals> handle;
    evtSetup.get<EcalTPGPedestalsRcd>().get(handle);
    const EcalTPGPedestals* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGPedestals>( new EcalTPGPedestals(*obj), dbOutput->beginOfTime(),dbOutput->endOfTime(),recordName);

  }  else if (container == "EcalTPGLinearizationConst") {
    edm::ESHandle<EcalTPGLinearizationConst> handle;
    evtSetup.get<EcalTPGLinearizationConstRcd>().get(handle);
    const EcalTPGLinearizationConst* obj = handle.product();

    dbOutput->createNewIOV<const EcalTPGLinearizationConst>( new EcalTPGLinearizationConst(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  }  else if (container == "EcalTPGSlidingWindow") {
    edm::ESHandle<EcalTPGSlidingWindow> handle;
    evtSetup.get<EcalTPGSlidingWindowRcd>().get(handle);
    const EcalTPGSlidingWindow* obj = handle.product();

    dbOutput->createNewIOV<const EcalTPGSlidingWindow>( new EcalTPGSlidingWindow(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  }
else if (container == "EcalTPGFineGrainEBIdMap") {
    edm::ESHandle<EcalTPGFineGrainEBIdMap> handle;
    evtSetup.get<EcalTPGFineGrainEBIdMapRcd>().get(handle);
    const EcalTPGFineGrainEBIdMap* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainEBIdMap>( new EcalTPGFineGrainEBIdMap(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGFineGrainStripEE") {
    edm::ESHandle<EcalTPGFineGrainStripEE> handle;
    evtSetup.get<EcalTPGFineGrainStripEERcd>().get(handle);
    const EcalTPGFineGrainStripEE* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainStripEE>( new EcalTPGFineGrainStripEE(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGFineGrainTowerEE") {
    edm::ESHandle<EcalTPGFineGrainTowerEE> handle;
    evtSetup.get<EcalTPGFineGrainTowerEERcd>().get(handle);
    const EcalTPGFineGrainTowerEE* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainTowerEE>( new EcalTPGFineGrainTowerEE(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGLutIdMap") {
    edm::ESHandle<EcalTPGLutIdMap> handle;
    evtSetup.get<EcalTPGLutIdMapRcd>().get(handle);
    const EcalTPGLutIdMap* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGLutIdMap>( new EcalTPGLutIdMap(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGWeightIdMap") {
    edm::ESHandle<EcalTPGWeightIdMap> handle;
    evtSetup.get<EcalTPGWeightIdMapRcd>().get(handle);
    const EcalTPGWeightIdMap* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGWeightIdMap>( new EcalTPGWeightIdMap(*obj), dbOutput->beginOfTime(),dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGWeightGroup") {
    edm::ESHandle<EcalTPGWeightGroup> handle;
    evtSetup.get<EcalTPGWeightGroupRcd>().get(handle);
    const EcalTPGWeightGroup* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGWeightGroup>( new EcalTPGWeightGroup(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGLutGroup") {
    edm::ESHandle<EcalTPGLutGroup> handle;
    evtSetup.get<EcalTPGLutGroupRcd>().get(handle);
    const EcalTPGLutGroup* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGLutGroup>( new EcalTPGLutGroup(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGFineGrainEBGroup") {
    edm::ESHandle<EcalTPGFineGrainEBGroup> handle;
    evtSetup.get<EcalTPGFineGrainEBGroupRcd>().get(handle);
    const EcalTPGFineGrainEBGroup* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainEBGroup>( new EcalTPGFineGrainEBGroup(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGPhysicsConst") {
    edm::ESHandle<EcalTPGPhysicsConst> handle;
    evtSetup.get<EcalTPGPhysicsConstRcd>().get(handle);
    const EcalTPGPhysicsConst* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGPhysicsConst>( new EcalTPGPhysicsConst(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGCrystalStatus") {
    edm::ESHandle<EcalTPGCrystalStatus> handle;
    evtSetup.get<EcalTPGCrystalStatusRcd>().get(handle);
    const EcalTPGCrystalStatus* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGCrystalStatus>( new EcalTPGCrystalStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTPGTowerStatus") {
    edm::ESHandle<EcalTPGTowerStatus> handle;
    evtSetup.get<EcalTPGTowerStatusRcd>().get(handle);
    const EcalTPGTowerStatus* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGTowerStatus>( new EcalTPGTowerStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  }
  
  
  else {
    throw cms::Exception("Unknown container");
  }

  std::cout<< "EcalTPGDBCopy wrote " << recordName << std::endl;
}
