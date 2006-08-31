#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibCalorimetry/EcalTBCondTools/interface/ReprocessEcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <map>
#include <set>


ReprocessEcalPedestals::ReprocessEcalPedestals(const edm::ParameterSet& iConfig) {
  m_cacheID = 0;
  /*  
  EBpedMeanX12_ = ps.getUntrackedParameter<double>("EBpedMeanX12", 200.);
  EBpedRMSX12_  = ps.getUntrackedParameter<double>("EBpedRMSX12",  1.10);
  EBpedMeanX6_  = ps.getUntrackedParameter<double>("EBpedMeanX6",  200.);
  EBpedRMSX6_   = ps.getUntrackedParameter<double>("EBpedRMSX6",   0.90);
  EBpedMeanX1_  = ps.getUntrackedParameter<double>("EBpedMeanX1",  200.);
  EBpedRMSX1_   = ps.getUntrackedParameter<double>("EBpedRMSX1",   0.62);
  */
}

ReprocessEcalPedestals::~ReprocessEcalPedestals() {

}

void ReprocessEcalPedestals::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup) {

  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if ( !dbOutput.isAvailable() ) {
    edm::LogError("ReprocessEcalPedestals") << "PoolDBOUtputService is not available";
    return;
  }

  size_t callbackToken = dbOutput->callbackToken("EcalPedestals");
  unsigned int irun = evt.id().run();
  unsigned long long currentTime = dbOutput->currentTime();
  edm::ESHandle<EcalPedestals> pedsHandle;
  evtSetup.get<EcalPedestalsRcd>().get(pedsHandle);
  unsigned long long cacheID = evtSetup.get<EcalPedestalsRcd>().cacheIdentifier();

  // Don't do anything unless the EventSetup pedestals have changed
  if (m_cacheID == cacheID) { return; }

  edm::LogInfo("ReprocessEcalPedestals") << "\nirun        " << irun
					 << "\ncurrentTime " << currentTime
					 << "\ncacheID     " << cacheID;

  const EcalPedestals* peds = pedsHandle.product();
  
  EBDetId detId;
  uint32_t rawId;

  if (m_pedCache.m_pedestals.size() == 0) {
    // Make a list of SM we are dealing with    
    EcalPedestalsMapIterator detId_iter;
    for (detId_iter = peds->m_pedestals.begin(); detId_iter != peds->m_pedestals.end(); detId_iter++) {
      rawId = (*detId_iter).first;
      detId = EBDetId(rawId);
      m_smSet.insert( detId.ism() );
    }

    std::set<int>::const_iterator sm_iter;
    for (sm_iter = m_smSet.begin(); sm_iter != m_smSet.end(); sm_iter++) {
      edm::LogInfo("ReprocessEcalPedestals") << "SMset has SM:  " << *sm_iter;
    }
      
  }

  // Merge the retrieved EcalPedestals with the one in the cache
  int SM;
    
  std::set<int>::const_iterator sm_iter;
  std::ostringstream msg;
  for (sm_iter = m_smSet.begin(); sm_iter != m_smSet.end(); sm_iter++) {
    SM = *sm_iter;
    for (int crystal = 1; crystal <= 1700; crystal++) {
      msg << "Set SM " << SM << " crystal " << crystal;
      detId = EBDetId(SM, crystal, EBDetId::SMCRYSTALMODE);
      rawId = detId.rawId();

      // Look for pedestals in the retrieved set and the cache
      EcalPedestalsMapIterator currIter = peds->m_pedestals.find(rawId);
      EcalPedestalsMapIterator cacheIter = m_pedCache.m_pedestals.find(rawId);
      if (currIter != peds->m_pedestals.end()) {
	msg << " to retrieved value";
	// Channel is in the retrieved set, add to cache
	m_pedCache.m_pedestals.insert(*currIter);
      } else if (cacheIter == m_pedCache.m_pedestals.end() ) {
	msg << " to nominal value";
	// Cache has no value for this channel, set to default
	EcalPedestals::Item item;
	item.mean_x12 = 200.0;  item.rms_x12 = 1.10;
	item.mean_x6  = 200.0;  item.rms_x6  = 0.90;
	item.mean_x1  = 200.0;  item.rms_x1  = 0.62;
	m_pedCache.m_pedestals[rawId] = item;
      } else {
	msg << " to cached value";
      }
      msg << std::endl;
    }
  }

  edm::LogInfo("ReprocessEcalPedestals") << msg.str();


  // Check that we have #sm * 1700 channels
  if (m_pedCache.m_pedestals.size() != m_smSet.size() * 1700) {
    throw cms::Exception("Pedestals object was not filled with enough channels");
  }

  // Write the cache to the DB
  dbOutput->newValidityForNewPayload<EcalPedestals>(new EcalPedestals(m_pedCache), currentTime, callbackToken);
  edm::LogInfo("ReprocessEcalPedestals") << "Wrote new set of pedestals";

  m_cacheID = cacheID;
  
}

void ReprocessEcalPedestals::endJob() {

}
