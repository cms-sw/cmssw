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
#include <fstream>



ReprocessEcalPedestals::ReprocessEcalPedestals(const edm::ParameterSet& iConfig) {
  m_cacheID = 99199; // Will not be the first one returned (we hope)
  m_isFirstRun = true;
  m_startFile = iConfig.getUntrackedParameter< std::string >("startfile", "");
  m_endFile = iConfig.getUntrackedParameter< std::string >("endfile", "");
  m_appendMode = iConfig.getParameter< bool > ("append");
  bool runUnsafe = iConfig.getUntrackedParameter< bool > ("unsafe", false);

  // Sanity checks
  if (!runUnsafe && m_endFile == "") {
    throw(cms::Exception("No endfile provided to use for further reprocessing.  Use unsafe mode if you are sure about this!"));
  }
  
  if (!runUnsafe && m_appendMode && m_startFile == "") {
    throw(cms::Exception("No startfile provided to initialize first set of EcalPedstals.  Use unsafe mode if you are sure about this!"));
  }

  // Initialize the pedCache with values from the file
  if (m_startFile != "") {
    loadCacheFromFile(m_startFile);
  }
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

  if (m_isFirstRun) {
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
  } else {
    // Write the cache to the DB.
    // If appendIOV is false, then the till time of the cache is
    // currentTime-1.
    // If appendIOV is true, then the till time of the PREVIOUS iov in
    // the target tag is set to set to currentTime-1, but the cache is 
    // written with an infinite iov.
    // We use curretnTime-1 because we are one run past where the
    // data changed in the source tag.
    dbOutput->newValidityForNewPayload<EcalPedestals>(new EcalPedestals(m_pedCache), currentTime-1, callbackToken);
    edm::LogInfo("ReprocessEcalPedestals") << "Wrote new set of pedestals for run " << currentTime-1;
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

  LogDebug("ReprocessEcalPedestals") << msg.str();

  // Check that we have #sm * 1700 channels
  if (m_pedCache.m_pedestals.size() != m_smSet.size() * 1700) {
    throw cms::Exception("Pedestals object was not filled with enough channels");
  }

  m_cacheID = cacheID;
  m_isFirstRun = false;
}



void ReprocessEcalPedestals::endJob() {

  if (!m_appendMode) {
    // Write the last cache of pedestals to the DB with an infinite IOV
    edm::Service<cond::service::PoolDBOutputService> dbOutput;
    if ( !dbOutput.isAvailable() ) {
      edm::LogError("ReprocessEcalPedestals") << "PoolDBOUtputService is not available";
      return;
    }
    size_t callbackToken = dbOutput->callbackToken("EcalPedestals");
    dbOutput->newValidityForNewPayload<EcalPedestals>(new EcalPedestals(m_pedCache), dbOutput->endOfTime(), callbackToken);
    edm::LogInfo("ReprocessEcalPedestals") << "Wrote new set of pedestals for run (inf)";
  } 
    
  // Write the pedestals in the cache to a file
  if (m_endFile != "") {
    dumpCacheToFile(m_endFile);
  }
}



void ReprocessEcalPedestals::loadCacheFromFile(std::string file) {
  edm::LogInfo("ReprocessEcalPedestals") << "Initializing cache with pedestals from "
					 << file;
  std::ifstream fileIn(file.c_str());
  if (!fileIn.good()) {
    throw cms::Exception("Could not open file for input");
  }
  
  uint32_t rawId;
  EcalPedestals::Item item;
  EBDetId detId;
  std::ostringstream msg;
  while (fileIn.good()) {
    fileIn >> rawId 
		>> item.mean_x1 >> item.rms_x1 
		>> item.mean_x6 >> item.rms_x6 
		>> item.mean_x12 >> item.rms_x12;
    if (fileIn.eof()) { break; }
    
    detId = EBDetId(rawId);
    msg << "Setting cache SM " << detId.ism() << " crystal " << detId.ic() 
	<< " from file" << std::endl;
    m_pedCache.m_pedestals[rawId] = item;
  }
  fileIn.close();
  LogDebug("ReprocessEcalPedestals") << msg.str();
}



void ReprocessEcalPedestals::dumpCacheToFile(std::string file) {
  edm::LogInfo("ReprocessEcalPedestals") << "Writing cache to "
					 << file;
  std::ofstream fileOut(file.c_str(), ios::out);
  if (!fileOut.good()) {
    throw cms::Exception("Could not open file for output");
  }
  
  uint32_t rawId;
  EcalPedestals::Item item;
  EBDetId detId;
  std::ostringstream msg;
  EcalPedestalsMapIterator detId_iter;
  for (detId_iter = m_pedCache.m_pedestals.begin(); detId_iter != m_pedCache.m_pedestals.end(); detId_iter++) {
    rawId = (*detId_iter).first;
    item = (*detId_iter).second;
    detId = EBDetId(rawId);
    msg << "Dumping SM " << detId.ism() << " crystal " << detId.ic() 
	<< " to file" << std::endl;
    fileOut << rawId << " "
	       << item.mean_x1 << " " << item.rms_x1 << " "
	       << item.mean_x6 << " " << item.rms_x6 << " "
	       << item.mean_x12 << " " << item.rms_x12 << std::endl;
  }
  fileOut.close();
  LogDebug("ReprocessEcalPedestals") << msg.str();
}
