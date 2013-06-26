///////////////////////////////////////////////////////////////////////
///
/// Module to write trigger bit mappings (AlCaRecoTriggerBits) to DB.
/// Can be configured to read an old one and update this by 
/// - removing old entries
/// - adding new ones
///
///////////////////////////////////////////////////////////////////////

#include <string>
#include <map>
#include <vector>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// What I want to write:
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
// Rcd for reading old one:
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"


class  AlCaRecoTriggerBitsRcdUpdate : public edm::EDAnalyzer {
public:
  explicit  AlCaRecoTriggerBitsRcdUpdate(const edm::ParameterSet& cfg);
  ~AlCaRecoTriggerBitsRcdUpdate() {}
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
  
private:
  typedef std::map<std::string, std::string> TriggerMap;
  AlCaRecoTriggerBits* createStartTriggerBits(bool startEmpty, const edm::EventSetup& evtSetup) const;
  bool removeKeysFromMap(const std::vector<std::string> &keys, TriggerMap &triggerMap) const;
  bool addTriggerLists(const std::vector<edm::ParameterSet> &triggerListsAdd,
		       AlCaRecoTriggerBits &bits) const;
  /// Takes over memory uresponsibility for 'bitsToWrite'. 
  void writeBitsToDB(AlCaRecoTriggerBits *bitsToWrite) const;

  unsigned int nEventCalls_;
  const unsigned int firstRunIOV_;
  const int lastRunIOV_;
  const bool startEmpty_; 
  const std::vector<std::string> listNamesRemove_;
  const std::vector<edm::ParameterSet> triggerListsAdd_;
};

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

AlCaRecoTriggerBitsRcdUpdate::AlCaRecoTriggerBitsRcdUpdate(const edm::ParameterSet& cfg)
  : nEventCalls_(0), 
    firstRunIOV_(cfg.getParameter<unsigned int>("firstRunIOV")),
    lastRunIOV_(cfg.getParameter<int>("lastRunIOV")),
    startEmpty_(cfg.getParameter<bool>("startEmpty")),
    listNamesRemove_(cfg.getParameter<std::vector<std::string> >("listNamesRemove")),
    triggerListsAdd_(cfg.getParameter<std::vector<edm::ParameterSet> >("triggerListsAdd"))
{
}
  
///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdUpdate::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
{
  if (nEventCalls_++ > 0) { // postfix increment!
    edm::LogWarning("BadConfig")
      << "@SUB=analyze" << "Writing to DB to be done only once, set\n"
      << "'process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))'\n"
      << " next time. But your writing is fine.)";
    return;
  }

  // create what to write - starting from empty or existing list (auto_ptr?)
  AlCaRecoTriggerBits *bitsToWrite = this->createStartTriggerBits(startEmpty_, iSetup);

  // remove some existing entries in map 
  this->removeKeysFromMap(listNamesRemove_, bitsToWrite->m_alcarecoToTrig);

  // now add new entries
  this->addTriggerLists(triggerListsAdd_, *bitsToWrite);

  // finally write to DB
  this->writeBitsToDB(bitsToWrite);

}

///////////////////////////////////////////////////////////////////////
AlCaRecoTriggerBits*  // auto_ptr?
AlCaRecoTriggerBitsRcdUpdate::createStartTriggerBits(bool startEmpty,
						     const edm::EventSetup& evtSetup) const
{
  if (startEmpty) {
    return new AlCaRecoTriggerBits;
  } else {
    edm::ESHandle<AlCaRecoTriggerBits> triggerBits;
    evtSetup.get<AlCaRecoTriggerBitsRcd>().get(triggerBits);
    return new AlCaRecoTriggerBits(*triggerBits); // copy old one
  }
}

///////////////////////////////////////////////////////////////////////
bool AlCaRecoTriggerBitsRcdUpdate::removeKeysFromMap(const std::vector<std::string> &keys,
						     TriggerMap &triggerMap) const
{
  for (std::vector<std::string>::const_iterator iKey = keys.begin(), endKey = keys.end(); 
       iKey != endKey; ++iKey) {
    if (triggerMap.find(*iKey) != triggerMap.end()) {
      // remove
//      edm::LogError("Temp") << "@SUB=removeKeysFromMap" << "Cannot yet remove '" << *iKey
// 			    << "' from map.";
// FIXME: test next line@
      triggerMap.erase(*iKey);
    } else { // not in list ==> misconfiguartion!
      throw cms::Exception("BadConfig") << "[AlCaRecoTriggerBitsRcdUpdate::removeKeysFromMap] "
					<< "Cannot remove key '" << *iKey << "' since not in "
					<< "list - typo in configuration?\n";
      return false;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////
bool AlCaRecoTriggerBitsRcdUpdate::addTriggerLists(const std::vector<edm::ParameterSet> &triggerListsAdd,
						   AlCaRecoTriggerBits &bits) const
{
  TriggerMap &triggerMap = bits.m_alcarecoToTrig;

  // loop on PSets, each containing the key (filter name) and a vstring with triggers
  for (std::vector<edm::ParameterSet>::const_iterator iSet = triggerListsAdd.begin();
       iSet != triggerListsAdd.end(); ++iSet) {
    
    const std::vector<std::string> paths(iSet->getParameter<std::vector<std::string> >("hltPaths"));
    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so we have to merge the paths into one string that will be decoded when needed:
    const std::string mergedPaths = bits.compose(paths);
    
    const std::string filter(iSet->getParameter<std::string>("listName"));
    if (triggerMap.find(filter) != triggerMap.end()) {
      throw cms::Exception("BadConfig") << "List name '" << filter << "' already in map, either " 
                                        << "remove from 'triggerListsAdd' or "
					<< " add to 'listNamesRemove'.\n";
    }
    triggerMap[filter] = mergedPaths;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdUpdate::writeBitsToDB(AlCaRecoTriggerBits *bitsToWrite) const
{
  edm::LogInfo("") << "Uploading to the database...";
  
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable()) {
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available.\n";
  }

  // ownership of bitsToWrite transferred
  // FIXME: Have to check that timetype is run number! How?
  const std::string recordName("AlCaRecoTriggerBitsRcd");
  if (poolDbService->isNewTagRequest(recordName)) { // tag not yet existing
    // lastRunIOV_ = -1 means infinity:
    const cond::Time_t lastRun = (lastRunIOV_ < 0 ? poolDbService->endOfTime() : lastRunIOV_);
    poolDbService->createNewIOV(bitsToWrite, firstRunIOV_, lastRun, recordName);
  } else { // tag exists, can only append
    if (lastRunIOV_ >= 0) {
      throw cms::Exception("BadConfig") << "Tag already exists, can only append until infinity,"
                                        << " but lastRunIOV = " << lastRunIOV_ << ".\n";
    }
    poolDbService->appendSinceTime(bitsToWrite, firstRunIOV_, recordName);
  }  
  
  edm::LogInfo("") << "...done for runs " << firstRunIOV_ << " to " << lastRunIOV_ 
                   << " (< 0 meaning infinity)!";
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlCaRecoTriggerBitsRcdUpdate);
