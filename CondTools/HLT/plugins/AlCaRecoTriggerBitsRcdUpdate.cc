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
#include <algorithm>
#include <set>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// What I want to write:
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
// Rcd for reading old one:
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"

class AlCaRecoTriggerBitsRcdUpdate : public edm::one::EDAnalyzer<> {
public:
  explicit AlCaRecoTriggerBitsRcdUpdate(const edm::ParameterSet &cfg);
  ~AlCaRecoTriggerBitsRcdUpdate() override = default;

  void analyze(const edm::Event &evt, const edm::EventSetup &evtSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  typedef std::map<std::string, std::string> TriggerMap;
  bool removeKeysFromMap(const std::vector<std::string> &keys, TriggerMap &triggerMap) const;
  bool replaceKeysFromMap(const std::vector<edm::ParameterSet> &alcarecoReplace, TriggerMap &triggerMap) const;
  bool addTriggerLists(const std::vector<edm::ParameterSet> &triggerListsAdd, AlCaRecoTriggerBits &bits) const;
  bool addPathsFromMap(const std::vector<edm::ParameterSet> &pathsToAdd, AlCaRecoTriggerBits &bits) const;
  bool removePathsFromMap(const std::vector<edm::ParameterSet> &pathsToRemove, AlCaRecoTriggerBits &bits) const;
  void writeBitsToDB(const AlCaRecoTriggerBits &bitsToWrite) const;

  const edm::ESGetToken<AlCaRecoTriggerBits, AlCaRecoTriggerBitsRcd> triggerBitsToken_;
  unsigned int nEventCalls_;
  const unsigned int firstRunIOV_;
  const int lastRunIOV_;
  const bool startEmpty_;
  const std::vector<std::string> listNamesRemove_;
  const std::vector<edm::ParameterSet> triggerListsAdd_;
  const std::vector<edm::ParameterSet> alcarecoReplace_;
  const std::vector<edm::ParameterSet> pathsToAdd_;
  const std::vector<edm::ParameterSet> pathsToRemove_;
};

///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdUpdate::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Plugin to write payloads of type AlCaRecoTriggerBits");
  desc.add<unsigned int>("firstRunIOV", 1);
  desc.add<int>("lastRunIOV", -1);
  desc.add<bool>("startEmpty", true);
  desc.add<std::vector<std::string>>("listNamesRemove", {});

  edm::ParameterSetDescription desc_triggeListsToAdd;
  desc_triggeListsToAdd.add<std::string>("listName");
  desc_triggeListsToAdd.add<std::vector<std::string>>("hltPaths");
  std::vector<edm::ParameterSet> default_triggerListsToAdd;
  desc.addVPSet("triggerListsAdd", desc_triggeListsToAdd, default_triggerListsToAdd);

  edm::ParameterSetDescription desc_alcarecoToReplace;
  desc_alcarecoToReplace.add<std::string>("oldKey");
  desc_alcarecoToReplace.add<std::string>("newKey");
  std::vector<edm::ParameterSet> default_alcarecoToReplace;
  desc.addVPSet("alcarecoToReplace", desc_alcarecoToReplace, default_alcarecoToReplace);

  edm::ParameterSetDescription desc_pathsToAdd;
  desc_pathsToAdd.add<std::string>("listName");
  desc_pathsToAdd.add<std::vector<std::string>>("hltPaths");
  std::vector<edm::ParameterSet> default_pathsToAdd;
  desc.addVPSet("pathsToAdd", desc_pathsToAdd, default_pathsToAdd);

  edm::ParameterSetDescription desc_pathsToRemove;
  desc_pathsToRemove.add<std::string>("listName");
  desc_pathsToRemove.add<std::vector<std::string>>("hltPaths");
  std::vector<edm::ParameterSet> default_pathsToRemove;
  desc.addVPSet("pathsToRemove", desc_pathsToRemove, default_pathsToRemove);

  descriptions.addWithDefaultLabel(desc);
}

///////////////////////////////////////////////////////////////////////
AlCaRecoTriggerBitsRcdUpdate::AlCaRecoTriggerBitsRcdUpdate(const edm::ParameterSet &cfg)
    : triggerBitsToken_(esConsumes()),
      nEventCalls_(0),
      firstRunIOV_(cfg.getParameter<unsigned int>("firstRunIOV")),
      lastRunIOV_(cfg.getParameter<int>("lastRunIOV")),
      startEmpty_(cfg.getParameter<bool>("startEmpty")),
      listNamesRemove_(cfg.getParameter<std::vector<std::string>>("listNamesRemove")),
      triggerListsAdd_(cfg.getParameter<std::vector<edm::ParameterSet>>("triggerListsAdd")),
      alcarecoReplace_(cfg.getParameter<std::vector<edm::ParameterSet>>("alcarecoToReplace")),
      pathsToAdd_(cfg.getParameter<std::vector<edm::ParameterSet>>("pathsToAdd")),
      pathsToRemove_(cfg.getParameter<std::vector<edm::ParameterSet>>("pathsToRemove")) {}

///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdUpdate::analyze(const edm::Event &evt, const edm::EventSetup &iSetup) {
  if (nEventCalls_++ > 0) {  // postfix increment!
    edm::LogWarning("BadConfig") << "@SUB=analyze"
                                 << "Writing to DB to be done only once, set\n"
                                 << "'process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))'\n"
                                 << " next time. But your writing is fine.)";
    return;
  }

  // create what to write - starting from empty or existing list
  std::unique_ptr<AlCaRecoTriggerBits> bitsToWrite;
  if (startEmpty_) {
    bitsToWrite = std::make_unique<AlCaRecoTriggerBits>();
  } else {
    bitsToWrite = std::make_unique<AlCaRecoTriggerBits>(iSetup.getData(triggerBitsToken_));
  }

  // remove some existing entries in map
  this->removeKeysFromMap(listNamesRemove_, bitsToWrite->m_alcarecoToTrig);

  // now add new entries
  this->addTriggerLists(triggerListsAdd_, *bitsToWrite);

  // now replace keys
  this->replaceKeysFromMap(alcarecoReplace_, bitsToWrite->m_alcarecoToTrig);

  // add paths to the existing key
  this->addPathsFromMap(pathsToAdd_, *bitsToWrite);

  // remove paths from the existing key
  this->removePathsFromMap(pathsToRemove_, *bitsToWrite);

  // finally write to DB
  this->writeBitsToDB(*bitsToWrite);
}

///////////////////////////////////////////////////////////////////////
bool AlCaRecoTriggerBitsRcdUpdate::removeKeysFromMap(const std::vector<std::string> &keys,
                                                     TriggerMap &triggerMap) const {
  for (std::vector<std::string>::const_iterator iKey = keys.begin(), endKey = keys.end(); iKey != endKey; ++iKey) {
    if (triggerMap.find(*iKey) != triggerMap.end()) {
      triggerMap.erase(*iKey);
    } else {  // not in list ==> misconfiguartion!
      throw cms::Exception("BadConfig") << "[AlCaRecoTriggerBitsRcdUpdate::removeKeysFromMap] "
                                        << "Cannot remove key '" << *iKey << "' since not in "
                                        << "list - typo in configuration?\n";
      return false;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////
bool AlCaRecoTriggerBitsRcdUpdate::replaceKeysFromMap(const std::vector<edm::ParameterSet> &alcarecoReplace,
                                                      TriggerMap &triggerMap) const {
  std::vector<std::pair<std::string, std::string>> keyPairs;
  keyPairs.reserve(alcarecoReplace.size());

  for (auto &iSet : alcarecoReplace) {
    const std::string oldKey(iSet.getParameter<std::string>("oldKey"));
    const std::string newKey(iSet.getParameter<std::string>("newKey"));
    keyPairs.push_back(std::make_pair(oldKey, newKey));
  }

  for (auto &iKey : keyPairs) {
    if (triggerMap.find(iKey.first) != triggerMap.end()) {
      std::string bitsToReplace = triggerMap[iKey.first];
      triggerMap.erase(iKey.first);
      triggerMap[iKey.second] = bitsToReplace;
    } else {  // not in list ==> misconfiguration!
      edm::LogWarning("AlCaRecoTriggerBitsRcdUpdate")
          << "[AlCaRecoTriggerBitsRcdUpdate::replaceKeysFromMap] "
          << "Cannot replace key '" << iKey.first << "with " << iKey.second << " since not in "
          << "list - typo in configuration?\n";
      return false;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////
bool AlCaRecoTriggerBitsRcdUpdate::addTriggerLists(const std::vector<edm::ParameterSet> &triggerListsAdd,
                                                   AlCaRecoTriggerBits &bits) const {
  TriggerMap &triggerMap = bits.m_alcarecoToTrig;

  // loop on PSets, each containing the key (filter name) and a vstring with triggers
  for (const auto &iSet : triggerListsAdd) {
    const std::vector<std::string> paths(iSet.getParameter<std::vector<std::string>>("hltPaths"));
    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so we have to merge the paths into one string that will be decoded when needed:
    const std::string mergedPaths = bits.compose(paths);

    const std::string filter(iSet.getParameter<std::string>("listName"));
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
bool AlCaRecoTriggerBitsRcdUpdate::addPathsFromMap(const std::vector<edm::ParameterSet> &pathsToAdd,
                                                   AlCaRecoTriggerBits &bits) const {
  TriggerMap &triggerMap = bits.m_alcarecoToTrig;  //read from the condition tag

  // loop on PSets, each containing the key (filter name) and a vstring with triggers
  for (const auto &iSet : pathsToAdd) {
    const std::string filter(iSet.getParameter<std::string>("listName"));
    std::string mergedPathsInKey;

    for (const auto &imap : triggerMap) {
      if (imap.first == filter)
        mergedPathsInKey = imap.second;  //paths in the condition tag
    }

    if (mergedPathsInKey.empty()) {
      throw cms::Exception("BadConfig") << "List name '" << filter << "' not found in the map, "
                                        << "if you want to add new key/paths, please use 'addTriggerLists'.\n";
    }

    auto const &pathsInKey = bits.decompose(mergedPathsInKey);
    auto const &paths = iSet.getParameter<std::vector<std::string>>("hltPaths");  //paths to add; from the configuration

    if (paths.empty()) {  // nothing to add ==> misconfiguration!
      throw cms::Exception("BadConfig") << "Didn't set any path to add!";
    }

    std::set<std::string> pathsSet{pathsInKey.begin(), pathsInKey.end()};
    std::copy(paths.begin(), paths.end(), std::inserter(pathsSet, pathsSet.end()));
    std::vector<std::string> const newPathsInKey{pathsSet.begin(), pathsSet.end()};

    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so we have to merge the paths into one string that will be decoded when needed:
    triggerMap[filter] = bits.compose(newPathsInKey);
  }

  return true;
}

///////////////////////////////////////////////////////////////////////
bool AlCaRecoTriggerBitsRcdUpdate::removePathsFromMap(const std::vector<edm::ParameterSet> &pathsToRemove,
                                                      AlCaRecoTriggerBits &bits) const {
  TriggerMap &triggerMap = bits.m_alcarecoToTrig;  //read from the condition tag

  // loop on PSets, each containing the key (filter name) and a vstring with triggers
  for (const auto &iSet : pathsToRemove) {
    const std::string filter(iSet.getParameter<std::string>("listName"));
    std::string mergedPathsInKey;

    for (const auto &imap : triggerMap) {
      if (imap.first == filter)
        mergedPathsInKey = imap.second;  //paths in the condition tag
    }

    if (mergedPathsInKey.empty()) {
      throw cms::Exception("BadConfig") << "List name '" << filter << "' not found in the map";
    }

    auto PathsInKey = bits.decompose(mergedPathsInKey);
    auto const paths(
        iSet.getParameter<std::vector<std::string>>("hltPaths"));  //paths to remove; from the configuration

    if (paths.empty()) {  // nothing to remove ==> misconfiguration!
      throw cms::Exception("BadConfig") << "Didn't set any path to remove!";
    }

    for (auto const &path : paths) {
      PathsInKey.erase(std::remove(PathsInKey.begin(), PathsInKey.end(), path), PathsInKey.end());
    }

    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so we have to merge the paths into one string that will be decoded when needed:
    triggerMap[filter] = bits.compose(PathsInKey);
  }

  return true;
}

///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdUpdate::writeBitsToDB(const AlCaRecoTriggerBits &bitsToWrite) const {
  edm::LogInfo("AlCaRecoTriggerBitsRcdUpdate") << "Uploading to the database...";

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable()) {
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available.\n";
  }

  const std::string recordName("AlCaRecoTriggerBitsRcd");

  // when updating existing tag, compare payload hashs and skip appending new hash if it's same with last iov's
  poolDbService->startTransaction();
  auto newHash = poolDbService->session().storePayload(bitsToWrite);
  cond::TagInfo_t tag_info;

  if (poolDbService->tagInfo(recordName, tag_info)) {
    if (newHash != tag_info.lastInterval.payloadId) {
      edm::LogInfo("AlCaRecoTriggerBitsRcdUpdate") << "## Appending to existing tag...";
      poolDbService->forceInit();
      poolDbService->appendSinceTime(newHash, firstRunIOV_, recordName);
    } else {
      edm::LogInfo("AlCaRecoTriggerBitsRcdUpdate") << "## Skipping update since hash is the same...";
    }

  } else {
    edm::LogInfo("AlCaRecoTriggerBitsRcdUpdate") << "## Creating new tag...";
    poolDbService->forceInit();
    poolDbService->createNewIOV(newHash, firstRunIOV_, recordName);
  }
  poolDbService->commitTransaction();

  edm::LogInfo("AlCaRecoTriggerBitsRcdUpdate")
      << "...done for runs " << firstRunIOV_ << " to " << lastRunIOV_ << " (< 0 meaning infinity)!";
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlCaRecoTriggerBitsRcdUpdate);
