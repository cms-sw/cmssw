#include "EventFilter//Utilities/interface/DAQSourceModelsScoutingRun3.h"

void DataModeScoutingRun3::makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                                                std::vector<int> const& numSources,
                                                std::string const& runDir) {
  std::filesystem::path runDirP(runDir);
  for (auto& baseDir : baseDirs) {
    std::filesystem::path baseDirP(baseDir);
    buPaths_.emplace_back(baseDirP / runDirP);
  }

  // store the number of sources in each BU
  buNumSources_ = numSources;
}

std::pair<bool, std::vector<std::string>> DataModeScoutingRun3::defineAdditionalFiles(std::string const& primaryName,
                                                                                      bool fileListMode) const {
  std::vector<std::string> additionalFiles;

  if (fileListMode) {
    // Expected file naming when working in file list mode
    for (int j = 1; j < buNumSources_[0]; j++) {
      additionalFiles.push_back(primaryName + "_" + std::to_string(j));
    }
    return std::make_pair(true, additionalFiles);
  }

  auto fullpath = std::filesystem::path(primaryName);
  auto fullname = fullpath.filename();

  for (size_t i = 0; i < buPaths_.size(); i++) {
    std::filesystem::path newPath = buPaths_[i] / fullname;

    if (i != 0) {
      // secondary files from other ramdisks
      additionalFiles.push_back(newPath.generic_string());
    }

    // add extra sources from the same ramdisk
    for (int j = 1; j < buNumSources_[i]; j++) {
      additionalFiles.push_back(newPath.generic_string() + "_" + std::to_string(j));
    }
  }
  return std::make_pair(true, additionalFiles);
}

void DataModeScoutingRun3::readEvent(edm::EventPrincipal& eventPrincipal) {
  assert(!events_.empty());

  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  // set provenance helpers
  uint32_t hdrEventID = currOrbit_;
  edm::EventID eventID = edm::EventID(daqSource_->eventRunNumber(), daqSource_->currentLumiSection(), hdrEventID);
  edm::EventAuxiliary aux(
      eventID, daqSource_->processGUID(), tstamp, events_[0]->isRealData(), edm::EventAuxiliary::PhysicsTrigger);

  aux.setProcessHistoryID(daqSource_->processHistoryID());
  daqSource_->makeEventWrapper(eventPrincipal, aux);

  // create scouting raw data collection
  std::unique_ptr<SDSRawDataCollection> rawData(new SDSRawDataCollection);

  // Fill the ScoutingRawDataCollection with valid orbit data from the multiple sources
  for (const auto& pair : sourceValidOrbitPair_) {
    fillSDSRawDataCollection(*rawData, (char*)events_[pair.second]->payload(), events_[pair.second]->eventSize());
  }

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<SDSRawDataCollection>(std::move(rawData)));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->branchDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());

  eventCached_ = false;
}

void DataModeScoutingRun3::fillSDSRawDataCollection(SDSRawDataCollection& rawData, char* buff, size_t len) {
  size_t pos = 0;

  // get the source ID
  int sourceId = *((uint32_t*)(buff + pos));
  pos += 4;

  // size of the orbit paylod
  size_t orbitSize = len - pos;

  // set the size (=orbit size) in the SRDColletion of the current source.
  // FRD size is expecting 8 bytes words, while scouting is using 4 bytes
  // words. This could be different for some future sources.
  FEDRawData& fedData = rawData.FEDData(sourceId);
  fedData.resize(orbitSize, 4);

  memcpy(fedData.data(), buff + pos, orbitSize);

  return;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeScoutingRun3::makeDaqProvenanceHelpers() {
  //set SRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(SDSRawDataCollection)), "SDSRawDataCollection", "SDSRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

bool DataModeScoutingRun3::nextEventView() {
  blockCompleted_ = false;
  if (eventCached_)
    return true;

  // move the data block address only for the sources processed
  // un the previous event by adding the last event size
  for (const auto& pair : sourceValidOrbitPair_) {
    dataBlockAddrs_[pair.first] += events_[pair.second]->size();
  }

  return makeEvents();
}

bool DataModeScoutingRun3::makeEvents() {
  // clear events and reset current orbit
  events_.clear();
  sourceValidOrbitPair_.clear();
  currOrbit_ = 0xFFFFFFFF;  // max uint
  assert(!blockCompleted_);

  // create current "events" (= orbits) list from each data source,
  // check if one dataBlock terminated earlier than others.
  for (int i = 0; i < numFiles_; i++) {
    if (dataBlockAddrs_[i] >= dataBlockMaxAddrs_[i]) {
      completedBlocks_[i] = true;
      continue;
    }

    // event contains data, add it to the events list
    events_.emplace_back(std::make_unique<FRDEventMsgView>(dataBlockAddrs_[i]));
    if (dataBlockAddrs_[i] + events_.back()->size() > dataBlockMaxAddrs_[i])
      throw cms::Exception("DAQSource::getNextEvent")
          << " event id:" << events_.back()->event() << " lumi:" << events_.back()->lumi()
          << " run:" << events_.back()->run() << " of size:" << events_.back()->size()
          << " bytes does not fit into the buffer or has corrupted header";

    // find the minimum orbit for the current event between all files
    if ((events_.back()->event() < currOrbit_) && (!completedBlocks_[i])) {
      currOrbit_ = events_.back()->event();
    }
  }

  // mark valid orbits from each data source
  // e.g. find when orbit is missing from one source
  bool allBlocksCompleted = true;
  int evt_idx = 0;
  for (int i = 0; i < numFiles_; i++) {
    if (completedBlocks_[i]) {
      continue;
    }

    if (events_[evt_idx]->event() != currOrbit_) {
      // current source (=i-th source) doesn't contain the expected orbit.
      // skip it, and move to the next orbit
    } else {
      // add a pair <current surce index, event index>
      // evt_idx can be different from variable i, as some data blocks can be
      // completed before others
      sourceValidOrbitPair_.emplace_back(std::make_pair(i, evt_idx));
      allBlocksCompleted = false;
    }

    evt_idx++;
  }

  if (allBlocksCompleted) {
    blockCompleted_ = true;
  }
  return !allBlocksCompleted;
}

bool DataModeScoutingRun3::checksumValid() { return true; }

std::string DataModeScoutingRun3::getChecksumError() const { return std::string(); }
