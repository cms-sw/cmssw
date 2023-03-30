#include "EventFilter/Utilities/interface/DAQSource.h"
#include "EventFilter/Utilities/interface/DAQSourceModelsFRD.h"

#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/TCDS/interface/TCDSRaw.h"

#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/Utilities/interface/GlobalEventNumber.h"
#include "EventFilter/Utilities/interface/DAQSourceModels.h"
#include "EventFilter/Utilities/interface/DAQSource.h"

#include "EventFilter/Utilities/interface/AuxiliaryMakers.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "EventFilter/Utilities/interface/crc32c.h"

void DataModeFRD::readEvent(edm::EventPrincipal& eventPrincipal) {
  std::unique_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  bool tcdsInRange;
  unsigned char* tcds_pointer = nullptr;
  edm::Timestamp tstamp = fillFEDRawDataCollection(*rawData, tcdsInRange, tcds_pointer);

  if (daqSource_->useL1EventID()) {
    uint32_t L1EventID = event_->event();
    edm::EventID eventID = edm::EventID(daqSource_->eventRunNumber(), daqSource_->currentLumiSection(), L1EventID);
    edm::EventAuxiliary aux(
        eventID, daqSource_->processGUID(), tstamp, event_->isRealData(), edm::EventAuxiliary::PhysicsTrigger);
    aux.setProcessHistoryID(daqSource_->processHistoryID());
    daqSource_->makeEventWrapper(eventPrincipal, aux);
  } else if (tcds_pointer == nullptr) {
    uint32_t L1EventID = event_->event();
    throw cms::Exception("DAQSource::read") << "No TCDS FED in event with FEDHeader EID -: " << L1EventID;
  } else {
    const FEDHeader fedHeader(tcds_pointer);
    tcds::Raw_v1 const* tcds = reinterpret_cast<tcds::Raw_v1 const*>(tcds_pointer + FEDHeader::length);
    edm::EventAuxiliary aux =
        evf::evtn::makeEventAuxiliary(tcds,
                                      daqSource_->eventRunNumber(),      //TODO_ eventRunNumber_
                                      daqSource_->currentLumiSection(),  //currentLumiSection_
                                      event_->isRealData(),
                                      static_cast<edm::EventAuxiliary::ExperimentType>(fedHeader.triggerType()),
                                      daqSource_->processGUID(),
                                      !daqSource_->fileListLoopMode(),
                                      !tcdsInRange);
    aux.setProcessHistoryID(daqSource_->processHistoryID());
    daqSource_->makeEventWrapper(eventPrincipal, aux);
  }

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<FEDRawDataCollection>(std::move(rawData)));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->branchDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
}

edm::Timestamp DataModeFRD::fillFEDRawDataCollection(FEDRawDataCollection& rawData,
                                                     bool& tcdsInRange,
                                                     unsigned char*& tcds_pointer) {
  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  uint32_t eventSize = event_->eventSize();
  unsigned char* event = (unsigned char*)event_->payload();
  tcds_pointer = nullptr;
  tcdsInRange = false;
  uint16_t selectedTCDSFed = 0;
  while (eventSize > 0) {
    assert(eventSize >= FEDTrailer::length);
    eventSize -= FEDTrailer::length;
    const FEDTrailer fedTrailer(event + eventSize);
    const uint32_t fedSize = fedTrailer.fragmentLength() << 3;  //trailer length counts in 8 bytes
    assert(eventSize >= fedSize - FEDHeader::length);
    eventSize -= (fedSize - FEDHeader::length);
    const FEDHeader fedHeader(event + eventSize);
    const uint16_t fedId = fedHeader.sourceID();
    if (fedId > FEDNumbering::MAXFEDID) {
      throw cms::Exception("DAQSource::fillFEDRawDataCollection") << "Out of range FED ID : " << fedId;
    } else if (fedId >= MINTCDSuTCAFEDID_ && fedId <= MAXTCDSuTCAFEDID_) {
      if (!selectedTCDSFed) {
        selectedTCDSFed = fedId;
        tcds_pointer = event + eventSize;
        if (fedId >= FEDNumbering::MINTCDSuTCAFEDID && fedId <= FEDNumbering::MAXTCDSuTCAFEDID) {
          tcdsInRange = true;
        }
      } else
        throw cms::Exception("DAQSource::fillFEDRawDataCollection")
            << "Second TCDS FED ID " << fedId << " found. First ID: " << selectedTCDSFed;
    }
    //take event ID from GTPE FED
    FEDRawData& fedData = rawData.FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), event + eventSize, fedSize);
  }
  assert(eventSize == 0);

  return tstamp;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeFRD::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(FEDRawDataCollection)), "FEDRawDataCollection", "FEDRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

bool DataModeFRD::nextEventView() {
  if (eventCached_)
    return true;
  event_ = std::make_unique<FRDEventMsgView>(dataBlockAddr_);
  if (event_->size() > dataBlockMax_) {
    throw cms::Exception("DAQSource::getNextEvent")
        << " event id:" << event_->event() << " lumi:" << event_->lumi() << " run:" << event_->run()
        << " of size:" << event_->size() << " bytes does not fit into a chunk of size:" << dataBlockMax_ << " bytes";
  }
  return true;
}

bool DataModeFRD::checksumValid() {
  crc_ = 0;
  if (event_->version() >= 5) {
    crc_ = crc32c(crc_, (const unsigned char*)event_->payload(), event_->eventSize());
    if (crc_ != event_->crc32c())
      return false;
  }
  return true;
}

std::string DataModeFRD::getChecksumError() const {
  std::stringstream ss;
  ss << "Found a wrong crc32c checksum: expected 0x" << std::hex << event_->crc32c() << " but calculated 0x" << crc_;
  return ss.str();
}

/*
 * FRD Multi Test
 */

void DataModeFRDStriped::makeDirectoryEntries(std::vector<std::string> const& baseDirs, std::string const& runDir) {
  std::filesystem::path runDirP(runDir);
  for (auto& baseDir : baseDirs) {
    std::filesystem::path baseDirP(baseDir);
    buPaths_.emplace_back(baseDirP / runDirP);
  }
}

void DataModeFRDStriped::readEvent(edm::EventPrincipal& eventPrincipal) {
  assert(!events_.empty());
  std::unique_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  bool tcdsInRange;
  unsigned char* tcds_pointer = nullptr;
  edm::Timestamp tstamp = fillFRDCollection(*rawData, tcdsInRange, tcds_pointer);

  auto const& event = events_[0];
  if (daqSource_->useL1EventID()) {
    uint32_t L1EventID = event->event();
    edm::EventID eventID = edm::EventID(daqSource_->eventRunNumber(), daqSource_->currentLumiSection(), L1EventID);
    edm::EventAuxiliary aux(
        eventID, daqSource_->processGUID(), tstamp, event->isRealData(), edm::EventAuxiliary::PhysicsTrigger);
    aux.setProcessHistoryID(daqSource_->processHistoryID());
    daqSource_->makeEventWrapper(eventPrincipal, aux);
  } else if (tcds_pointer == nullptr) {
    uint32_t L1EventID = event->event();
    throw cms::Exception("DAQSource::read") << "No TCDS FED in event with FEDHeader EID -: " << L1EventID;
  } else {
    const FEDHeader fedHeader(tcds_pointer);
    tcds::Raw_v1 const* tcds = reinterpret_cast<tcds::Raw_v1 const*>(tcds_pointer + FEDHeader::length);
    edm::EventAuxiliary aux =
        evf::evtn::makeEventAuxiliary(tcds,
                                      daqSource_->eventRunNumber(),
                                      daqSource_->currentLumiSection(),
                                      event->isRealData(),
                                      static_cast<edm::EventAuxiliary::ExperimentType>(fedHeader.triggerType()),
                                      daqSource_->processGUID(),
                                      !daqSource_->fileListLoopMode(),
                                      !tcdsInRange);
    aux.setProcessHistoryID(daqSource_->processHistoryID());
    daqSource_->makeEventWrapper(eventPrincipal, aux);
  }
  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<FEDRawDataCollection>(std::move(rawData)));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->branchDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
  eventCached_ = false;
}

edm::Timestamp DataModeFRDStriped::fillFRDCollection(FEDRawDataCollection& rawData,
                                                     bool& tcdsInRange,
                                                     unsigned char*& tcds_pointer) {
  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  tcds_pointer = nullptr;
  tcdsInRange = false;
  uint16_t selectedTCDSFed = 0;
  int selectedTCDSFileIndex = -1;
  for (size_t index = 0; index < events_.size(); index++) {
    uint32_t eventSize = events_[index]->eventSize();
    unsigned char* event = (unsigned char*)events_[index]->payload();
    while (eventSize > 0) {
      assert(eventSize >= FEDTrailer::length);
      eventSize -= FEDTrailer::length;
      const FEDTrailer fedTrailer(event + eventSize);
      const uint32_t fedSize = fedTrailer.fragmentLength() << 3;  //trailer length counts in 8 bytes
      assert(eventSize >= fedSize - FEDHeader::length);
      eventSize -= (fedSize - FEDHeader::length);
      const FEDHeader fedHeader(event + eventSize);
      const uint16_t fedId = fedHeader.sourceID();
      if (fedId > FEDNumbering::MAXFEDID) {
        throw cms::Exception("DataModeFRDStriped:::fillFRDCollection") << "Out of range FED ID : " << fedId;
      } else if (fedId >= MINTCDSuTCAFEDID_ && fedId <= MAXTCDSuTCAFEDID_) {
        if (!selectedTCDSFed) {
          selectedTCDSFed = fedId;
          selectedTCDSFileIndex = index;
          tcds_pointer = event + eventSize;
          if (fedId >= FEDNumbering::MINTCDSuTCAFEDID && fedId <= FEDNumbering::MAXTCDSuTCAFEDID) {
            tcdsInRange = true;
          }
        } else if (!testing_)
          throw cms::Exception("DataModeFRDStriped:::fillFRDCollection")
              << "Second TCDS FED ID " << fedId << " found in file " << selectedTCDSFileIndex
              << ". First ID: " << selectedTCDSFed << " in file " << index;
      }
      FEDRawData& fedData = rawData.FEDData(fedId);
      fedData.resize(fedSize);
      memcpy(fedData.data(), event + eventSize, fedSize);
    }
    assert(eventSize == 0);
  }

  return tstamp;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeFRDStriped::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(FEDRawDataCollection)), "FEDRawDataCollection", "FEDRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

/* TODO: adapt to multi-fils
bool DataModeFRD::nextEventView() {
  if (eventCached_) return true;
  event_ = std::make_unique<FRDEventMsgView>(dataBlockAddr_);
  if (event_->size() > dataBlockMax_) {
    throw cms::Exception("DAQSource::getNextEvent")
      << " event id:" << event_->event() << " lumi:" << event_->lumi() << " run:" << event_->run()
      << " of size:" << event_->size() << " bytes does not fit into a chunk of size:" << dataBlockMax_
      << " bytes";
  }
  return true;
}
*/

bool DataModeFRDStriped::checksumValid() {
  bool status = true;
  for (size_t i = 0; i < events_.size(); i++) {
    uint32_t crc = 0;
    auto const& event = events_[i];
    if (event->version() >= 5) {
      crc = crc32c(crc, (const unsigned char*)event->payload(), event->eventSize());
      if (crc != event->crc32c()) {
        std::ostringstream ss;
        ss << "Found a wrong crc32c checksum at readout index " << i << ": expected 0x" << std::hex << event->crc32c()
           << " but calculated 0x" << crc << ". ";
        crcMsg_ += ss.str();
        status = false;
      }
    }
  }
  return status;
}

std::string DataModeFRDStriped::getChecksumError() const { return crcMsg_; }

/*
  read multiple input files for this model
*/

std::pair<bool, std::vector<std::string>> DataModeFRDStriped::defineAdditionalFiles(std::string const& primaryName,
                                                                                    bool fileListMode) const {
  std::vector<std::string> additionalFiles;

  if (fileListMode) {
    //for the unit test
    additionalFiles.push_back(primaryName + "_1");
    return std::make_pair(true, additionalFiles);
  }

  auto fullpath = std::filesystem::path(primaryName);
  auto fullname = fullpath.filename();

  for (size_t i = 1; i < buPaths_.size(); i++) {
    std::filesystem::path newPath = buPaths_[i] / fullname;
    additionalFiles.push_back(newPath.generic_string());
  }
  return std::make_pair(true, additionalFiles);
}

bool DataModeFRDStriped::nextEventView() {
  blockCompleted_ = false;
  if (eventCached_)
    return true;
  for (unsigned int i = 0; i < events_.size(); i++) {
    //add last event length to each stripe
    dataBlockAddrs_[i] += events_[i]->size();
  }
  return makeEvents();
}

bool DataModeFRDStriped::makeEvents() {
  events_.clear();
  assert(!blockCompleted_);
  for (int i = 0; i < numFiles_; i++) {
    if (dataBlockAddrs_[i] >= dataBlockMaxAddrs_[i]) {
      //must be exact
      assert(dataBlockAddrs_[i] == dataBlockMaxAddrs_[i]);
      blockCompleted_ = true;
      return false;
    } else {
      if (blockCompleted_)
        throw cms::Exception("DataModeFRDStriped::makeEvents")
            << "not all striped blocks were completed at the same time";
    }
    if (blockCompleted_)
      continue;
    events_.emplace_back(std::make_unique<FRDEventMsgView>(dataBlockAddrs_[i]));
    if (dataBlockAddrs_[i] + events_[i]->size() > dataBlockMaxAddrs_[i])
      throw cms::Exception("DAQSource::getNextEvent")
          << " event id:" << events_[i]->event() << " lumi:" << events_[i]->lumi() << " run:" << events_[i]->run()
          << " of size:" << events_[i]->size() << " bytes does not fit into the buffer or has corrupted header";
  }
  return !blockCompleted_;
}
