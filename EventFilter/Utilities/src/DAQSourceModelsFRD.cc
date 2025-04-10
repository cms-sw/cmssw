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

using namespace edm::streamer;

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
      daqProvenanceHelpers_[0]->productDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
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
  unsigned int fedsInEvent = 0;
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

    fedsInEvent++;
    if (verifyFEDs_ || !expectedFedsInEvent_) {
      if (fedIdSet_.find(fedId) == fedIdSet_.end()) {
        if (expectedFedsInEvent_)
          throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection") << "FED Id: " << fedId << " was not found in previous events";
        else
          fedIdSet_.insert(fedId);
      }
    }
  }
  assert(eventSize == 0);

  if (!fedsInEvent)
    throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
      << "Event " << event_->event() << " does not contain any FEDs";
  else if (!expectedFedsInEvent_) {
    expectedFedsInEvent_ = fedsInEvent;
    if (fedIdSet_.size() != fedsInEvent)
      throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
        << "First received event: " << event_->event() << " contains duplicate FEDs";
  }
  else if (fedsInEvent != expectedFedsInEvent_)
    throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
      << "Event " << event_->event() << " does not contain same number of FEDs as previous: " << fedsInEvent << "/" << expectedFedsInEvent_;

  return tstamp;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeFRD::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(FEDRawDataCollection)), "FEDRawDataCollection", "FEDRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

void DataModeFRD::makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) {
  dataBlockAddr_ = addr;
  dataBlockMax_ = rawFile->currentChunkSize();
  eventCached_ = false;
  nextEventView(rawFile);
  eventCached_ = true;
}

bool DataModeFRD::nextEventView(RawInputFile*) {
  if (eventCached_)
    return true;
  event_ = std::make_unique<FRDEventMsgView>(dataBlockAddr_);
  if (event_->size() > dataBlockMax_) {
    throw cms::Exception("DAQSource::getNextEvent")
        << " event id:" << event_->event() << " lumi:" << event_->lumi() << " run:" << event_->run()
        << " of size:" << event_->size() << " bytes does not fit into a chunk of size:" << dataBlockMax_ << " bytes";
  }
  if (event_->version() < 5)
    throw cms::Exception("DAQSource::getNextEvent")
        << "Unsupported FRD version " << event_->version() << ". Minimum supported is v5.";
  return true;
}

bool DataModeFRD::checksumValid() {
  crc_ = 0;
  crc_ = crc32c(crc_, (const unsigned char*)event_->payload(), event_->eventSize());
  if (crc_ != event_->crc32c())
    return false;
  else
    return true;
}

std::string DataModeFRD::getChecksumError() const {
  std::stringstream ss;
  ss << "Found a wrong crc32c checksum: expected 0x" << std::hex << event_->crc32c() << " but calculated 0x" << crc_;
  return ss.str();
}

/*
 * FRD preRead
 */

void DataModeFRDPreUnpack::unpackEvent(edm::streamer::FRDEventMsgView* eview, UnpackedRawEventWrapper* ec, unsigned int ls) {
  //TODO: also walk the file and build checksum
  FEDRawDataCollection* rawData = new FEDRawDataCollection;
  bool tcdsInRange;
  unsigned char* tcds_pointer = nullptr;
  std::string errmsg;
  bool err = false;
  edm::Timestamp tstamp = fillFEDRawDataCollection(eview, *rawData, tcdsInRange, tcds_pointer, err, errmsg);
  ec->setRawData(rawData);

  uint32_t L1EventID = eview->event();
  if (err) {
    ec->setError(errmsg);
  } else if (daqSource_->useL1EventID()) {
    //filelist mode run override not available with this model currently (source sets it too late)
    edm::EventID eventID = edm::EventID(ec->run(), ls, L1EventID);
    ec->setAux(new edm::EventAuxiliary(
        eventID, daqSource_->processGUID(), tstamp, eview->isRealData(), edm::EventAuxiliary::PhysicsTrigger));
    ec->aux()->setProcessHistoryID(daqSource_->processHistoryID());
  } else if (tcds_pointer == nullptr) {
    std::stringstream ss;
    ss << "No TCDS FED in event with FEDHeader EID -: " << L1EventID;
    ec->setError(ss.str());
  } else {
    const FEDHeader fedHeader(tcds_pointer);
    tcds::Raw_v1 const* tcds = reinterpret_cast<tcds::Raw_v1 const*>(tcds_pointer + FEDHeader::length);
    edm::EventAuxiliary* aux = new edm::EventAuxiliary();  //allocate empty aux
    *aux = evf::evtn::makeEventAuxiliary(tcds,
                                         ec->run(),
                                         ls,
                                         eview->isRealData(),
                                         static_cast<edm::EventAuxiliary::ExperimentType>(fedHeader.triggerType()),
                                         daqSource_->processGUID(),
                                         !daqSource_->fileListLoopMode(),
                                         !tcdsInRange);
    ec->setAux(aux);
    ec->aux()->setProcessHistoryID(daqSource_->processHistoryID());
    ec->setRun(eview->run());
  }
}

void DataModeFRDPreUnpack::readEvent(edm::EventPrincipal& eventPrincipal) {
  if (ec_->error())
    throw cms::Exception("DAQSource::read") << ec_->errmsg();

  daqSource_->makeEventWrapper(eventPrincipal, *ec_->aux());

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<FEDRawDataCollection>(std::move(ec_->rawDataRef())));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->productDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
}

void DataModeFRDPreUnpack::unpackFile(RawInputFile* currentFile) {
  const uint64_t fileSize = currentFile->fileSize_;
  const unsigned rawHeaderSize = currentFile->rawHeaderSize_;

  //TODO: set threadError for issues in this function
  if (rawHeaderSize > 0) {
    assert(fileSize >= rawHeaderSize);
  }
  assert(fileSize >= headerSize());

  uint64_t bufpos = rawHeaderSize;

  while (bufpos < fileSize) {  //loop while there is file/events to read

    assert(bufpos + headerSize() <= fileSize);

    //fit to buffer model
    auto dataBlockAddr = (unsigned char*)currentFile->chunks_[0]->buf_ + bufpos;

    //first view for header only, check if it fits
    auto eview = std::make_unique<FRDEventMsgView>(dataBlockAddr);

    assert(bufpos + eview->size() <= fileSize);
    bufpos += eview->size();

    //create event wrapper
    //we will store this per each event queued to fwk
    UnpackedRawEventWrapper* ec = new UnpackedRawEventWrapper();

    assert(eview->version() >= 5);

    //crc check
    uint32_t crc = crc32c(0, (const unsigned char*)eview->payload(), eview->eventSize());
    if (crc != eview->crc32c()) {
      std::stringstream ss;
      ss << "Found a wrong crc32c checksum: expected 0x" << std::hex << eview->crc32c() << " but calculated 0x" << crc;
      ec->setChecksumError(ss.str());
      //unpackEvent(eview.get(), ec);
    } else
      unpackEvent(eview.get(), ec, currentFile->lumi_);
    currentFile->queue(ec);
  }
}

edm::Timestamp DataModeFRDPreUnpack::fillFEDRawDataCollection(edm::streamer::FRDEventMsgView* eview,
                                                              FEDRawDataCollection& rawData,
                                                              bool& tcdsInRange,
                                                              unsigned char*& tcds_pointer,
                                                              bool& err,
                                                              std::string& errmsg) {
  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  try {
    uint32_t eventSize = eview->eventSize();
    unsigned char* event = (unsigned char*)eview->payload();
    tcds_pointer = nullptr;
    tcdsInRange = false;
    uint16_t selectedTCDSFed = 0;
    unsigned int fedsInEvent = 0;
    while (eventSize > 0) {
      assert(eventSize >= FEDTrailer::length);
      eventSize -= FEDTrailer::length;
      const FEDTrailer fedTrailer(event + eventSize);
      const uint32_t fedSize = fedTrailer.fragmentLength() << 3;  //trailer length counts in 8 bytes
      assert(eventSize >= fedSize - FEDHeader::length);
      eventSize -= (fedSize - FEDHeader::length);
      const FEDHeader fedHeader(event + eventSize);
      const uint16_t fedId = fedHeader.sourceID();
      if (fedId > FEDNumbering::MAXFEDID)
        throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
          << "Out of range FED ID : " << fedId;
      else if (fedId >= MINTCDSuTCAFEDID_ && fedId <= MAXTCDSuTCAFEDID_) {
        if (!selectedTCDSFed) {
          selectedTCDSFed = fedId;
          tcds_pointer = event + eventSize;
          if (fedId >= FEDNumbering::MINTCDSuTCAFEDID && fedId <= FEDNumbering::MAXTCDSuTCAFEDID) {
            tcdsInRange = true;
          }
        } else
          throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
            << "Second TCDS FED ID " << fedId << " found. First ID: " << selectedTCDSFed;
      }
      //take event ID from GTPE FED
      FEDRawData& fedData = rawData.FEDData(fedId);
      fedData.resize(fedSize);
      memcpy(fedData.data(), event + eventSize, fedSize);

      fedsInEvent++;
      if (verifyFEDs_ || !expectedFedsInEvent_) {
        if (fedIdSet_.find(fedId) == fedIdSet_.end()) {
          if (expectedFedsInEvent_)
            throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection") << "FEDID " << fedId << " was not found in previous events";
          else
            fedIdSet_.insert(fedId);
        }
      }
    }
    assert(eventSize == 0);

    if (!fedsInEvent)
      throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
        << "Event " << event_->event() << " does not contain any FEDs";
    else if (!expectedFedsInEvent_) {
      expectedFedsInEvent_ = fedsInEvent;
      if (fedIdSet_.size() != fedsInEvent)
        throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
          << "First received event: " << event_->event() << " contains duplicate FEDs";
    }
    else if (fedsInEvent != expectedFedsInEvent_)
      throw cms::Exception("DataModeFRDPreUnpack:::fillFRDCollection")
        << "Event " << event_->event() << " does not contain same number of FEDs as previous: " << fedsInEvent << "/" << expectedFedsInEvent_;

  } catch (cms::Exception &e) {
    err = true;
    errmsg = e.what();
  }
  return tstamp;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeFRDPreUnpack::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(FEDRawDataCollection)), "FEDRawDataCollection", "FEDRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

void DataModeFRDPreUnpack::makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) {
  dataBlockAddr_ = addr;
  dataBlockMax_ = rawFile->currentChunkSize();
  eventCached_ = false;
  nextEventView(rawFile);
  eventCached_ = true;
}

bool DataModeFRDPreUnpack::nextEventView(RawInputFile* currentFile) {
  if (eventCached_)
    return true;
  event_ = std::make_unique<FRDEventMsgView>(dataBlockAddr_);
  if (event_->size() > dataBlockMax_) {
    throw cms::Exception("DAQSource::getNextEvent")
        << " event id:" << event_->event() << " lumi:" << event_->lumi() << " run:" << event_->run()
        << " of size:" << event_->size() << " bytes does not fit into a chunk of size:" << dataBlockMax_ << " bytes";
  }

  if (event_->version() < 5)
    throw cms::Exception("DAQSource::getNextEvent")
        << "Unsupported FRD version " << event_->version() << ". Minimum supported is v5.";

  currentFile->popQueue(ec_);
  return true;
}

bool DataModeFRDPreUnpack::checksumValid() { return !ec_->checksumError(); }

std::string DataModeFRDPreUnpack::getChecksumError() const { return ec_->errmsg(); }

/*
 * FRD Multi Source
 */

void DataModeFRDStriped::makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                                              std::vector<int> const& numSources,
                                              std::vector<int> const& sourceIDs,
                                              std::string const& sourceIdentifier,
                                              std::string const& runDir) {
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
      daqProvenanceHelpers_[0]->productDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
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
  unsigned int fedsInEvent = 0;
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
              << "Second TCDS FED ID " << fedId << " found in file at index" << selectedTCDSFileIndex
              << ". First ID: " << selectedTCDSFed << " found in file at index " << (uint64_t)index;
      }
      FEDRawData& fedData = rawData.FEDData(fedId);
      fedData.resize(fedSize);
      memcpy(fedData.data(), event + eventSize, fedSize);

      fedsInEvent++;
      if (verifyFEDs_ || !expectedFedsInEvent_) {
        if (fedIdSet_.find(fedId) == fedIdSet_.end()) {
          if (expectedFedsInEvent_)
            throw cms::Exception("DataModeFRDStriped:::fillFRDCollection")
              << "FEDID " << fedId << " from the file at index " << (uint64_t)index << " was not found in previous events";
          else
            fedIdSet_.insert(fedId);
        }
      }
    }
    assert(eventSize == 0);
  }

  if (!fedsInEvent)
    throw cms::Exception("DataModeFRDStriped:::fillFRDCollection")
        << "Event " << events_.at(0)->event() << " does not contain any FEDs";
  else if (!expectedFedsInEvent_) {
    expectedFedsInEvent_ = fedsInEvent;
    if (fedIdSet_.size() != fedsInEvent) {
      throw cms::Exception("DataModeFRDStriped:::fillFRDCollection")
        << "First received event: " << events_.at(0)->event() << " contains duplicate FEDs";
     }
  }
  else if (fedsInEvent != expectedFedsInEvent_)
    throw cms::Exception("DataModeFRDStriped:::fillFRDCollection")
        << "Event " << events_.at(0)->event() << " does not contain same number of FEDs as previous: "
        << fedsInEvent << "/" << expectedFedsInEvent_;

  return tstamp;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeFRDStriped::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(FEDRawDataCollection)), "FEDRawDataCollection", "FEDRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

bool DataModeFRDStriped::checksumValid() {
  bool status = true;
  for (size_t i = 0; i < events_.size(); i++) {
    uint32_t crc = 0;
    auto const& event = events_[i];
    crc = crc32c(crc, (const unsigned char*)event->payload(), event->eventSize());
    if (crc != event->crc32c()) {
      std::ostringstream ss;
      ss << "Found a wrong crc32c checksum at readout index " << i << ": expected 0x" << std::hex << event->crc32c()
         << " but calculated 0x" << crc << ". ";
      crcMsg_ += ss.str();
      status = false;
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

void DataModeFRDStriped::makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) {
  fileHeaderSize_ = rawFile->rawHeaderSize_;
  std::vector<uint64_t> const& fileSizes = rawFile->fileSizes_;
  numFiles_ = fileSizes.size();
  //add offset address for each file payload
  dataBlockAddrs_.clear();
  dataBlockAddrs_.push_back(addr);
  dataBlockMaxAddrs_.clear();
  dataBlockMaxAddrs_.push_back(addr + fileSizes[0] - fileHeaderSize_);
  auto fileAddr = addr;
  for (unsigned int i = 1; i < fileSizes.size(); i++) {
    fileAddr += fileSizes[i - 1];
    dataBlockAddrs_.push_back(fileAddr);
    dataBlockMaxAddrs_.push_back(fileAddr + fileSizes[i] - fileHeaderSize_);
  }

  dataBlockMax_ = rawFile->currentChunkSize();
  blockCompleted_ = false;
  //set event cached as we set initial address here
  bool result = makeEvents();
  assert(result);
  eventCached_ = true;
  setDataBlockInitialized(true);
}

bool DataModeFRDStriped::nextEventView(RawInputFile*) {
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
  int completed = 0;
  uint64_t testEvtId = 0;

  for (int i = 0; i < numFiles_; i++) {
    if (dataBlockAddrs_[i] >= dataBlockMaxAddrs_[i]) {
      //must be exact
      assert(dataBlockAddrs_[i] == dataBlockMaxAddrs_[i]);
      blockCompleted_ = true;
      completed++;
      continue;
    }
    if (blockCompleted_)
      continue;
    events_.emplace_back(std::make_unique<FRDEventMsgView>(dataBlockAddrs_[i]));

    if (testEvtId == 0)
      testEvtId = events_[i]->event();
    else if (testEvtId !=  events_[i]->event())
      throw cms::Exception("DAQSource::getNextEvent")
          << " event id mismatch:" << events_[i]->event() << " while in previously parsed RDEventMsgView (other file):" << testEvtId;

    if (dataBlockAddrs_[i] + events_[i]->size() > dataBlockMaxAddrs_[i])
      throw cms::Exception("DAQSource::getNextEvent")
          << " event id:" << events_[i]->event() << " lumi:" << events_[i]->lumi() << " run:" << events_[i]->run()
          << " of size:" << events_[i]->size() << " bytes does not fit into the buffer or has corrupted header";

    if (events_[i]->version() < 5)
      throw cms::Exception("DAQSource::getNextEvent")
          << "Unsupported FRD version " << events_[i]->version() << ". Minimum supported is v5.";
  }
  if (completed < numFiles_) {
    for (int i = 0; i < numFiles_; i++) {
      if (dataBlockAddrs_[i] == dataBlockMaxAddrs_[i]) {
        edm::LogError("dataModeFRDStriped::makeEvents") << "incomplete file block read from directory " << buPaths_[i];
        errorDetected_ = true;
      }
    }
  }
  return !blockCompleted_;
}
