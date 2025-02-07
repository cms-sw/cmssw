#include "EventFilter/Utilities/interface/DAQSource.h"
#include "EventFilter/Utilities/interface/DAQSourceModelsDTH.h"

#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <bitset>

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


using namespace evf;

void DataModeDTH::readEvent(edm::EventPrincipal& eventPrincipal) {
  std::unique_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  edm::Timestamp tstamp = fillFEDRawDataCollection(*rawData);

  edm::EventID eventID = edm::EventID(daqSource_->eventRunNumber(), daqSource_->currentLumiSection(), nextEventID_);
  edm::EventAuxiliary aux(
      eventID, daqSource_->processGUID(), tstamp, isRealData(), edm::EventAuxiliary::PhysicsTrigger);
  aux.setProcessHistoryID(daqSource_->processHistoryID());
  daqSource_->makeEventWrapper(eventPrincipal, aux);

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<FEDRawDataCollection>(std::move(rawData)));
  eventPrincipal.put(
      daqProvenanceHelpers_[0]->productDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
  eventCached_ = false;
}

edm::Timestamp DataModeDTH::fillFEDRawDataCollection(FEDRawDataCollection& rawData) {
  //generate timestamp for this event until parsing of TCDS2 data is available
  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  for (size_t i = 0; i < eventFragments_.size(); i++) {
    auto fragTrailer = eventFragments_[i];
    uint8_t* payload = (uint8_t*)fragTrailer->payload();
    auto fragSize = fragTrailer->payloadSizeBytes();
    /*
    //Slink header and trailer
    assert(fragSize >= (FEDTrailer::length + FEDHeader::length));
    const FEDHeader fedHeader(payload);
    const FEDTrailer fedTrailer((uint8_t*)fragTrailer - FEDTrailer::length);
    const uint32_t fedSize = fedTrailer.fragmentLength() << 3;  //trailer length counts in 8 bytes
    const uint16_t fedId = fedHeader.sourceID();
*/

    //SLinkRocket header and trailer
    if (fragSize < sizeof(SLinkRocketTrailer_v3) + sizeof(SLinkRocketHeader_v3))
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid fragment size: " << fragSize;

    const SLinkRocketHeader_v3* fedHeader = (const SLinkRocketHeader_v3*)payload;
    const SLinkRocketTrailer_v3* fedTrailer =
        (const SLinkRocketTrailer_v3*)((uint8_t*)fragTrailer - sizeof(SLinkRocketTrailer_v3));

    //check SLR trailer first as it comes just before fragmen trailer
    if (!fedTrailer->verifyMarker())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid SLinkRocket trailer";
    if (!fedHeader->verifyMarker())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid SLinkRocket header";

    const uint32_t fedSize = fedTrailer->eventLenBytes();
    const uint16_t fedId = fedHeader->sourceID();

    /*
     *  @SM: CRC16 in trailer was not checked up to Run3, no need to do production check
     *  if we already check orbit CRC32. If CRC16 check is to be added,
     *  in phase1 crc16 was calculated on sequential 64-byte little-endian words
     *  (see FWCore/Utilities/interface/CRC16.h).
     *  See also optimized pclmulqdq implementation in XDAQ.
     *  Note: check if for phase-2 crc16 is still based on 8-byte words
    */
    //const uint32_t crc16 = fedTrailer->crc();

    if (fedSize != fragSize)
      throw cms::Exception("DAQSource::DAQSourceModelsDTH")
          << "Fragment size mismatch. From DTHTrailer: " << fragSize << " and from SLinkRocket trailer: " << fedSize;
    FEDRawData& fedData = rawData.FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), payload, fedSize);  //copy with header and trailer
  }
  return tstamp;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeDTH::makeDaqProvenanceHelpers() {
  //use also FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(FEDRawDataCollection)), "FEDRawDataCollection", "FEDRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}

void DataModeDTH::makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) {

  //addr points to beginning of the main file orbit block

  //get file array info
  auto numFiles = rawFile->fileSizes_.size();

  //initialize address tracking for files in the buffer: add primary file

  auto buf = rawFile->chunks_[0]->buf_;

  //all fragment addresses could be merged into a pair or tuple and reserve size
  addrsEnd_.clear();
  addrsStart_.clear();
  constexpr size_t hsize = sizeof(evf::DTHOrbitHeader_v1);
  unsigned char* nextEnd = nullptr;
  firstOrbitHeader_ = nullptr;

  for (unsigned i = 0; i < numFiles; i++) {
    bool ohThisFile = false;
    //intial orbit header was advanced over by source (first file only)
    auto nextAddr = buf + rawFile->bufferOffsets_[i];
    auto startAddr = nextAddr;//save start position of the orbit
    auto maxAddr = buf + rawFile->bufferEnds_[i];//end of stripe / file


    LogDebug("DataModeDTH") << "make data block view for file " << i << " at offsets: " << rawFile->bufferOffsets_[i] << " to: " << rawFile->bufferEnds_[i]
                            << " blockAddr: 0x" << std::hex << (uint64_t)nextAddr << " chunkOffset: 0x"
                            << std::hex << (uint64_t)(nextAddr - buf);

    checksumValid_ = true;
    if (!checksumError_.empty())
      checksumError_ = std::string();

    while (nextAddr < maxAddr) {
      //ensure header fits
      assert(nextAddr + hsize < maxAddr);

      auto orbitHeader = (evf::DTHOrbitHeader_v1*)(nextAddr);

      if (!orbitHeader->verifyMarker())
        throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid DTH orbit marker";
      if (i == 0) {
        //get initial orbit number and find all subsequent orbits with the same nr in this file
        ohThisFile = true;
        if (!firstOrbitHeader_)
          firstOrbitHeader_ = orbitHeader;
        else {
          assert(orbitHeader->runNumber() == firstOrbitHeader_->runNumber());
          if (orbitHeader->orbitNumber() != firstOrbitHeader_->orbitNumber()) {
            break;
          }
          assert(orbitHeader->eventCount() == firstOrbitHeader_->eventCount());
        }
      } else {
        //check that orbit headers in all files are consistent with first
        assert(firstOrbitHeader_);
        assert(orbitHeader->runNumber() == firstOrbitHeader_->runNumber());

        if (!ohThisFile) {
          //each file must contain at least one orbit nr of the first file
          assert(orbitHeader->orbitNumber() == firstOrbitHeader_->orbitNumber());
          ohThisFile = true;
        } else
          if (orbitHeader->orbitNumber() != firstOrbitHeader_->orbitNumber())
            break;
        assert(orbitHeader->eventCount() == firstOrbitHeader_->eventCount());
      }

      if (verifyChecksum_) {
        auto crc = crc32c(0U, (const uint8_t*)orbitHeader->payload(), orbitHeader->payloadSizeBytes());
        if (crc != orbitHeader->crc()) {
          checksumValid_ = false;
          if (!checksumError_.empty())
            checksumError_ += "\n";
          checksumError_ +=
            fmt::format("Found a wrong crc32c checksum in orbit header v{} run: {} orbit: {} sourceId: {} wcount: {} events: {} flags: {}. Expected {:x} but calculated {:x}",
                        orbitHeader->version(),
                        orbitHeader->runNumber(),
                        orbitHeader->orbitNumber(),
                        orbitHeader->sourceID(),
                        orbitHeader->packed_word_count(),
                        orbitHeader->eventCount(),
                        orbitHeader->flags(),
                        orbitHeader->crc(),
                        crc);
        }
      }
      LogDebug("DataModeDTH") << "DTH orbit block version:"  << orbitHeader->version()
                              << " sourceID:" << orbitHeader->sourceID()
                              << " run:" << orbitHeader->runNumber()
                              << " orbitNr:" << orbitHeader->orbitNumber()
                              << " evtFragments:" << orbitHeader->eventCount()
                              << " crc32c:" << orbitHeader->crc()
                              << " flagMask:" << std::hex << orbitHeader->flags();
      //push current orbit to the list of orbits
      auto srcOrbitSize = orbitHeader->totalSize();
      addrsStart_.push_back(nextAddr + hsize);
      addrsEnd_.push_back(nextAddr + srcOrbitSize);

      //update position in the buffer
      nextAddr += srcOrbitSize;
      nextEnd = nextAddr;
      assert(nextEnd <= maxAddr);  //boundary check
    }

    //require orbit header in each file
    assert(ohThisFile);

    //report first file block size
    if (i == 0) {
      //assert(nextEnd > nextAddr);
      dataBlockSize_ = nextEnd - startAddr;
    }

    //advance buffer position to next orbit
    //rawFile->bufferOffsets_[i] += nextAddr - startAddr;
    rawFile->advanceBuffer(nextEnd - startAddr, i);
  }
  //update next pointer
  //firstOrbitHeader_ = nextOrbitHeader;

  eventCached_ = false;
  blockCompleted_ = false;
  nextEventView(rawFile);
  eventCached_ = true;
}

bool DataModeDTH::nextEventView(RawInputFile*) {
  if (eventCached_)
    return true;

  blockCompleted_ = false;

  bool blockCompletedAll = !addrsEnd_.empty() ? true : false;
  bool blockCompletedAny = false;
  eventFragments_.clear();
  size_t last_eID = 0;

  for (size_t i = 0; i < addrsEnd_.size(); i++) {

    if (addrsEnd_[i] == addrsStart_[i]) {
      blockCompletedAny = true;
      continue;
    } else {
      assert(addrsEnd_[i] > addrsStart_[i]);
      blockCompletedAll = false;
      if (blockCompletedAny) continue;
    }

    evf::DTHFragmentTrailer_v1* trailer =
        (evf::DTHFragmentTrailer_v1*)(addrsEnd_[i] - sizeof(evf::DTHFragmentTrailer_v1));

    if (!trailer->verifyMarker())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid DTH trailer marker";

    assert((uint8_t*)trailer >= addrsStart_[i]);

    uint64_t eID = trailer->eventID();
    eventFragments_.push_back(trailer);
    auto payload_size = trailer->payloadSizeBytes();
    if (payload_size > evf::SLR_MAX_EVENT_LEN)  //max possible by by SlinkRocket (1 MB)
      throw cms::Exception("DAQSource::DAQSourceModelsDTH")
          << "DTHFragment size " << payload_size << " larger than the SLinkRocket limit of " << evf::SLR_MAX_EVENT_LEN;

    if (i == 0) {
      nextEventID_ = eID;
      last_eID = eID;
    } else if (last_eID != nextEventID_)
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Inconsistent event number between fragments";

    if (trailer->flags())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH")
          << "Detected error condition in DTH trailer of event " << trailer->eventID()
          << " flags: " << std::bitset<16>(trailer->flags());

    LogDebug("DataModeDTH") << "DTH fragment trailer in block " << i << " eventID: " << trailer->eventID()
                            << " payloadSizeBytes: " <<  trailer->payloadSizeBytes()
                            << " crc: " << trailer->crc()
                            << " flagMask: " << std::hex << trailer->flags();

    //update address array
    addrsEnd_[i] -= sizeof(evf::DTHFragmentTrailer_v1) + payload_size;

    /* --> moved to beginning
    if (addrsEnd_[i] == addrsStart_[i]) {
      blockCompletedAny = true;
    } else {
      assert(addrsEnd_[i] > addrsStart_[i]);
      blockCompletedAll = false;
    }*/
  }
  if (blockCompletedAny != blockCompletedAll)
    throw cms::Exception("DAQSource::DAQSourceModelsDTH")
        << "Some orbit sources have inconsistent number of event fragments.";

  if (blockCompletedAll) {
    blockCompleted_ = blockCompletedAll;
    firstOrbitHeader_ = nullptr;
    return false;
  }
  return true;
}

//striped mode functions
void DataModeDTH::makeDirectoryEntries(std::vector<std::string> const& baseDirs,
                                              std::vector<int> const& numSources,
                                              std::vector<int> const& sourceIDs,
                                              std::string const& sourceIdentifier,
                                              std::string const& runDir) {
  std::filesystem::path runDirP(runDir);
  for (auto& baseDir : baseDirs) {
    std::filesystem::path baseDirP(baseDir);
    buPaths_.emplace_back(baseDirP / runDirP);
  }
  if (!sourceIdentifier.empty()) {
    sid_pattern_ = std::regex("_" + sourceIdentifier + R"(\d+_)");

    for (auto sourceID : sourceIDs) {
      std::stringstream ss;
      ss << "_" + sourceIdentifier << std::setfill('0') << std::setw(4) <<  std::to_string(sourceID);
      buSourceStrings_.push_back(ss.str());
    }

    if (baseDirs.size() != numSources.size())
      throw cms::Exception("DataModeDTH::makeDirectoryEntries") << "Number of defined directories not compatible with numSources list length";

    unsigned int sum = 0;
    for (auto numSource: numSources) {
      buNumSources_.push_back(numSource);
      sum += numSource;
    }

    if (sum != sourceIDs.size())
      throw cms::Exception("DataModeDTH::makeDirectoryEntries") << "Number of defined sources not consistent with the list of sourceIDs";
  }
}

std::pair<bool, std::vector<std::string>> DataModeDTH::defineAdditionalFiles(std::string const& primaryName,
                                                                                    bool fileListMode) const {
  //non-striped mode
  if (!buPaths_.size())
    return std::make_pair(true, std::vector<std::string>());

  std::vector<std::string> additionalFiles;

  //not touching primary file name as found by input mechanism. Format assumes source is last parameter in the filename
  auto extpos = primaryName.rfind(".");
  auto indexpos = primaryName.find("_index");
  assert(indexpos != std::string::npos);
  auto cutoff = primaryName.find("_", indexpos + 1); //search after index
  if (cutoff == std::string::npos) cutoff = extpos; //no source
  auto slashpos = primaryName.rfind("/", indexpos);
  auto startoff = slashpos == std::string::npos ? 0 : slashpos + 1;//determine if directory path is returned

  std::string primStem = primaryName.substr(startoff, cutoff - startoff);
  std::string ext = primaryName.substr(extpos);

  if (!buSourceStrings_.empty()) {
    int counter = 0;
    for (size_t i = 0; i < buPaths_.size(); i++) {
      for (size_t j = 0; j < (size_t) buNumSources_[i]; j++) {
        std::string replacement = buPaths_[i].generic_string() + ("/" + primStem + buSourceStrings_[counter] + ext);
        counter++;
        if (i==0 && j==0) continue;
        additionalFiles.push_back(replacement);
      }
    }
  }
  else {
    auto fullpath = std::filesystem::path(primStem + ext);
    auto fullname = fullpath.filename();
    for (size_t i = 1; i < buPaths_.size(); i++) {
      std::filesystem::path newPath = buPaths_[i] / fullname;
      additionalFiles.push_back(newPath.generic_string());
    }
  }
  return std::make_pair(true, additionalFiles);
}

//count events in raw file (in absence of file header) and return open file descriptor
int DataModeDTH::eventCounterCallback(std::string const& name, int& rawFd, int64_t& totalSize, uint32_t sLS, bool& found) const {

  uint32_t orbit_count = 0;
  uint32_t event_count = 0;

  auto fileClose = [&]() -> int {
    if (rawFd != -1) {
      close(rawFd);
      rawFd = -1;
    }
    return -1;
  };

  if ((rawFd = ::open(name.c_str(), O_RDONLY)) < 0) {
    assert(rawFd == -1);
    found = false;
    edm::LogError("EvFDaqDirector")
      << "parseFRDFileHeader - failed to open input file -: " << name << " : " << strerror(errno);
    return -1;
  }
  found = true;

  struct stat st;
  if (fstat(rawFd, &st) == -1) {
    edm::LogError("DAQSourceModelsDTH") << "rawCounter - unable to stat " << name << " : " << strerror(errno);
    return fileClose();
  }

  int firstSourceId = -1;
  unsigned char hdr[sizeof(DTHOrbitHeader_v1)];

  totalSize = 0;
  while (true) {
    auto buf_sz = sizeof(DTHOrbitHeader_v1);
    ssize_t sz_read = ::read(rawFd, hdr, buf_sz);
    if (sz_read < 0) {
      edm::LogError("DAQSourceModelsDTH") << "unable to read header of " << name << " : " << strerror(errno);
      return fileClose();
    }
    if ((size_t)sz_read < buf_sz) {
      edm::LogError("EvFDaqDirector") << "DTH header larger than the the remaining file size: " << name;
      return fileClose();
    }
    totalSize += sz_read;

    DTHOrbitHeader_v1* oh = (DTHOrbitHeader_v1*)hdr;
    LogDebug("EvFDaqDirector") << "orbit check: orbit:" << oh->orbitNumber() << " source:" << oh->sourceID()
                               << " eventCount:" << oh->eventCount();

    if (!oh->verifyMarker()) {
      edm::LogError("EvFDaqDirector") << "Invalid DTH header encountered";
      return fileClose();
    }
    if (!oh->verifyMarker() || oh->version() != 1) {
      edm::LogError("EvFDaqDirector") << "Unexpected DTH header version " << oh->version();
      return fileClose();
    }

    if (firstSourceId == -1)
      firstSourceId = oh->sourceID();
    if (oh->sourceID() == (unsigned)firstSourceId) {
      orbit_count++;
      event_count += oh->eventCount();
    }
    //else skip counting events from all source IDs in the file (assume they are same)
    auto payloadSize = oh->totalSize() - sizeof(DTHOrbitHeader_v1);
    totalSize += payloadSize;
    if (totalSize > st.st_size) {
      edm::LogError("EvFDaqDirector") << "DTH header can not be beyond file size: " << name;
      return fileClose();
    }
    //seek to the next orbit header
    auto new_offset = lseek(rawFd, payloadSize, SEEK_CUR);

    //if (new_offset < st.st_size) {
    if (new_offset < totalSize) {
      edm::LogError("EvFDaqDirector") << "Unexpected end of file: " << name;
      return fileClose();
    }

    if (new_offset == st.st_size) {
      lseek(rawFd, 0, SEEK_SET);
      break;
    }
  }
  return event_count;
}
