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
      daqProvenanceHelpers_[0]->branchDescription(), std::move(edp), daqProvenanceHelpers_[0]->dummyProvenance());
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

  for (size_t i=0; i<eventFragments_.size(); i++) {

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

    const SLinkRocketHeader_v3* fedHeader = (const SLinkRocketHeader_v3*) payload;
    const SLinkRocketTrailer_v3* fedTrailer = (const SLinkRocketTrailer_v3*) ((uint8_t*)fragTrailer - sizeof(SLinkRocketTrailer_v3));

    //check SLR trailer first as it comes just before fragmen trailer
    if (!fedTrailer->verifyMarker())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid SLinkRocket trailer";
    if (!fedHeader->verifyMarker())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid SLinkRocket header";

    const uint32_t fedSize = fedTrailer->eventLenBytes();
    const uint16_t fedId = fedHeader->sourceID();

    /*
     *  @SM: CRC16 in trailer was not checked up to Run3, no need to do production check.
     *  if we already check orbit CRC32.If CRC16 check is to be added,
     *  in phase1 crc16 was calculated on sequential 64-byte little-endian words
     *  (see FWCore/Utilities/interface/CRC16.h).
     *  See also optimized pclmulqdq implementation in XDAQ.
     *  Note: check if for phase-2 crc16 is still based on 8-byte words
    */
    //const uint32_t crc16 = fedTrailer->crc();

    if (fedSize != fragSize)
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Fragment size mismatch. From DTHTrailer: " << fragSize << " and from SLinkRocket trailer: " << fedSize;
    FEDRawData& fedData = rawData.FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), payload, fedSize); //copy with header and trailer
  }
  return tstamp;
}

std::vector<std::shared_ptr<const edm::DaqProvenanceHelper>>& DataModeDTH::makeDaqProvenanceHelpers() {
  //set FRD data collection
  daqProvenanceHelpers_.clear();
  daqProvenanceHelpers_.emplace_back(std::make_shared<const edm::DaqProvenanceHelper>(
      edm::TypeID(typeid(FEDRawDataCollection)), "FEDRawDataCollection", "FEDRawDataCollection", "DAQSource"));
  return daqProvenanceHelpers_;
}


void DataModeDTH::makeDataBlockView(unsigned char* addr, RawInputFile* rawFile) {

    //TODO: optimize by merging into a pair or tuple and reserve size
    addrsEnd_.clear();
    addrsStart_.clear();
    constexpr size_t hsize = sizeof(evf::DTHOrbitHeader_v1);

    LogDebug("DataModeDTH::makeDataBlockView") << "blockAddr: 0x" << std::hex << (uint64_t) addr << " chunkOffset: 0x" << std::hex << (uint64_t)(addr - rawFile->chunks_[0]->buf_);

    //intial orbit header was advanced over by source
    size_t maxAllowedSize = rawFile->fileSizeLeft() + headerSize();
    auto nextAddr = addr;
    checksumValid_ = true;
    if (checksumError_.size())
      checksumError_ = std::string();

    firstOrbitHeader_ = nullptr;
    while (nextAddr < addr + maxAllowedSize) {

      //ensure header fits
      assert(nextAddr + hsize < addr + maxAllowedSize);

      auto orbitHeader = (evf::DTHOrbitHeader_v1*)(nextAddr);
      if (!orbitHeader->verifyMarker())
        throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid DTH orbit marker";
      if (!firstOrbitHeader_) {
        firstOrbitHeader_ = orbitHeader;
      }
      else {
        assert(orbitHeader->runNumber() == firstOrbitHeader_->runNumber());
        if (orbitHeader->orbitNumber() != firstOrbitHeader_->orbitNumber()) {
          firstOrbitHeader_ = orbitHeader;
          //next orbit ID reached, do not include this orbit in this block
          break;
        }
      }

      auto srcOrbitSize = orbitHeader->totalSize();
      auto nextEnd = nextAddr + srcOrbitSize;
      assert(nextEnd <= addr + maxAllowedSize);//boundary check

      if (verifyChecksum_) {
        auto crc = crc32c(0U, (const uint8_t*)orbitHeader->payload(), orbitHeader->payloadSizeBytes());
        if (crc != orbitHeader->crc()) {
          checksumValid_ = false;
          if (checksumError_.size()) checksumError_ += "\n";
          checksumError_ += fmt::format("Found a wrong crc32c checksum in orbit: {} sourceID: {}. Expected {:x} but calculated {:x}",
                                        orbitHeader->orbitNumber(), orbitHeader->sourceID(), orbitHeader->crc(), crc);
        }
      }

      addrsStart_.push_back(nextAddr + hsize);
      addrsEnd_.push_back(nextAddr + srcOrbitSize);
      nextAddr +=  srcOrbitSize;
    }
    dataBlockSize_ = nextAddr - addr;

    eventCached_ = false;
    nextEventView(rawFile);
    eventCached_ = true;
  }


bool DataModeDTH::nextEventView(RawInputFile*) {
  blockCompleted_ = false;
  if (eventCached_)
    return true;

  bool blockCompletedAll = addrsEnd_.size() ? true: false;
  bool blockCompletedAny = false;
  eventFragments_.clear();
  size_t last_eID = 0;

  for (size_t i=0; i<addrsEnd_.size(); i++) {
    evf::DTHFragmentTrailer_v1* trailer = (evf::DTHFragmentTrailer_v1*)(addrsEnd_[i] -  sizeof(evf::DTHFragmentTrailer_v1));

    if (!trailer->verifyMarker())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Invalid DTH trailer marker";

    assert((uint8_t*)trailer >= addrsStart_[i]);

    uint64_t eID = trailer->eventID();
    eventFragments_.push_back(trailer);
    auto payload_size = trailer->payloadSizeBytes();
    if(payload_size > evf::SLR_MAX_EVENT_LEN) //max possible by by SlinkRocket (1 MB)
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "DTHFragment size " << payload_size
        << " larger than the SLinkRocket limit of " << evf::SLR_MAX_EVENT_LEN;

    if (i==0) {
      nextEventID_ = eID;
      last_eID = eID;
    }
    else
      if(last_eID != nextEventID_)
        throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Inconsistent event number between fragments";

    //update address array
    addrsEnd_[i] -= sizeof(evf::DTHFragmentTrailer_v1) + payload_size;

    if (trailer->flags())
      throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Detected error condition in DTH trailer of event "
        << trailer->eventID() << " flags: "<< std::bitset<16>(trailer->flags());

    if (addrsEnd_[i] == addrsStart_[i]) {
      blockCompletedAny = true;
    }
    else {
      assert(addrsEnd_[i] > addrsStart_[i]);
      blockCompletedAll = false;
    }
  }
  if (blockCompletedAny != blockCompletedAll)
    throw cms::Exception("DAQSource::DAQSourceModelsDTH") << "Some orbit sources have inconsistent number of event fragments.";

  if (blockCompletedAll) {
    blockCompleted_ = blockCompletedAll;
    firstOrbitHeader_ = nullptr;
    return false;
  }
  return true;
}
