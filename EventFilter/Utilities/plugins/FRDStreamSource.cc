#include <iostream>
#include <memory>
#include <zlib.h>

#include "IOPool/Streamer/interface/FRDEventMessage.h"
#include "IOPool/Streamer/interface/FRDFileHeader.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "DataFormats/TCDS/interface/TCDSRaw.h"

#include "EventFilter/Utilities/interface/GlobalEventNumber.h"

#include "EventFilter/Utilities/plugins/FRDStreamSource.h"
#include "EventFilter/Utilities/interface/crc32c.h"

FRDStreamSource::FRDStreamSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
    : ProducerSourceFromFiles(pset, desc, true),
      verifyAdler32_(pset.getUntrackedParameter<bool>("verifyAdler32", true)),
      verifyChecksum_(pset.getUntrackedParameter<bool>("verifyChecksum", true)),
      useL1EventID_(pset.getUntrackedParameter<bool>("useL1EventID", false)) {
  fileNames_ = fileNames(0), itFileName_ = fileNames_.begin();
  endFileName_ = fileNames_.end();
  openFile(*itFileName_);
  produces<FEDRawDataCollection>();
}

bool FRDStreamSource::setRunAndEventInfo(edm::EventID& id,
                                         edm::TimeValue_t& theTime,
                                         edm::EventAuxiliary::ExperimentType& eType) {
  if (fin_.peek() == EOF) {
    if (++itFileName_ == endFileName_) {
      fin_.close();
      return false;
    }
    if (!openFile(*itFileName_)) {
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo") << "could not open file " << *itFileName_;
    }
  }
  //look for FRD header at beginning of the file and skip it
  if (fin_.tellg() == 0) {
    constexpr size_t buf_sz = sizeof(FRDFileHeaderIdentifier);
    char hdr[sizeof(FRDFileHeader_v2)];
    fin_.read(hdr, buf_sz);

    if (fin_.gcount() == 0)
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo")
          << "Unable to read file or empty file" << *itFileName_;
    else if (fin_.gcount() < (ssize_t)buf_sz) {
      //no header, very small file, go to event parsing
      fin_.seekg(0);
    } else {
      FRDFileHeaderIdentifier* fileId = (FRDFileHeaderIdentifier*)hdr;
      uint16_t frd_version = getFRDFileHeaderVersion(fileId->id_, fileId->version_);

      if (frd_version == 1) {
        constexpr size_t buf_sz_cont = sizeof(FRDFileHeaderContent_v1);
        fin_.read(hdr, buf_sz_cont);
        FRDFileHeaderContent_v1* fhContent = (FRDFileHeaderContent_v1*)hdr;
        if (fin_.gcount() != buf_sz_cont || fhContent->headerSize_ != sizeof(FRDFileHeader_v1))
          throw cms::Exception("FRDStreamSource::setRunAndEventInfo")
              << "Invalid FRD file header (size mismatch) in file " << *itFileName_;
      } else if (frd_version == 2) {
        constexpr size_t buf_sz_cont = sizeof(FRDFileHeaderContent_v2);
        fin_.read(hdr, buf_sz_cont);
        FRDFileHeaderContent_v2* fhContent = (FRDFileHeaderContent_v2*)hdr;
        if (fin_.gcount() != buf_sz_cont || fhContent->headerSize_ != sizeof(FRDFileHeader_v2))
          throw cms::Exception("FRDStreamSource::setRunAndEventInfo")
              << "Invalid FRD file header (size mismatch) in file " << *itFileName_;
      } else if (frd_version > 2) {
        throw cms::Exception("FRDStreamSource::setRunAndEventInfo") << "Unknown header version " << frd_version;
      } else {
        //no header
        fin_.seekg(0, fin_.beg);
      }
    }
  }

  if (detectedFRDversion_ == 0) {
    fin_.read((char*)&detectedFRDversion_, sizeof(uint16_t));
    fin_.read((char*)&flags_, sizeof(uint16_t));
    assert(detectedFRDversion_ > 0 && detectedFRDversion_ <= FRDHeaderMaxVersion);
    if (buffer_.size() < FRDHeaderVersionSize[detectedFRDversion_])
      buffer_.resize(FRDHeaderVersionSize[detectedFRDversion_]);
    *((uint32_t*)(&buffer_[0])) = detectedFRDversion_;
    fin_.read(&buffer_[0] + sizeof(uint32_t), FRDHeaderVersionSize[detectedFRDversion_] - sizeof(uint32_t));
    assert(fin_.gcount() == FRDHeaderVersionSize[detectedFRDversion_] - (unsigned int)(sizeof(uint32_t)));
  } else {
    if (buffer_.size() < FRDHeaderVersionSize[detectedFRDversion_])
      buffer_.resize(FRDHeaderVersionSize[detectedFRDversion_]);
    fin_.read(&buffer_[0], FRDHeaderVersionSize[detectedFRDversion_]);
    assert(fin_.gcount() == FRDHeaderVersionSize[detectedFRDversion_]);
  }

  std::unique_ptr<FRDEventMsgView> frdEventMsg(new FRDEventMsgView(&buffer_[0]));
  if (useL1EventID_)
    id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), frdEventMsg->event());

  const uint32_t totalSize = frdEventMsg->size();
  if (totalSize > buffer_.size()) {
    buffer_.resize(totalSize);
  }
  if (totalSize > FRDHeaderVersionSize[detectedFRDversion_]) {
    fin_.read(&buffer_[0] + FRDHeaderVersionSize[detectedFRDversion_],
              totalSize - FRDHeaderVersionSize[detectedFRDversion_]);
    if (fin_.gcount() != totalSize - FRDHeaderVersionSize[detectedFRDversion_]) {
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo") << "premature end of file " << *itFileName_;
    }
    frdEventMsg = std::make_unique<FRDEventMsgView>(&buffer_[0]);
  }

  if (verifyChecksum_ && frdEventMsg->version() >= 5) {
    uint32_t crc = 0;
    crc = crc32c(crc, (const unsigned char*)frdEventMsg->payload(), frdEventMsg->eventSize());
    if (crc != frdEventMsg->crc32c()) {
      throw cms::Exception("FRDStreamSource::getNextEvent") << "Found a wrong crc32c checksum: expected 0x" << std::hex
                                                            << frdEventMsg->crc32c() << " but calculated 0x" << crc;
    }
  } else if (verifyAdler32_ && frdEventMsg->version() >= 3) {
    uint32_t adler = adler32(0L, Z_NULL, 0);
    adler = adler32(adler, (Bytef*)frdEventMsg->payload(), frdEventMsg->eventSize());

    if (adler != frdEventMsg->adler32()) {
      throw cms::Exception("FRDStreamSource::setRunAndEventInfo")
          << "Found a wrong Adler32 checksum: expected 0x" << std::hex << frdEventMsg->adler32() << " but calculated 0x"
          << adler;
    }
  }

  rawData_ = std::make_unique<FEDRawDataCollection>();

  uint32_t eventSize = frdEventMsg->eventSize();
  unsigned char* event = (unsigned char*)frdEventMsg->payload();
  bool foundTCDSFED = false;
  bool foundGTPFED = false;

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
      throw cms::Exception("FedRawDataInputSource::fillFEDRawDataCollection") << "Out of range FED ID : " << fedId;
    }
    if (fedId == FEDNumbering::MINTCDSuTCAFEDID) {
      foundTCDSFED = true;
      tcds::Raw_v1 const* tcds = reinterpret_cast<tcds::Raw_v1 const*>(event + eventSize + FEDHeader::length);
      id = edm::EventID(frdEventMsg->run(), tcds->header.lumiSection, tcds->header.eventNumber);
      eType = static_cast<edm::EventAuxiliary::ExperimentType>(fedHeader.triggerType());
      theTime = static_cast<edm::TimeValue_t>(((uint64_t)tcds->bst.gpstimehigh << 32) | tcds->bst.gpstimelow);
    }

    if (fedId == FEDNumbering::MINTriggerGTPFEDID && !foundTCDSFED) {
      foundGTPFED = true;
      const bool GTPEvmBoardSense = evf::evtn::evm_board_sense(event + eventSize, fedSize);
      if (!useL1EventID_) {
        if (GTPEvmBoardSense)
          id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), evf::evtn::get(event + eventSize, true));
        else
          id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), evf::evtn::get(event + eventSize, false));
      }
      //evf::evtn::evm_board_setformat(fedSize);
      const uint64_t gpsl = evf::evtn::getgpslow(event + eventSize);
      const uint64_t gpsh = evf::evtn::getgpshigh(event + eventSize);
      theTime = static_cast<edm::TimeValue_t>((gpsh << 32) + gpsl);
    }

    //take event ID from GTPE FED
    if (fedId == FEDNumbering::MINTriggerEGTPFEDID && !foundGTPFED && !foundTCDSFED && !useL1EventID_) {
      if (evf::evtn::gtpe_board_sense(event + eventSize)) {
        id = edm::EventID(frdEventMsg->run(), frdEventMsg->lumi(), evf::evtn::gtpe_get(event + eventSize));
      }
    }
    FEDRawData& fedData = rawData_->FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), event + eventSize, fedSize);
  }
  assert(eventSize == 0);

  return true;
}

void FRDStreamSource::produce(edm::Event& e) { e.put(std::move(rawData_)); }

bool FRDStreamSource::openFile(const std::string& fileName) {
  std::cout << " open file.. " << fileName << std::endl;
  fin_.close();
  fin_.clear();
  size_t pos = fileName.find(':');
  if (pos != std::string::npos) {
    std::string prefix = fileName.substr(0, pos);
    if (prefix != "file")
      return false;
    pos++;
  } else
    pos = 0;

  fin_.open(fileName.substr(pos).c_str(), std::ios::in | std::ios::binary);
  return fin_.is_open();
}

//////////////////////////////////////////
// define this class as an input source //
//////////////////////////////////////////
DEFINE_FWK_INPUT_SOURCE(FRDStreamSource);

// Keep old naming from DAQ1
using ErrorStreamSource = FRDStreamSource;
DEFINE_FWK_INPUT_SOURCE(ErrorStreamSource);
