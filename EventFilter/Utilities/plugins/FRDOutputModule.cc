#include "EventFilter/Utilities/plugins/FRDOutputModule.h"

// system headers
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// C++ headers
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"
#include "IOPool/Streamer/interface/FRDFileHeader.h"
#include "EventFilter/Utilities/interface/crc32c.h"

FRDOutputModule::FRDOutputModule(edm::ParameterSet const& ps)
    : edm::one::OutputModuleBase::OutputModuleBase(ps),
      edm::one::OutputModule<edm::one::WatchLuminosityBlocks>(ps),
      token_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("source"))),
      frdVersion_(ps.getUntrackedParameter<unsigned int>("frdVersion")),
      frdFileVersion_(ps.getUntrackedParameter<unsigned int>("frdFileVersion")),
      filePrefix_(ps.getUntrackedParameter<std::string>("filePrefix")),
      fileName_(ps.getUntrackedParameter<std::string>("fileName")) {}

FRDOutputModule::~FRDOutputModule() {}

void FRDOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag("rawDataCollector"));
  desc.addUntracked<unsigned int>("frdFileVersion", 1), desc.addUntracked<unsigned int>("frdVersion", 6);
  desc.addUntracked<std::string>("filePrefix", "");
  desc.addUntracked<std::string>("fileName", "");
  descriptions.addWithDefaultLabel(desc);
}

void FRDOutputModule::write(edm::EventForOutput const& e) {
  // serialize the FEDRawDataCollection into the format that we expect for
  // FRDEventMsgView objects (may be better ways to do this)
  edm::Handle<FEDRawDataCollection> fedBuffers;
  e.getByToken(token_, fedBuffers);

  // determine the expected size of the FRDEvent IN BYTES !!!!!
  assert(frdVersion_ <= FRDHeaderMaxVersion);
  int headerSize = FRDHeaderVersionSize[frdVersion_];
  int expectedSize = headerSize;
  int nFeds = frdVersion_ < 3 ? 1024 : FEDNumbering::lastFEDId() + 1;

  for (int idx = 0; idx < nFeds; ++idx) {
    FEDRawData singleFED = fedBuffers->FEDData(idx);
    expectedSize += singleFED.size();
  }

  // build the FRDEvent into a temporary buffer
  std::unique_ptr<std::vector<unsigned char>> workBuffer(
      std::make_unique<std::vector<unsigned char>>(expectedSize + 256));
  uint32* bufPtr = (uint32*)(workBuffer.get()->data());
  if (frdVersion_ <= 5) {
    *bufPtr++ = (uint32)frdVersion_;  // version number only
  } else {
    uint16 flags = 0;
    if (!e.eventAuxiliary().isRealData())
      flags |= FRDEVENT_MASK_ISGENDATA;
    *(uint16*)bufPtr = (uint16)(frdVersion_ & 0xffff);
    *((uint16*)bufPtr + 1) = flags;
    bufPtr++;
  }
  *bufPtr++ = (uint32)e.id().run();
  *bufPtr++ = (uint32)e.luminosityBlock();
  *bufPtr++ = (uint32)e.id().event();
  if (frdVersion_ == 4)
    *bufPtr++ = 0;  //64-bit event id high part

  if (frdVersion_ < 3) {
    uint32 fedsize[1024];
    for (int idx = 0; idx < 1024; ++idx) {
      FEDRawData singleFED = fedBuffers->FEDData(idx);
      fedsize[idx] = singleFED.size();
      //std::cout << "fed size " << singleFED.size()<< std::endl;
    }
    memcpy(bufPtr, fedsize, 1024 * sizeof(uint32));
    bufPtr += 1024;
  } else {
    *bufPtr++ = expectedSize - headerSize;
    *bufPtr++ = 0;
    if (frdVersion_ <= 4)
      *bufPtr++ = 0;
  }
  uint32* payloadPtr = bufPtr;
  for (int idx = 0; idx < nFeds; ++idx) {
    FEDRawData singleFED = fedBuffers->FEDData(idx);
    if (singleFED.size() > 0) {
      memcpy(bufPtr, singleFED.data(), singleFED.size());
      bufPtr += singleFED.size() / 4;
    }
  }
  if (frdVersion_ > 4) {
    //crc32c checksum
    uint32_t crc = 0;
    *(payloadPtr - 1) = crc32c(crc, (const unsigned char*)payloadPtr, expectedSize - headerSize);
  } else if (frdVersion_ >= 3) {
    //adler32 checksum
    uint32 adlera = 1;
    uint32 adlerb = 0;
    cms::Adler32((const char*)payloadPtr, expectedSize - headerSize, adlera, adlerb);
    *(payloadPtr - 1) = (adlerb << 16) | adlera;
  }

  // create the FRDEventMsgView and use the template consumer to write it out
  FRDEventMsgView msg(workBuffer.get()->data());

  //write
  ssize_t retval = ::write(outfd_, (void*)msg.startAddress(), msg.size());

  if ((unsigned)retval != msg.size()) {
    throw cms::Exception("FRDOutputModule", "write")
        << "Error writing FED Raw Data event data to " << fileName_ << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }

  perFileEventCount_++;
  perFileSize_ += msg.size();
}

void FRDOutputModule::beginLuminosityBlock(edm::LuminosityBlockForOutput const& lumiBlock) {
  int ls = lumiBlock.id().luminosityBlock();

  if (outfd_ != -1)
    finishFileWrite(lumiBlock.run(), ls);

  if (fileWritten_)
    throw cms::Exception("RawEventFileWriterForBU", "beginLuminosityBlock")
        << "Multiple lumisections not supported in the same FRD file!";

  std::string fileName;
  if (fileName_.empty()) {
    std::stringstream ss;
    ss << (filePrefix_.empty() ? "" : filePrefix_ + "_") << "run" << std::setfill('0') << std::setw(6)
       << lumiBlock.run() << "_ls" << std::setfill('0') << std::setw(4) << ls << "_index000000.raw";
    fileName = ss.str();
  } else {
    //use exact filename (will be overwritten by last LS content if input contains multiple lumisections)
    fileName = fileName_;
  }

  outfd_ = open(fileName.c_str(), O_WRONLY | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
  ftruncate(outfd_, 0);

  if (outfd_ < 0) {  //attention here... it may happen that outfd_ is *not* set (e.g. missing initialize call...)
    throw cms::Exception("RawEventFileWriterForBU", "beginLuminosityBlock")
        << "Error opening FED Raw Data event output file: " << fileName << ": " << strerror(errno) << "\n";
  }
  edm::LogInfo("RawEventFileWriterForBU") << " Opened " << fileName;

  perFileEventCount_ = 0;
  perFileSize_ = 0;

  adlera_ = 1;
  adlerb_ = 0;

  if (frdFileVersion_ == 1) {
    //reserve space for file header
    ftruncate(outfd_, sizeof(FRDFileHeader_v1));
    lseek(outfd_, sizeof(FRDFileHeader_v1), SEEK_SET);
    perFileSize_ = sizeof(FRDFileHeader_v1);
  } else if (frdFileVersion_ == 2) {
    ftruncate(outfd_, sizeof(FRDFileHeader_v2));
    lseek(outfd_, sizeof(FRDFileHeader_v2), SEEK_SET);
    perFileSize_ = sizeof(FRDFileHeader_v2);
  } else
    throw cms::Exception("RawEventFileWriterForBU", "beginLuminosityBlock")
        << "Unsupported FRD version " << frdFileVersion_;
}

void FRDOutputModule::endLuminosityBlock(edm::LuminosityBlockForOutput const& lumiBlock) {
  finishFileWrite(lumiBlock.run(), lumiBlock.id().luminosityBlock());
}

void FRDOutputModule::finishFileWrite(unsigned int run, int ls) {
  if (outfd_ == -1)
    return;

  if (frdFileVersion_ == 1) {
    //rewind
    lseek(outfd_, 0, SEEK_SET);
    FRDFileHeader_v1 frdFileHeader(perFileEventCount_, (uint32_t)ls, perFileSize_);
    ::write(outfd_, (char*)&frdFileHeader, sizeof(FRDFileHeader_v1));
  } else if (frdFileVersion_ == 2) {
    lseek(outfd_, 0, SEEK_SET);
    FRDFileHeader_v2 frdFileHeader(0, perFileEventCount_, (uint32_t)run, (uint32_t)ls, perFileSize_);
    ::write(outfd_, (char*)&frdFileHeader, sizeof(FRDFileHeader_v2));
  }
  close(outfd_);
  outfd_ = -1;
  if (!fileName_.empty())
    fileWritten_ = true;

  edm::LogInfo("FRDOutputModule") << "closed RAW input file";
}
