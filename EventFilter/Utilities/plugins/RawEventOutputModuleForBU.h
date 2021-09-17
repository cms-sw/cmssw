#ifndef IOPool_Streamer_interface_RawEventOutputModuleForBU_h
#define IOPool_Streamer_interface_RawEventOutputModuleForBU_h

#include <memory>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/crc32c.h"
#include "EventFilter/Utilities/plugins/EvFBuildingThrottle.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

template <class Consumer>
class RawEventOutputModuleForBU : public edm::one::OutputModule<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
  typedef unsigned int uint32;
  /**
   * Consumers are suppose to provide:
   *   void doOutputEvent(const FRDEventMsgView& msg)
   *   void start()
   *   void stop()
   */

public:
  explicit RawEventOutputModuleForBU(edm::ParameterSet const& ps);
  ~RawEventOutputModuleForBU() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void write(edm::EventForOutput const& e) override;
  void beginRun(edm::RunForOutput const&) override;
  void endRun(edm::RunForOutput const&) override;
  void writeRun(const edm::RunForOutput&) override {}
  void writeLuminosityBlock(const edm::LuminosityBlockForOutput&) override {}

  void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
  void endLuminosityBlock(edm::LuminosityBlockForOutput const&) override;

  std::unique_ptr<Consumer> templateConsumer_;
  const edm::EDGetTokenT<FEDRawDataCollection> token_;
  const unsigned int numEventsPerFile_;
  const unsigned int frdVersion_;
  unsigned long long totsize = 0LL;
  unsigned long long writtensize = 0LL;
  unsigned long long writtenSizeLast = 0LL;
  unsigned int totevents = 0;
  unsigned int index_ = 0;
  timeval startOfLastLumi;
  bool firstLumi_ = true;
};

template <class Consumer>
RawEventOutputModuleForBU<Consumer>::RawEventOutputModuleForBU(edm::ParameterSet const& ps)
    : edm::one::OutputModuleBase::OutputModuleBase(ps),
      edm::one::OutputModule<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks>(ps),
      templateConsumer_(new Consumer(ps)),
      token_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("source"))),
      numEventsPerFile_(ps.getParameter<unsigned int>("numEventsPerFile")),
      frdVersion_(ps.getParameter<unsigned int>("frdVersion")) {}

template <class Consumer>
RawEventOutputModuleForBU<Consumer>::~RawEventOutputModuleForBU() {}

template <class Consumer>
void RawEventOutputModuleForBU<Consumer>::write(edm::EventForOutput const& e) {
  unsigned int ls = e.luminosityBlock();
  if (totevents > 0 && totevents % numEventsPerFile_ == 0) {
    index_++;
    std::string filename = edm::Service<evf::EvFDaqDirector>()->getOpenRawFilePath(ls, index_);
    std::string destinationDir = edm::Service<evf::EvFDaqDirector>()->buBaseRunDir();
    templateConsumer_->initialize(destinationDir, filename, ls);
  }
  totevents++;
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

  totsize += expectedSize;
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
  writtensize += msg.size();

  templateConsumer_->doOutputEvent(msg);
}

template <class Consumer>
void RawEventOutputModuleForBU<Consumer>::beginRun(edm::RunForOutput const&) {
  // edm::Service<evf::EvFDaqDirector>()->updateBuLock(1);
  templateConsumer_->start();
}

template <class Consumer>
void RawEventOutputModuleForBU<Consumer>::endRun(edm::RunForOutput const&) {
  templateConsumer_->stop();
}

template <class Consumer>
void RawEventOutputModuleForBU<Consumer>::beginLuminosityBlock(edm::LuminosityBlockForOutput const& ls) {
  index_ = 0;
  std::string filename = edm::Service<evf::EvFDaqDirector>()->getOpenRawFilePath(ls.id().luminosityBlock(), index_);
  std::string destinationDir = edm::Service<evf::EvFDaqDirector>()->buBaseRunDir();
  std::cout << " writing to destination dir " << destinationDir << " name: " << filename << std::endl;
  templateConsumer_->initialize(destinationDir, filename, ls.id().luminosityBlock());
  //edm::Service<evf::EvFDaqDirector>()->updateBuLock(ls.id().luminosityBlock()+1);
  if (!firstLumi_) {
    timeval now;
    ::gettimeofday(&now, nullptr);
    //long long elapsedusec = (now.tv_sec - startOfLastLumi.tv_sec)*1000000+now.tv_usec-startOfLastLumi.tv_usec;
    /*     std::cout << "(now.tv_sec - startOfLastLumi.tv_sec) " << now.tv_sec <<"-" << startOfLastLumi.tv_sec */
    /* 	      <<" (now.tv_usec-startOfLastLumi.tv_usec) " << now.tv_usec << "-" << startOfLastLumi.tv_usec << std::endl; */
    /*     std::cout << "elapsedusec " << elapsedusec << "  totevents " << totevents << "  size (GB)" << writtensize  */
    /* 	      << "  rate " << (writtensize-writtenSizeLast)/elapsedusec << " MB/s" <<std::endl; */
    writtenSizeLast = writtensize;
    ::gettimeofday(&startOfLastLumi, nullptr);
    //edm::Service<evf::EvFDaqDirector>()->writeLsStatisticsBU(ls.id().luminosityBlock(), totevents, totsize, elapsedusec);
  } else
    ::gettimeofday(&startOfLastLumi, nullptr);
  totevents = 0;
  totsize = 0LL;
  firstLumi_ = false;
}

template <class Consumer>
void RawEventOutputModuleForBU<Consumer>::endLuminosityBlock(edm::LuminosityBlockForOutput const& ls) {
  //  templateConsumer_->touchlock(ls.id().luminosityBlock(),basedir);
  templateConsumer_->endOfLS(ls.id().luminosityBlock());
}

template <class Consumer>
void RawEventOutputModuleForBU<Consumer>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag("rawDataCollector"));
  desc.add<unsigned int>("numEventsPerFile", 100);
  desc.add<unsigned int>("frdVersion", 6);
  Consumer::extendDescription(desc);

  descriptions.addWithDefaultLabel(desc);
}

#endif  // IOPool_Streamer_interface_RawEventOutputModuleForBU_h
