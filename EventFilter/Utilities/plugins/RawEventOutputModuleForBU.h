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
  std::vector<unsigned int> sourceIdList_;
  unsigned int totevents_ = 0;
  unsigned int index_ = 0;
};

template <class Consumer>
RawEventOutputModuleForBU<Consumer>::RawEventOutputModuleForBU(edm::ParameterSet const& ps)
    : edm::one::OutputModuleBase::OutputModuleBase(ps),
      edm::one::OutputModule<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks>(ps),
      templateConsumer_(new Consumer(ps)),
      token_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("source"))),
      numEventsPerFile_(ps.getParameter<unsigned int>("numEventsPerFile")),
      frdVersion_(ps.getParameter<unsigned int>("frdVersion")),
      sourceIdList_(ps.getUntrackedParameter<std::vector<unsigned int>>("sourceIdList", std::vector<unsigned int>())) {
  if (frdVersion_ > 0 && frdVersion_ < 5)
    throw cms::Exception("RawEventOutputModuleForBU")
        << "Generating data with FRD version " << frdVersion_ << " is no longer supported";
  else if (frdVersion_ > edm::streamer::FRDHeaderMaxVersion)
    throw cms::Exception("RawEventOutputModuleForBU") << "Unknown FRD version " << frdVersion_;
}

template <class Consumer>
RawEventOutputModuleForBU<Consumer>::~RawEventOutputModuleForBU() {}

template <class Consumer>
void RawEventOutputModuleForBU<Consumer>::write(edm::EventForOutput const& e) {
  //using namespace edm::streamer;

  if (totevents_ > 0 && totevents_ % numEventsPerFile_ == 0) {
    index_++;
    unsigned int ls = e.luminosityBlock();
    std::string filename = edm::Service<evf::EvFDaqDirector>()->getOpenRawFilePath(ls, index_);
    std::string destinationDir = edm::Service<evf::EvFDaqDirector>()->buBaseRunDir();
    int run = edm::Service<evf::EvFDaqDirector>()->getRunNumber();
    templateConsumer_->initialize(destinationDir, filename, run, ls);
  }
  totevents_++;
  // serialize the FEDRawDataCollection into the format that we expect for
  // FRDEventMsgView objects (may be better ways to do this)
  edm::Handle<FEDRawDataCollection> fedBuffers;
  e.getByToken(token_, fedBuffers);

  // determine the expected size of the FRDEvent IN bytes
  int headerSize = edm::streamer::FRDHeaderVersionSize[frdVersion_];
  int expectedSize = headerSize;
  int nFeds = FEDNumbering::lastFEDId() + 1;

  if (sourceIdList_.size()) {
    for (int idx : sourceIdList_) {
      FEDRawData singleFED = fedBuffers->FEDData(idx);
      expectedSize += singleFED.size();
    }
  } else {
    for (int idx = 0; idx < nFeds; ++idx) {
      FEDRawData singleFED = fedBuffers->FEDData(idx);
      expectedSize += singleFED.size();
    }
  }

  // build the FRDEvent into a temporary buffer
  std::unique_ptr<std::vector<unsigned char>> workBuffer(
      std::make_unique<std::vector<unsigned char>>(expectedSize + 256));
  uint32_t* bufPtr = (uint32_t*)(workBuffer.get()->data());

  if (frdVersion_) {
    if (frdVersion_ <= 5) {
      //32-bits version field
      *bufPtr++ = (uint32_t)frdVersion_;
    } else {
      //16 bits version and 16 bits flags
      uint16_t flags = 0;
      if (!e.eventAuxiliary().isRealData())
        flags |= edm::streamer::FRDEVENT_MASK_ISGENDATA;
      *(uint16_t*)bufPtr = (uint16_t)(frdVersion_ & 0xffff);
      *((uint16_t*)bufPtr + 1) = flags;
      bufPtr++;
    }
    *bufPtr++ = (uint32_t)e.id().run();
    *bufPtr++ = (uint32_t)e.luminosityBlock();
    *bufPtr++ = (uint32_t)e.id().event();
    *bufPtr++ = expectedSize - headerSize;
    *bufPtr++ = 0;
  }
  uint32_t* payloadPtr = bufPtr;
  if (sourceIdList_.size())
    for (int idx : sourceIdList_) {
      FEDRawData singleFED = fedBuffers->FEDData(idx);
      if (singleFED.size() > 0) {
        memcpy(bufPtr, singleFED.data(), singleFED.size());
        bufPtr += singleFED.size() / 4;
      }
    }
  else
    for (int idx = 0; idx < nFeds; ++idx) {
      FEDRawData singleFED = fedBuffers->FEDData(idx);
      if (singleFED.size() > 0) {
        memcpy(bufPtr, singleFED.data(), singleFED.size());
        bufPtr += singleFED.size() / 4;
      }
    }
  if (frdVersion_) {
    //crc32c checksum
    uint32_t crc = 0;
    *(payloadPtr - 1) = crc32c(crc, (const unsigned char*)payloadPtr, expectedSize - headerSize);

    // create the FRDEventMsgView and use the template consumer to write it out
    edm::streamer::FRDEventMsgView msg(workBuffer.get()->data());
    templateConsumer_->doOutputEvent(msg);
  } else {
    //write only raw FEDs
    templateConsumer_->doOutputEvent((void*)workBuffer.get()->data(), expectedSize);
  }
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
  int run = edm::Service<evf::EvFDaqDirector>()->getRunNumber();
  std::cout << " writing to destination dir " << destinationDir << " name: " << filename << std::endl;
  totevents_ = 0;
  templateConsumer_->initialize(destinationDir, filename, run, ls.id().luminosityBlock());
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
  desc.addUntracked<std::vector<unsigned int>>("sourceIdList", std::vector<unsigned int>());
  Consumer::extendDescription(desc);

  descriptions.addWithDefaultLabel(desc);
}

#endif  // IOPool_Streamer_interface_RawEventOutputModuleForBU_h
