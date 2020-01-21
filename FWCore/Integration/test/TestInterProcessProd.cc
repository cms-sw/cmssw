#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

#include <stdio.h>
#include <iostream>

#include "TBufferFile.h"

#include "boost/interprocess/shared_memory_object.hpp"
#include "boost/interprocess/managed_shared_memory.hpp"
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace boost::interprocess;

namespace testinter {

  class SMReadBuffer {
    std::pair<char*, std::size_t> buffer_;
    char* bufferIndex_;
    char bufferOldIndex_;
    std::array<std::string, 2> bufferNames_;
    std::unique_ptr<managed_shared_memory> sm_;

  public:
    SMReadBuffer(std::string const& iUniqueName, char* iBufferIndex)
        : buffer_{nullptr, 0}, bufferIndex_{iBufferIndex}, bufferOldIndex_{3} {
      *bufferIndex_ = 0;
      bufferNames_[0] = iUniqueName + "buffer0";
      bufferNames_[1] = iUniqueName + "buffer1";
    }

    bool mustGetBufferAgain() const { return *bufferIndex_ != bufferOldIndex_; }

    std::pair<char*, std::size_t> buffer() {
      if (mustGetBufferAgain()) {
        sm_ = std::make_unique<managed_shared_memory>(open_only, bufferNames_[*bufferIndex_].c_str());
        buffer_ = sm_->find<char>("buffer");
        bufferOldIndex_ = *bufferIndex_;
      }
      return buffer_;
    }
  };

  template <typename T>
  class Deserializer {
  public:
    Deserializer(SMReadBuffer& iBuffer)
        : buffer_{iBuffer}, class_{TClass::GetClass(typeid(T))}, bufferFile_(TBuffer::kRead) {}

    T deserialize() {
      T value;
      if (buffer_.mustGetBufferAgain()) {
        auto buff = buffer_.buffer();
        bufferFile_.SetBuffer(buff.first, buff.second, kFALSE);
      }

      class_->ReadBuffer(bufferFile_, &value);
      bufferFile_.Reset();
      return value;
    }

  private:
    SMReadBuffer& buffer_;
    TClass* const class_;
    TBufferFile bufferFile_;
  };

  class ControllerChannel {
  public:
    ControllerChannel(std::string const& iName, int id)
        : id_{id},
          managed_sm_{open_or_create, unique_name(iName).c_str(), 1024},
          toWorkerBufferIndex_{bufferIndex("ToWorker", managed_sm_)},
          fromWorkerBufferIndex_{bufferIndex("FromWorker", managed_sm_)},
          named_mtx_{open_or_create, unique_name("mtx").c_str()},
          named_cndFromMain_{open_or_create, unique_name("cndFromMain").c_str()},
          named_cndToMain_{open_or_create, unique_name("cndToMain").c_str()} {
      managed_sm_.destroy<bool>("stop");
      stop_ = managed_sm_.construct<bool>("stop")(false);
      assert(stop_);

      managed_sm_.destroy<edm::Transition>("transitionType");
      transitionType_ = managed_sm_.construct<edm::Transition>("transitionType")(edm::Transition::NumberOfTransitions);
      assert(transitionType_);

      managed_sm_.destroy<unsigned long long>("transitionID");
      transitionID_ = managed_sm_.construct<unsigned long long>("transitionID")(0);
      assert(transitionID_);
    }

    template <typename F>
    void setupWorker(F&& iF) {
      scoped_lock<named_mutex> lock(named_mtx_);
      iF();
      using namespace boost::posix_time;
      std::cout << id_ << " waiting for external process" << std::endl;

      if (not named_cndToMain_.timed_wait(lock, microsec_clock::universal_time() + seconds(60))) {
        std::cout << id_ << " FAILED waiting for external process" << std::endl;
        throw cms::Exception("ExternalFailed");
      } else {
        std::cout << id_ << " done waiting for external process" << std::endl;
      }
    }

    template <typename F>
    bool doTransition(F&& iF, edm::Transition iTrans, unsigned long long iTransitionID) {
      std::cout << id_ << " taking from lock" << std::endl;
      scoped_lock<named_mutex> lock(named_mtx_);

      if (not wait(lock, iTrans, iTransitionID)) {
        return false;
      }

      iF();
      return true;
    }

    char* toWorkerBufferIndex() { return toWorkerBufferIndex_; }
    char* fromWorkerBufferIndex() { return fromWorkerBufferIndex_; }

    void stopWorker() {
      scoped_lock<named_mutex> lock(named_mtx_);
      *stop_ = true;
      named_cndFromMain_.notify_all();
    }

    ~ControllerChannel() {
      managed_sm_.destroy<bool>("stop");
      managed_sm_.destroy<unsigned int>("transitionType");
      managed_sm_.destroy<unsigned long long>("transitionID");
      managed_sm_.destroy<char>("bufferIndexToWorker");
      managed_sm_.destroy<char>("bufferIndexFromWorker");

      named_mutex::remove(unique_name("mtx").c_str());
      named_condition::remove(unique_name("cndFromMain").c_str());
      named_condition::remove(unique_name("cndToMain").c_str());
    }

  private:
    static char* bufferIndex(const char* iWhich, managed_shared_memory& mem) {
      std::string name("bufferIndex");
      name += iWhich;
      mem.destroy<char>(name.c_str());
      char* v = mem.construct<char>(name.c_str())();
      return v;
    }

    std::string unique_name(std::string iBase) {
      auto pid = getpid();
      iBase += std::to_string(pid);
      iBase += "_";
      iBase += std::to_string(id_);

      return iBase;
    }

    bool wait(scoped_lock<named_mutex>& lock, edm::Transition iTrans, unsigned long long iTransID) {
      *transitionType_ = iTrans;
      *transitionID_ = iTransID;
      {
        std::cout << id_ << " notifying" << std::endl;
        named_cndFromMain_.notify_all();
      }

      std::cout << id_ << " waiting" << std::endl;
      using namespace boost::posix_time;
      if (not named_cndToMain_.timed_wait(lock, microsec_clock::universal_time() + seconds(60))) {
        return false;
      }
      return true;
    }

    int id_;
    managed_shared_memory managed_sm_;
    char* toWorkerBufferIndex_;
    char* fromWorkerBufferIndex_;

    named_mutex named_mtx_;
    named_condition named_cndFromMain_;

    named_condition named_cndToMain_;

    edm::Transition* transitionType_;
    unsigned long long* transitionID_;
    bool* stop_;
  };
  struct StreamCache {
    StreamCache(const char* iConfig, int id)
        : id_{id},
          channel_("testProd", id_),
          readBuffer_{unique_name("testProd").c_str(), channel_.fromWorkerBufferIndex()},
          deserializer_{readBuffer_},
          br_deserializer_{readBuffer_},
          er_deserializer_{readBuffer_},
          bl_deserializer_{readBuffer_},
          el_deserializer_(readBuffer_) {
      //make sure output is flushed before popen does any writing
      fflush(stdout);
      fflush(stderr);

      channel_.setupWorker([&]() {
        std::cout << id_ << " starting external process" << std::endl;
        pipe_ = popen(unique_name("cmsInterProcess testProd ").c_str(), "w");

        if (NULL == pipe_) {
          abort();
        }

        {
          auto length = strlen(iConfig);
          auto nlines = std::to_string(std::count(iConfig, iConfig + length, '\n'));
          auto result = fwrite(nlines.data(), sizeof(char), nlines.size(), pipe_);
          assert(result = nlines.size());
          result = fwrite(iConfig, sizeof(char), strlen(iConfig), pipe_);
          assert(result == strlen(iConfig));
          fflush(pipe_);
        }
      });
    }

    template <typename SERIAL>
    auto doTransition(SERIAL& iDeserializer, edm::Transition iTrans, unsigned long long iTransitionID)
        -> decltype(iDeserializer.deserialize()) {
      decltype(iDeserializer.deserialize()) value;
      if (not channel_.doTransition(
              [&value, this]() {
                value = deserializer_.deserialize();
                std::cout << id_ << " from shared memory " << value.size() << std::endl;
              },
              iTrans,
              iTransitionID)) {
        std::cout << id_ << " FAILED waiting for external process" << std::endl;
        externalFailed_ = true;
        throw cms::Exception("ExternalFailed");
      }
      return value;
    }
    edmtest::ThingCollection produce(unsigned long long iTransitionID) {
      return doTransition(deserializer_, edm::Transition::Event, iTransitionID);
    }

    edmtest::ThingCollection beginRunProduce(unsigned long long iTransitionID) {
      return doTransition(br_deserializer_, edm::Transition::BeginRun, iTransitionID);
    }

    edmtest::ThingCollection endRunProduce(unsigned long long iTransitionID) {
      if (not externalFailed_) {
        return doTransition(er_deserializer_, edm::Transition::EndRun, iTransitionID);
      }
      return edmtest::ThingCollection();
    }

    edmtest::ThingCollection beginLumiProduce(unsigned long long iTransitionID) {
      return doTransition(bl_deserializer_, edm::Transition::BeginLuminosityBlock, iTransitionID);
    }

    edmtest::ThingCollection endLumiProduce(unsigned long long iTransitionID) {
      if (not externalFailed_) {
        return doTransition(el_deserializer_, edm::Transition::EndLuminosityBlock, iTransitionID);
      }
      return edmtest::ThingCollection();
    }

    ~StreamCache() {
      channel_.stopWorker();
      pclose(pipe_);
    }

  private:
    std::string unique_name(std::string iBase) {
      auto pid = getpid();
      iBase += std::to_string(pid);
      iBase += "_";
      iBase += std::to_string(id_);

      return iBase;
    }

    int id_;
    FILE* pipe_;
    ControllerChannel channel_;
    SMReadBuffer readBuffer_;

    testinter::Deserializer<edmtest::ThingCollection> deserializer_;
    testinter::Deserializer<edmtest::ThingCollection> br_deserializer_;
    testinter::Deserializer<edmtest::ThingCollection> er_deserializer_;
    testinter::Deserializer<edmtest::ThingCollection> bl_deserializer_;
    testinter::Deserializer<edmtest::ThingCollection> el_deserializer_;
    bool externalFailed_ = false;
  };

  struct RunCache {
    mutable edmtest::ThingCollection thingCollection_;
  };
  struct LumiCache {
    mutable edmtest::ThingCollection thingCollection_;
  };
}  // namespace testinter

class TestInterProcessProd : public edm::global::EDProducer<edm::StreamCache<testinter::StreamCache>,
                                                            edm::RunCache<testinter::RunCache>,
                                                            edm::BeginRunProducer,
                                                            edm::EndRunProducer,
                                                            edm::LuminosityBlockCache<testinter::LumiCache>,
                                                            edm::BeginLuminosityBlockProducer,
                                                            edm::EndLuminosityBlockProducer> {
public:
  TestInterProcessProd(edm::ParameterSet const&);

  std::unique_ptr<testinter::StreamCache> beginStream(edm::StreamID) const final;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const final;

  void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const final;
  std::shared_ptr<testinter::RunCache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final;
  void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final;
  void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const final {}
  void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const final;

  void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const final;
  std::shared_ptr<testinter::LumiCache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                                   edm::EventSetup const&) const final;
  void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const final;
  void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const final;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final {}
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const final;

private:
  edm::EDPutTokenT<edmtest::ThingCollection> const token_;
  edm::EDPutTokenT<edmtest::ThingCollection> const brToken_;
  edm::EDPutTokenT<edmtest::ThingCollection> const erToken_;
  edm::EDPutTokenT<edmtest::ThingCollection> const blToken_;
  edm::EDPutTokenT<edmtest::ThingCollection> const elToken_;

  mutable testinter::StreamCache* stream0Cache_ = nullptr;
  mutable std::atomic<testinter::StreamCache*> availableForBeginLumi_;
  mutable std::atomic<unsigned int> lastLumiIndex_ = 0;
};

TestInterProcessProd::TestInterProcessProd(edm::ParameterSet const&)
    : token_{produces<edmtest::ThingCollection>()},
      brToken_{produces<edmtest::ThingCollection, edm::Transition::BeginRun>("beginRun")},
      erToken_{produces<edmtest::ThingCollection, edm::Transition::EndRun>("endRun")},
      blToken_{produces<edmtest::ThingCollection, edm::Transition::BeginLuminosityBlock>("beginLumi")},
      elToken_{produces<edmtest::ThingCollection, edm::Transition::EndLuminosityBlock>("endLumi")} {}

std::unique_ptr<testinter::StreamCache> TestInterProcessProd::beginStream(edm::StreamID iID) const {
  constexpr auto v = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.gen = cms.EDProducer("ThingProducer", nThings = cms.int32(100), grow=cms.bool(True), offsetDelta = cms.int32(1))
process.moduleToTest(process.gen)
process.add_(cms.Service("InitRootHandlers", UnloadRootSigHandler=cms.untracked.bool(True)))
)_";

  auto cache = std::make_unique<testinter::StreamCache>(v, iID.value());
  if (iID.value() == 0) {
    stream0Cache_ = cache.get();

    availableForBeginLumi_ = stream0Cache_;
  }

  return cache;
}

void TestInterProcessProd::produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto value = streamCache(iID)->produce(iEvent.id().event());
  iEvent.emplace(token_, value);
}

void TestInterProcessProd::globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&) const {
  auto v = stream0Cache_->beginRunProduce(iRun.run());
  iRun.emplace(brToken_, v);
}
std::shared_ptr<testinter::RunCache> TestInterProcessProd::globalBeginRun(edm::Run const&,
                                                                          edm::EventSetup const&) const {
  return std::make_shared<testinter::RunCache>();
}

void TestInterProcessProd::streamBeginRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const {
  if (iID.value() != 0) {
    (void)streamCache(iID)->beginRunProduce(iRun.run());
  }
}
void TestInterProcessProd::streamEndRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const {
  if (iID.value() == 0) {
    runCache(iRun.index())->thingCollection_ = streamCache(iID)->endRunProduce(iRun.run());
  } else {
    (void)streamCache(iID)->endRunProduce(iRun.run());
  }
}
void TestInterProcessProd::globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&) const {
  iRun.emplace(erToken_, std::move(runCache(iRun.index())->thingCollection_));
}

void TestInterProcessProd::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& iLuminosityBlock,
                                                             edm::EventSetup const&) const {
  while (not availableForBeginLumi_.load()) {
  }

  auto v = availableForBeginLumi_.load()->beginLumiProduce(iLuminosityBlock.run());
  iLuminosityBlock.emplace(blToken_, v);

  lastLumiIndex_.store(iLuminosityBlock.index());
}

std::shared_ptr<testinter::LumiCache> TestInterProcessProd::globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                                                       edm::EventSetup const&) const {
  return std::make_shared<testinter::LumiCache>();
}

void TestInterProcessProd::streamBeginLuminosityBlock(edm::StreamID iID,
                                                      edm::LuminosityBlock const& iLuminosityBlock,
                                                      edm::EventSetup const&) const {
  auto cache = streamCache(iID);
  if (cache != availableForBeginLumi_.load()) {
    (void)cache->beginLumiProduce(iLuminosityBlock.run());
  } else {
    availableForBeginLumi_ = nullptr;
  }
}

void TestInterProcessProd::streamEndLuminosityBlock(edm::StreamID iID,
                                                    edm::LuminosityBlock const& iLuminosityBlock,
                                                    edm::EventSetup const&) const {
  if (iID.value() == 0) {
    luminosityBlockCache(iLuminosityBlock.index())->thingCollection_ =
        streamCache(iID)->endLumiProduce(iLuminosityBlock.run());
  } else {
    (void)streamCache(iID)->endLumiProduce(iLuminosityBlock.run());
  }

  if (lastLumiIndex_ == iLuminosityBlock.index()) {
    testinter::StreamCache* expected = nullptr;

    availableForBeginLumi_.compare_exchange_strong(expected, streamCache(iID));
  }
}

void TestInterProcessProd::globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLuminosityBlock,
                                                           edm::EventSetup const&) const {
  iLuminosityBlock.emplace(elToken_, std::move(luminosityBlockCache(iLuminosityBlock.index())->thingCollection_));
}

DEFINE_FWK_MODULE(TestInterProcessProd);