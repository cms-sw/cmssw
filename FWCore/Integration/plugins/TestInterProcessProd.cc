#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

#include <cstdio>
#include <iostream>

#include "FWCore/SharedMemory/interface/ReadBuffer.h"
#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/SharedMemory/interface/ROOTDeserializer.h"

using namespace edm::shared_memory;
namespace testinter {

  struct StreamCache {
    StreamCache(const std::string& iConfig, int id)
        : id_{id},
          channel_("testProd", id_, 60),
          readBuffer_{channel_.sharedMemoryName(), channel_.fromWorkerBufferInfo()},
          deserializer_{readBuffer_},
          br_deserializer_{readBuffer_},
          er_deserializer_{readBuffer_},
          bl_deserializer_{readBuffer_},
          el_deserializer_(readBuffer_) {
      //make sure output is flushed before popen does any writing
      fflush(stdout);
      fflush(stderr);

      channel_.setupWorker([&]() {
        using namespace std::string_literals;
        std::cout << id_ << " starting external process" << std::endl;
        pipe_ = popen(("cmsTestInterProcess "s + channel_.sharedMemoryName() + " " + channel_.uniqueID()).c_str(), "w");

        if (nullptr == pipe_) {
          abort();
        }

        {
          auto nlines = std::to_string(std::count(iConfig.begin(), iConfig.end(), '\n'));
          auto result = fwrite(nlines.data(), sizeof(char), nlines.size(), pipe_);
          assert(result == nlines.size());
          result = fwrite(iConfig.data(), sizeof(char), iConfig.size(), pipe_);
          assert(result == iConfig.size());
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
        throw edm::Exception(edm::errors::ExternalFailure);
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
    ReadBuffer readBuffer_;

    using TCDeserializer = ROOTDeserializer<edmtest::ThingCollection, ReadBuffer>;
    TCDeserializer deserializer_;
    TCDeserializer br_deserializer_;
    TCDeserializer er_deserializer_;
    TCDeserializer bl_deserializer_;
    TCDeserializer el_deserializer_;
    bool externalFailed_ = false;
  };

  struct RunCache {
    //Only stream 0 sets this at stream end Run and it is read at global end run
    // the framework guarantees those calls can not happen simultaneously
    CMS_THREAD_SAFE mutable edmtest::ThingCollection thingCollection_;
  };
  struct LumiCache {
    //Only stream 0 sets this at stream end Lumi and it is read at global end Lumi
    // the framework guarantees those calls can not happen simultaneously
    CMS_THREAD_SAFE mutable edmtest::ThingCollection thingCollection_;
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

  std::string config_;

  //This is set at beginStream and used for globalBeginRun
  //The framework guarantees that non of those can happen concurrently
  CMS_THREAD_SAFE mutable testinter::StreamCache* stream0Cache_ = nullptr;
  //A stream which has finished processing the last lumi is used for the
  // call to globalBeginLuminosityBlockProduce
  mutable std::atomic<testinter::StreamCache*> availableForBeginLumi_;
  //Streams all see the lumis in the same order, we want to be sure to pick a stream cache
  // to use at globalBeginLumi which just finished the most recent lumi and not a previous one
  mutable std::atomic<unsigned int> lastLumiIndex_ = 0;
};

TestInterProcessProd::TestInterProcessProd(edm::ParameterSet const& iPSet)
    : token_{produces<edmtest::ThingCollection>()},
      brToken_{produces<edmtest::ThingCollection, edm::Transition::BeginRun>("beginRun")},
      erToken_{produces<edmtest::ThingCollection, edm::Transition::EndRun>("endRun")},
      blToken_{produces<edmtest::ThingCollection, edm::Transition::BeginLuminosityBlock>("beginLumi")},
      elToken_{produces<edmtest::ThingCollection, edm::Transition::EndLuminosityBlock>("endLumi")},
      config_{iPSet.getUntrackedParameter<std::string>("@python_config")} {}

std::unique_ptr<testinter::StreamCache> TestInterProcessProd::beginStream(edm::StreamID iID) const {
  auto const label = moduleDescription().moduleLabel();

  using namespace std::string_literals;

  std::string config = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
)_";
  config += "process."s + label + "=" + config_ + "\n";
  config += "process.moduleToTest(process."s + label + ")\n";
  config += R"_(
process.add_(cms.Service("InitRootHandlers", UnloadRootSigHandler=cms.untracked.bool(True)))
  )_";

  auto cache = std::make_unique<testinter::StreamCache>(config, iID.value());
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
