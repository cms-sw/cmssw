#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/Common/interface/RandomNumberGeneratorState.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include <cstdio>
#include <iostream>

#include "FWCore/SharedMemory/interface/ReadBuffer.h"
#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/SharedMemory/interface/ROOTDeserializer.h"
#include "FWCore/SharedMemory/interface/WriteBuffer.h"
#include "FWCore/SharedMemory/interface/ROOTSerializer.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/engineIDulong.h"
#include "CLHEP/Random/RanecuEngine.h"

using namespace edm::shared_memory;
namespace testinter {

  using ReturnedType = std::pair<edmtest::IntProduct, edm::RandomNumberGeneratorState>;

  struct StreamCache {
    StreamCache(const std::string& iConfig, int id)
        : id_{id},
          channel_("testProd", id_, 60),
          readBuffer_{channel_.sharedMemoryName(), channel_.fromWorkerBufferInfo()},
          writeBuffer_{std::string("Rand") + channel_.sharedMemoryName(), channel_.toWorkerBufferInfo()},
          deserializer_{readBuffer_},
          bl_deserializer_{readBuffer_},
          randSerializer_{writeBuffer_} {
      //make sure output is flushed before popen does any writing
      fflush(stdout);
      fflush(stderr);

      channel_.setupWorker([&]() {
        using namespace std::string_literals;
        std::cout << id_ << " starting external process" << std::endl;
        pipe_ = popen(("cmsTestInterProcessRandom "s + channel_.sharedMemoryName() + " " + channel_.uniqueID()).c_str(),
                      "w");

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
                std::cout << id_ << " from shared memory " << value.first.value << std::endl;
              },
              iTrans,
              iTransitionID)) {
        std::cout << id_ << " FAILED waiting for external process" << std::endl;
        externalFailed_ = true;
        throw edm::Exception(edm::errors::ExternalFailure);
      }
      return value;
    }
    edmtest::IntProduct produce(unsigned long long iTransitionID, edm::StreamID iStream) {
      edm::Service<edm::RandomNumberGenerator> gen;
      auto& engine = gen->getEngine(iStream);
      edm::RandomNumberGeneratorState state{engine.put(), engine.getSeed()};
      randSerializer_.serialize(state);
      auto v = doTransition(deserializer_, edm::Transition::Event, iTransitionID);
      if (v.second.state_[0] != CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
        engine.setSeed(v.second.seed_, 0);
      }
      engine.get(v.second.state_);
      return v.first;
    }

    ReturnedType beginLumiProduce(edm::RandomNumberGeneratorState const& iState,
                                  unsigned long long iTransitionID,
                                  edm::LuminosityBlockIndex iLumi) {
      edm::Service<edm::RandomNumberGenerator> gen;
      //NOTE: root serialize requires a `void*` not a `void const*` even though it doesn't modify the object
      randSerializer_.serialize(const_cast<edm::RandomNumberGeneratorState&>(iState));
      return doTransition(bl_deserializer_, edm::Transition::BeginLuminosityBlock, iTransitionID);
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
    WriteBuffer writeBuffer_;

    using TCDeserializer = ROOTDeserializer<ReturnedType, ReadBuffer>;
    TCDeserializer deserializer_;
    TCDeserializer bl_deserializer_;
    using TCSerializer = ROOTSerializer<edm::RandomNumberGeneratorState, WriteBuffer>;
    TCSerializer randSerializer_;

    bool externalFailed_ = false;
  };

}  // namespace testinter

class TestInterProcessRandomProd
    : public edm::global::EDProducer<edm::StreamCache<testinter::StreamCache>,
                                     edm::LuminosityBlockCache<edm::RandomNumberGeneratorState>,
                                     edm::BeginLuminosityBlockProducer> {
public:
  TestInterProcessRandomProd(edm::ParameterSet const&);

  std::unique_ptr<testinter::StreamCache> beginStream(edm::StreamID) const final;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const final;

  void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final {}
  void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final {}

  std::shared_ptr<edm::RandomNumberGeneratorState> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                                              edm::EventSetup const&) const final;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final {}

  void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const final;
  void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const final;
  void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const final;

private:
  edm::EDPutTokenT<edmtest::IntProduct> const token_;
  edm::EDPutTokenT<edmtest::IntProduct> const blToken_;

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

TestInterProcessRandomProd::TestInterProcessRandomProd(edm::ParameterSet const& iPSet)
    : token_{produces<edmtest::IntProduct>()},
      blToken_{produces<edmtest::IntProduct, edm::Transition::BeginLuminosityBlock>("lumi")},
      config_{iPSet.getUntrackedParameter<std::string>("@python_config")} {}

std::unique_ptr<testinter::StreamCache> TestInterProcessRandomProd::beginStream(edm::StreamID iID) const {
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

void TestInterProcessRandomProd::produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto value = streamCache(iID)->produce(iEvent.id().event(), iID);
  iEvent.emplace(token_, value);
}

std::shared_ptr<edm::RandomNumberGeneratorState> TestInterProcessRandomProd::globalBeginLuminosityBlock(
    edm::LuminosityBlock const& iLumi, edm::EventSetup const&) const {
  edm::Service<edm::RandomNumberGenerator> gen;
  auto& engine = gen->getEngine(iLumi.index());
  return std::make_shared<edm::RandomNumberGeneratorState>(engine.put(), engine.getSeed());
}

void TestInterProcessRandomProd::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& iLuminosityBlock,
                                                                   edm::EventSetup const&) const {
  while (not availableForBeginLumi_.load()) {
  }

  auto v = availableForBeginLumi_.load()->beginLumiProduce(
      *luminosityBlockCache(iLuminosityBlock.index()), iLuminosityBlock.luminosityBlock(), iLuminosityBlock.index());
  edm::Service<edm::RandomNumberGenerator> gen;
  auto& engine = gen->getEngine(iLuminosityBlock.index());
  if (v.second.state_[0] != CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
    engine.setSeed(v.second.seed_, 0);
  }
  engine.get(v.second.state_);

  iLuminosityBlock.emplace(blToken_, v.first);

  lastLumiIndex_.store(iLuminosityBlock.index());
}

void TestInterProcessRandomProd::streamBeginLuminosityBlock(edm::StreamID iID,
                                                            edm::LuminosityBlock const& iLuminosityBlock,
                                                            edm::EventSetup const&) const {
  auto cache = streamCache(iID);
  if (cache != availableForBeginLumi_.load()) {
    (void)cache->beginLumiProduce(
        *luminosityBlockCache(iLuminosityBlock.index()), iLuminosityBlock.luminosityBlock(), iLuminosityBlock.index());
  } else {
    availableForBeginLumi_ = nullptr;
  }
}

void TestInterProcessRandomProd::streamEndLuminosityBlock(edm::StreamID iID,
                                                          edm::LuminosityBlock const& iLuminosityBlock,
                                                          edm::EventSetup const&) const {
  if (lastLumiIndex_ == iLuminosityBlock.index()) {
    testinter::StreamCache* expected = nullptr;

    availableForBeginLumi_.compare_exchange_strong(expected, streamCache(iID));
  }
}

DEFINE_FWK_MODULE(TestInterProcessRandomProd);
