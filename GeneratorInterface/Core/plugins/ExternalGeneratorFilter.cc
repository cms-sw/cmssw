#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/ExternalGeneratorEventInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ExternalGeneratorLumiInfo.h"

#include "FWCore/SharedMemory/interface/ReadBuffer.h"
#include "FWCore/SharedMemory/interface/WriteBuffer.h"
#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/SharedMemory/interface/ROOTDeserializer.h"
#include "FWCore/SharedMemory/interface/ROOTSerializer.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/engineIDulong.h"
#include "CLHEP/Random/RanecuEngine.h"

#include <cstdio>
#include <iostream>

using namespace edm::shared_memory;
namespace externalgen {

  struct StreamCache {
    StreamCache(const std::string& iConfig, int id, bool verbose, unsigned int waitTime)
        : id_{id},
          channel_("extGen", id_, waitTime),
          readBuffer_{channel_.sharedMemoryName(), channel_.fromWorkerBufferInfo()},
          writeBuffer_{std::string("Rand") + channel_.sharedMemoryName(), channel_.toWorkerBufferInfo()},
          deserializer_{readBuffer_},
          er_deserializer_{readBuffer_},
          bl_deserializer_{readBuffer_},
          el_deserializer_(readBuffer_),
          randSerializer_(writeBuffer_) {
      //make sure output is flushed before popen does any writing
      fflush(stdout);
      fflush(stderr);

      channel_.setupWorker([&]() {
        using namespace std::string_literals;
        using namespace std::filesystem;
        edm::LogSystem("ExternalProcess") << id_ << " starting external process \n";
        std::string verboseCommand;
        if (verbose) {
          verboseCommand = "--verbose ";
        }
        auto curDir = current_path();
        auto newDir = path("thread"s + std::to_string(id_));
        create_directory(newDir);
        current_path(newDir);
        pipe_ =
            popen(("cmsExternalGenerator "s + verboseCommand + channel_.sharedMemoryName() + " " + channel_.uniqueID())
                      .c_str(),
                  "w");
        current_path(curDir);

        if (nullptr == pipe_) {
          abort();
        }

        {
          auto nlines = std::to_string(std::count(iConfig.begin(), iConfig.end(), '\n'));
          auto result = fwrite(nlines.data(), sizeof(char), nlines.size(), pipe_);
          assert(result = nlines.size());
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
              [&value, &iDeserializer]() { value = iDeserializer.deserialize(); }, iTrans, iTransitionID)) {
        externalFailed_ = true;
        throw cms::Exception("ExternalFailed")
            << "failed waiting for external process. Timed out after " << channel_.maxWaitInSeconds() << " seconds.";
      }
      return value;
    }
    ExternalGeneratorEventInfo produce(edm::StreamID iStream, unsigned long long iTransitionID) {
      edm::Service<edm::RandomNumberGenerator> gen;
      auto& engine = gen->getEngine(iStream);
      edm::RandomNumberGeneratorState state{engine.put(), engine.getSeed()};
      randSerializer_.serialize(state);

      return doTransition(deserializer_, edm::Transition::Event, iTransitionID);
    }

    std::optional<GenRunInfoProduct> endRunProduce(unsigned long long iTransitionID) {
      if (not externalFailed_) {
        return doTransition(er_deserializer_, edm::Transition::EndRun, iTransitionID);
      }
      return {};
    }

    ExternalGeneratorLumiInfo beginLumiProduce(unsigned long long iTransitionID,
                                               edm::RandomNumberGeneratorState const& iState) {
      //NOTE: root serialize requires a `void*` not a `void const*` even though it doesn't modify the object
      randSerializer_.serialize(const_cast<edm::RandomNumberGeneratorState&>(iState));
      return doTransition(bl_deserializer_, edm::Transition::BeginLuminosityBlock, iTransitionID);
    }

    std::optional<GenLumiInfoProduct> endLumiProduce(unsigned long long iTransitionID) {
      if (not externalFailed_) {
        return doTransition(el_deserializer_, edm::Transition::EndLuminosityBlock, iTransitionID);
      }
      return {};
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

    template <typename T>
    using Deserializer = ROOTDeserializer<T, ReadBuffer>;
    Deserializer<ExternalGeneratorEventInfo> deserializer_;
    Deserializer<GenRunInfoProduct> er_deserializer_;
    Deserializer<ExternalGeneratorLumiInfo> bl_deserializer_;
    Deserializer<GenLumiInfoProduct> el_deserializer_;
    ROOTSerializer<edm::RandomNumberGeneratorState, WriteBuffer> randSerializer_;

    bool externalFailed_ = false;
  };

  struct RunCache {
    //Only stream 0 sets this at stream end Run and it is read at global end run
    // the framework guarantees those calls can not happen simultaneously
    CMS_THREAD_SAFE mutable GenRunInfoProduct runInfo_;
  };
  struct LumiCache {
    LumiCache(std::vector<unsigned long> iState, long iSeed) : randomState_(std::move(iState), iSeed) {}
    //Only stream 0 sets this at stream end Lumi and it is read at global end Lumi
    // the framework guarantees those calls can not happen simultaneously
    CMS_THREAD_SAFE mutable edm::RandomNumberGeneratorState randomState_;
  };
}  // namespace externalgen

class ExternalGeneratorFilter : public edm::global::EDFilter<edm::StreamCache<externalgen::StreamCache>,
                                                             edm::RunCache<externalgen::RunCache>,
                                                             edm::EndRunProducer,
                                                             edm::LuminosityBlockCache<externalgen::LumiCache>,
                                                             edm::LuminosityBlockSummaryCache<GenLumiInfoProduct>,
                                                             edm::BeginLuminosityBlockProducer,
                                                             edm::EndLuminosityBlockProducer> {
public:
  ExternalGeneratorFilter(edm::ParameterSet const&);

  std::unique_ptr<externalgen::StreamCache> beginStream(edm::StreamID) const final;
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const final;

  std::shared_ptr<externalgen::RunCache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final;
  void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final;
  void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const final {}
  void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const final;

  void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const final;
  std::shared_ptr<externalgen::LumiCache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                                     edm::EventSetup const&) const final;
  std::shared_ptr<GenLumiInfoProduct> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                        edm::EventSetup const&) const final;
  void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const final;
  void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const final;
  void streamEndLuminosityBlockSummary(edm::StreamID,
                                       edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       GenLumiInfoProduct*) const final;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final {}
  void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       GenLumiInfoProduct*) const final {}
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock&,
                                       edm::EventSetup const&,
                                       GenLumiInfoProduct const*) const final;

private:
  edm::EDPutTokenT<edm::HepMCProduct> const hepMCToken_;
  edm::EDPutTokenT<GenEventInfoProduct> const genEventToken_;
  edm::EDPutTokenT<GenRunInfoProduct> const runInfoToken_;
  edm::EDPutTokenT<GenLumiInfoHeader> const lumiHeaderToken_;
  edm::EDPutTokenT<GenLumiInfoProduct> const lumiInfoToken_;

  std::string const config_;
  bool const verbose_;
  unsigned int waitTime_;
  std::string const extraConfig_;

  //This is set at beginStream and used for globalBeginRun
  //The framework guarantees that non of those can happen concurrently
  CMS_THREAD_SAFE mutable externalgen::StreamCache* stream0Cache_ = nullptr;
  //A stream which has finished processing the last lumi is used for the
  // call to globalBeginLuminosityBlockProduce
  mutable std::atomic<externalgen::StreamCache*> availableForBeginLumi_;
  //Streams all see the lumis in the same order, we want to be sure to pick a stream cache
  // to use at globalBeginLumi which just finished the most recent lumi and not a previous one
  mutable std::atomic<unsigned int> lastLumiIndex_ = 0;
};

ExternalGeneratorFilter::ExternalGeneratorFilter(edm::ParameterSet const& iPSet)
    : hepMCToken_{produces<edm::HepMCProduct>("unsmeared")},
      genEventToken_{produces<GenEventInfoProduct>()},
      runInfoToken_{produces<GenRunInfoProduct, edm::Transition::EndRun>()},
      lumiHeaderToken_{produces<GenLumiInfoHeader, edm::Transition::BeginLuminosityBlock>()},
      lumiInfoToken_{produces<GenLumiInfoProduct, edm::Transition::EndLuminosityBlock>()},
      config_{iPSet.getUntrackedParameter<std::string>("@python_config")},
      verbose_{iPSet.getUntrackedParameter<bool>("_external_process_verbose_")},
      waitTime_{iPSet.getUntrackedParameter<unsigned int>("_external_process_waitTime_")},
      extraConfig_{iPSet.getUntrackedParameter<std::string>("_external_process_extraConfig_")} {}

std::unique_ptr<externalgen::StreamCache> ExternalGeneratorFilter::beginStream(edm::StreamID iID) const {
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
  if (not extraConfig_.empty()) {
    config += "\n";
    config += extraConfig_;
  }

  auto cache = std::make_unique<externalgen::StreamCache>(config, iID.value(), verbose_, waitTime_);
  if (iID.value() == 0) {
    stream0Cache_ = cache.get();

    availableForBeginLumi_ = stream0Cache_;
  }

  return cache;
}

bool ExternalGeneratorFilter::filter(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto value = streamCache(iID)->produce(iID, iEvent.id().event());

  edm::Service<edm::RandomNumberGenerator> gen;
  auto& engine = gen->getEngine(iID);
  //if (value.randomState_.state_[0] != CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
  //  engine.setSeed(value.randomState_.seed_, 0);
  //}
  engine.get(value.randomState_.state_);

  iEvent.emplace(hepMCToken_, std::move(value.hepmc_));
  iEvent.emplace(genEventToken_, std::move(value.eventInfo_));
  return value.keepEvent_;
}

std::shared_ptr<externalgen::RunCache> ExternalGeneratorFilter::globalBeginRun(edm::Run const&,
                                                                               edm::EventSetup const&) const {
  return std::make_shared<externalgen::RunCache>();
}

void ExternalGeneratorFilter::streamBeginRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const {}
void ExternalGeneratorFilter::streamEndRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const {
  if (iID.value() == 0) {
    runCache(iRun.index())->runInfo_ = *streamCache(iID)->endRunProduce(iRun.run());
  } else {
    (void)streamCache(iID)->endRunProduce(iRun.run());
  }
}
void ExternalGeneratorFilter::globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&) const {
  iRun.emplace(runInfoToken_, std::move(runCache(iRun.index())->runInfo_));
}

void ExternalGeneratorFilter::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& iLuminosityBlock,
                                                                edm::EventSetup const&) const {
  while (not availableForBeginLumi_.load()) {
  }

  auto v = availableForBeginLumi_.load()->beginLumiProduce(
      iLuminosityBlock.luminosityBlock(), luminosityBlockCache(iLuminosityBlock.index())->randomState_);

  edm::Service<edm::RandomNumberGenerator> gen;
  auto& engine = gen->getEngine(iLuminosityBlock.index());
  engine.get(v.randomState_.state_);

  iLuminosityBlock.emplace(lumiHeaderToken_, std::move(v.header_));

  lastLumiIndex_.store(iLuminosityBlock.index());
}

std::shared_ptr<externalgen::LumiCache> ExternalGeneratorFilter::globalBeginLuminosityBlock(
    edm::LuminosityBlock const& iLumi, edm::EventSetup const&) const {
  edm::Service<edm::RandomNumberGenerator> gen;
  auto& engine = gen->getEngine(iLumi.index());
  auto s = engine.put();
  return std::make_shared<externalgen::LumiCache>(s, engine.getSeed());
}

std::shared_ptr<GenLumiInfoProduct> ExternalGeneratorFilter::globalBeginLuminosityBlockSummary(
    edm::LuminosityBlock const&, edm::EventSetup const&) const {
  return std::make_shared<GenLumiInfoProduct>();
}

void ExternalGeneratorFilter::streamBeginLuminosityBlock(edm::StreamID iID,
                                                         edm::LuminosityBlock const& iLuminosityBlock,
                                                         edm::EventSetup const&) const {
  auto cache = streamCache(iID);
  if (cache != availableForBeginLumi_.load()) {
    (void)cache->beginLumiProduce(iLuminosityBlock.run(), luminosityBlockCache(iLuminosityBlock.index())->randomState_);
  } else {
    availableForBeginLumi_ = nullptr;
  }
}

void ExternalGeneratorFilter::streamEndLuminosityBlock(edm::StreamID iID,
                                                       edm::LuminosityBlock const& iLuminosityBlock,
                                                       edm::EventSetup const&) const {}

void ExternalGeneratorFilter::streamEndLuminosityBlockSummary(edm::StreamID iID,
                                                              edm::LuminosityBlock const& iLuminosityBlock,
                                                              edm::EventSetup const&,
                                                              GenLumiInfoProduct* iProduct) const {
  iProduct->mergeProduct(*streamCache(iID)->endLumiProduce(iLuminosityBlock.run()));

  if (lastLumiIndex_ == iLuminosityBlock.index()) {
    externalgen::StreamCache* expected = nullptr;

    availableForBeginLumi_.compare_exchange_strong(expected, streamCache(iID));
  }
}

void ExternalGeneratorFilter::globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLuminosityBlock,
                                                              edm::EventSetup const&,
                                                              GenLumiInfoProduct const* iProduct) const {
  iLuminosityBlock.emplace(lumiInfoToken_, *iProduct);
}

DEFINE_FWK_MODULE(ExternalGeneratorFilter);
