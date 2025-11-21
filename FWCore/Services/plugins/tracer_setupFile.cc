#include "tracer_setupFile.h"
#include "monitor_file_utilities.h"

#include <chrono>
#include <numeric>

#include <sstream>
#include <type_traits>
#include <cassert>
#include <typeindex>

#include "FWCore/Concurrency/interface/ThreadSafeOutputFileStream.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/AbstractServices/interface/TimingServiceBase.h"

using namespace edm::service::monitor_file_utilities;

namespace {
  using duration_t = std::chrono::microseconds;
  using steady_clock = std::chrono::steady_clock;

  enum class Step : char {
    preSourceTransition = 'S',
    postSourceTransition = 's',
    preModulePrefetching = 'P',
    postModulePrefetching = 'p',
    preModuleEventAcquire = 'A',
    postModuleEventAcquire = 'a',
    preModuleTransition = 'M',
    preEventReadFromSource = 'R',
    postEventReadFromSource = 'r',
    preModuleEventDelayedGet = 'D',
    postModuleEventDelayedGet = 'd',
    postModuleTransition = 'm',
    preESModulePrefetching = 'Q',
    postESModulePrefetching = 'q',
    preESModule = 'N',
    postESModule = 'n',
    preESModuleAcquire = 'B',
    postESModuleAcquire = 'b',
    preFrameworkTransition = 'F',
    postFrameworkTransition = 'f'
  };

  enum class Phase : short {
    destruction = -16,
    endJob = -12,
    endStream = -11,
    writeProcessBlock = -10,
    endProcessBlock = -9,
    globalWriteRun = -7,
    globalEndRun = -6,
    streamEndRun = -5,
    globalWriteLumi = -4,
    globalEndLumi = -3,
    streamEndLumi = -2,
    clearEvent = -1,
    Event = 0,
    streamBeginLumi = 2,
    globalBeginLumi = 3,
    streamBeginRun = 5,
    globalBeginRun = 6,
    accessInputProcessBlock = 8,
    beginProcessBlock = 9,
    openFile = 10,
    beginStream = 11,
    beginJob = 12,
    esSync = 13,
    esSyncEnqueue = 14,
    getNextTransition = 15,
    construction = 16,
    finalizeEDModules = 17,
    finalizeEventSetupConfiguration = 18,
    scheduleConsistencyCheck = 19,
    createRunLumiEvents = 20,
    finishSchedule = 21,
    constructESModules = 22,
    startServices = 23,
    processPython = 24
  };

  std::ostream& operator<<(std::ostream& os, Step const s) {
    os << static_cast<std::underlying_type_t<Step>>(s);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, Phase const s) {
    os << static_cast<std::underlying_type_t<Phase>>(s);
    return os;
  }

  template <Step S, typename... ARGS>
  std::string assembleMessage(ARGS const... args) {
    std::ostringstream oss;
    oss << S;
    concatenate(oss, args...);
    oss << '\n';
    return oss.str();
  }

  Phase toTransitionImpl(edm::StreamContext const& iContext) {
    using namespace edm;
    switch (iContext.transition()) {
      case StreamContext::Transition::kBeginStream:
        return Phase::beginStream;
      case StreamContext::Transition::kBeginRun:
        return Phase::streamBeginRun;
      case StreamContext::Transition::kBeginLuminosityBlock:
        return Phase::streamBeginLumi;
      case StreamContext::Transition::kEvent:
        return Phase::Event;
      case StreamContext::Transition::kEndLuminosityBlock:
        return Phase::streamEndLumi;
      case StreamContext::Transition::kEndRun:
        return Phase::streamEndRun;
      case StreamContext::Transition::kEndStream:
        return Phase::endStream;
      default:
        break;
    }
    assert(false);
    return Phase::Event;
  }

  auto toTransition(edm::StreamContext const& iContext) -> std::underlying_type_t<Phase> {
    return static_cast<std::underlying_type_t<Phase>>(toTransitionImpl(iContext));
  }

  Phase toTransitionImpl(edm::GlobalContext const& iContext) {
    using namespace edm;
    switch (iContext.transition()) {
      case GlobalContext::Transition::kBeginProcessBlock:
        return Phase::beginProcessBlock;
      case GlobalContext::Transition::kAccessInputProcessBlock:
        return Phase::accessInputProcessBlock;
      case GlobalContext::Transition::kBeginRun:
        return Phase::globalBeginRun;
      case GlobalContext::Transition::kBeginLuminosityBlock:
        return Phase::globalBeginLumi;
      case GlobalContext::Transition::kEndLuminosityBlock:
        return Phase::globalEndLumi;
      case GlobalContext::Transition::kWriteLuminosityBlock:
        return Phase::globalWriteLumi;
      case GlobalContext::Transition::kEndRun:
        return Phase::globalEndRun;
      case GlobalContext::Transition::kWriteRun:
        return Phase::globalWriteRun;
      case GlobalContext::Transition::kEndProcessBlock:
        return Phase::endProcessBlock;
      case GlobalContext::Transition::kWriteProcessBlock:
        return Phase::writeProcessBlock;
      default:
        break;
    }
    assert(false);
    return Phase::Event;
  }
  auto toTransition(edm::GlobalContext const& iContext) -> std::underlying_type_t<Phase> {
    return static_cast<std::underlying_type_t<Phase>>(toTransitionImpl(iContext));
  }

  unsigned int toTransitionIndex(edm::GlobalContext const& iContext) {
    if (iContext.transition() == edm::GlobalContext::Transition::kBeginProcessBlock or
        iContext.transition() == edm::GlobalContext::Transition::kEndProcessBlock or
        iContext.transition() == edm::GlobalContext::Transition::kWriteProcessBlock or
        iContext.transition() == edm::GlobalContext::Transition::kAccessInputProcessBlock) {
      return 0;
    }
    if (iContext.transition() == edm::GlobalContext::Transition::kBeginRun or
        iContext.transition() == edm::GlobalContext::Transition::kEndRun or
        iContext.transition() == edm::GlobalContext::Transition::kWriteRun) {
      return iContext.runIndex();
    }
    return iContext.luminosityBlockIndex();
  }

  template <Step S>
  struct ESModuleState {
    ESModuleState(std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile,
                  decltype(steady_clock::now()) beginTime,
                  std::shared_ptr<std::vector<std::type_index>> recordIndices)
        : logFile_{logFile}, recordIndices_{recordIndices}, beginTime_{beginTime} {}

    void operator()(edm::eventsetup::EventSetupRecordKey const& iKey,
                    edm::ESModuleCallingContext const& iContext) const {
      using namespace edm;
      auto const t = std::chrono::duration_cast<duration_t>(steady_clock::now() - beginTime_).count();
      auto top = iContext.getTopModuleCallingContext();
      short int phase = 0;
      unsigned long phaseID = 0xFFFFFFFF;
      if (top->type() == ParentContext::Type::kGlobal) {
        auto global = top->globalContext();
        phase = toTransition(*global);
        phaseID = toTransitionIndex(*global);
      } else if (top->type() == ParentContext::Type::kStream) {
        auto stream = top->getStreamContext();
        phase = toTransition(*stream);
        phaseID = stream_id(*stream);
      } else if (top->type() == ParentContext::Type::kPlaceInPath) {
        auto stream = top->getStreamContext();
        phase = toTransition(*stream);
        phaseID = stream_id(*stream);
      }
      auto recordIndex = findRecordIndices(iKey);
      long long requestingModuleID;
      decltype(iContext.moduleCallingContext()->callID()) requestingCallID;
      if (iContext.type() == ESParentContext::Type::kModule) {
        requestingModuleID = iContext.moduleCallingContext()->moduleDescription()->id();
        requestingCallID = iContext.moduleCallingContext()->callID();
      } else {
        requestingModuleID =
            -1 * static_cast<long long>(iContext.esmoduleCallingContext()->componentDescription()->id_ + 1);
        requestingCallID = iContext.esmoduleCallingContext()->callID();
      }
      auto msg = assembleMessage<S>(phase,
                                    phaseID,
                                    iContext.componentDescription()->id_ + 1,
                                    recordIndex,
                                    iContext.callID(),
                                    requestingModuleID,
                                    requestingCallID,
                                    t);
      logFile_->write(std::move(msg));
    }

  private:
    int findRecordIndices(edm::eventsetup::EventSetupRecordKey const& iKey) const {
      auto index = std::type_index(iKey.type().value());
      auto itFind = std::find(recordIndices_->begin(), recordIndices_->end(), index);
      return itFind - recordIndices_->begin();
    }

    std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile_;
    std::shared_ptr<std::vector<std::type_index>> recordIndices_;
    decltype(steady_clock::now()) beginTime_;
  };

  template <Step S>
  struct GlobalEDModuleState {
    GlobalEDModuleState(std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile,
                        decltype(steady_clock::now()) beginTime)
        : logFile_{logFile}, beginTime_{beginTime} {}

    void operator()(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
      using namespace edm;
      auto const t = std::chrono::duration_cast<duration_t>(steady_clock::now() - beginTime_).count();
      long requestingModuleID = 0;
      decltype(mcc.parent().moduleCallingContext()->callID()) requestingCallID = 0;
      if (mcc.type() == ParentContext::Type::kModule) {
        requestingModuleID = module_id(*mcc.parent().moduleCallingContext());
        requestingCallID = mcc.parent().moduleCallingContext()->callID();
      }
      auto msg = assembleMessage<S>(toTransition(gc),
                                    toTransitionIndex(gc),
                                    module_id(mcc),
                                    module_callid(mcc),
                                    requestingModuleID,
                                    requestingCallID,
                                    t);
      logFile_->write(std::move(msg));
    }

  private:
    std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile_;
    decltype(steady_clock::now()) beginTime_;
  };

  template <Step S>
  struct StreamEDModuleState {
    StreamEDModuleState(std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile,
                        decltype(steady_clock::now()) beginTime)
        : logFile_{logFile}, beginTime_{beginTime} {}

    void operator()(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
      using namespace edm;
      auto const t = std::chrono::duration_cast<duration_t>(steady_clock::now() - beginTime_).count();
      long requestingModuleID = 0;
      decltype(mcc.parent().moduleCallingContext()->callID()) requestingCallID = 0;
      if (mcc.type() == ParentContext::Type::kModule) {
        requestingModuleID = module_id(*mcc.parent().moduleCallingContext());
        requestingCallID = mcc.parent().moduleCallingContext()->callID();
      }
      auto msg = assembleMessage<S>(
          toTransition(sc), stream_id(sc), module_id(mcc), module_callid(mcc), requestingModuleID, requestingCallID, t);
      logFile_->write(std::move(msg));
    }

  private:
    std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile_;
    decltype(steady_clock::now()) beginTime_;
  };

  struct ModuleCtrDtr {
    long long beginConstruction = 0;
    long long endConstruction = 0;
    long long beginDestruction = 0;
    long long endDestruction = 0;
  };

  struct StartupTimes {
    using time_diff =
        decltype(std::chrono::duration_cast<duration_t>(steady_clock::now() - steady_clock::now()).count());
    time_diff endServiceConstruction = 0;
    time_diff beginFinalizeEDModules = 0;
    time_diff endFinalizeEDModules = 0;
    time_diff beginFinalizeEventSetupConfiguration = 0;
    time_diff endFinalizeEventSetupConfiguration = 0;
    time_diff beginScheduleConsistencyCheck = 0;
    time_diff endScheduleConsistencyCheck = 0;
    time_diff beginCreateRunLumiEvents = 0;
    time_diff endCreateRunLumiEvents = 0;
    time_diff beginFinishSchedule = 0;
    time_diff endFinishSchedule = 0;
    time_diff beginConstructESModules = 0;
    time_diff endConstructESModules = 0;
  };
}  // namespace

namespace edm::service::tracer {
  void setupFile(std::string const& iFileName, edm::ActivityRegistry& iRegistry) {
    using namespace std::chrono;

    if (iFileName.empty()) {
      return;
    }

    auto logFile = std::make_shared<edm::ThreadSafeOutputFileStream>(iFileName);

    const auto beginTime = TimingServiceBase::jobStartTime();

    auto esModuleLabelsPtr = std::make_shared<std::vector<std::string>>();
    auto& esModuleLabels = *esModuleLabelsPtr;
    auto esmoduleCtrDtrPtr = std::make_shared<std::vector<ModuleCtrDtr>>();
    auto& esmoduleCtrDtr = *esmoduleCtrDtrPtr;

    iRegistry.watchPreESModuleConstruction([&esmoduleCtrDtr, &esModuleLabels, beginTime](auto const& md) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();

      auto const mid = md.id_ + 1;  //NOTE: we want the id to start at 1 not 0
      if (mid < esmoduleCtrDtr.size()) {
        esmoduleCtrDtr[mid].beginConstruction = t;
      } else {
        esmoduleCtrDtr.resize(mid + 1);
        esmoduleCtrDtr.back().beginConstruction = t;
      }
      auto const* label = md.label_.empty() ? (&md.type_) : (&md.label_);
      if (mid < esModuleLabels.size()) {
        esModuleLabels[mid] = *label;
      } else {
        esModuleLabels.resize(mid + 1);
        esModuleLabels.back() = *label;
      }
    });
    iRegistry.watchPostESModuleConstruction([&esmoduleCtrDtr, beginTime](auto const& md) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      esmoduleCtrDtr[md.id_ + 1].endConstruction = t;
    });

    //acquire names for all the ED and ES modules
    auto moduleCtrDtrPtr = std::make_shared<std::vector<ModuleCtrDtr>>();
    auto& moduleCtrDtr = *moduleCtrDtrPtr;
    auto moduleLabelsPtr = std::make_shared<std::vector<std::string>>();
    auto& moduleLabels = *moduleLabelsPtr;
    iRegistry.watchPreModuleConstruction([&moduleLabels, &moduleCtrDtr, beginTime](ModuleDescription const& md) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();

      auto const mid = md.id();
      if (mid < moduleLabels.size()) {
        moduleLabels[mid] = md.moduleLabel();
        moduleCtrDtr[mid].beginConstruction = t;
      } else {
        moduleLabels.resize(mid + 1);
        moduleLabels.back() = md.moduleLabel();
        moduleCtrDtr.resize(mid + 1);
        moduleCtrDtr.back().beginConstruction = t;
      }
    });
    iRegistry.watchPostModuleConstruction([&moduleCtrDtr, beginTime](auto const& md) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      moduleCtrDtr[md.id()].endConstruction = t;
    });

    iRegistry.watchPreModuleDestruction([&moduleCtrDtr, beginTime](auto const& md) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      moduleCtrDtr[md.id()].beginDestruction = t;
    });
    iRegistry.watchPostModuleDestruction([&moduleCtrDtr, beginTime](auto const& md) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      moduleCtrDtr[md.id()].endDestruction = t;
    });

    auto sourceCtrPtr = std::make_shared<ModuleCtrDtr>();
    auto& sourceCtr = *sourceCtrPtr;
    iRegistry.watchPreSourceConstruction([&sourceCtr, beginTime](auto const&) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      sourceCtr.beginConstruction = t;
    });
    iRegistry.watchPostSourceConstruction([&sourceCtr, beginTime](auto const&) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      sourceCtr.endConstruction = t;
    });

    auto recordIndices = std::make_shared<std::vector<std::type_index>>();
    iRegistry.watchEventSetupConfiguration(
        [logFile, recordIndices](auto const& recordsToResolvers, auto const&) mutable {
          std::ostringstream oss;

          auto recordKeys = recordsToResolvers.recordKeys();
          std::sort(recordKeys.begin(), recordKeys.end());
          std::vector<std::string> recordNames;
          //want id to start at 1 not 0
          recordNames.reserve(recordKeys.size() + 1);
          recordNames.emplace_back("");
          recordIndices->reserve(recordKeys.size() + 1);
          recordIndices->push_back(std::type_index(typeid(void)));
          for (auto const& r : recordKeys) {
            recordNames.push_back(r.name());
            recordIndices->push_back(std::type_index(r.type().value()));
          }

          moduleIdToLabel(oss, recordNames, 'R', "Record ID", "Record name");
          logFile->write(oss.str());
        });

    auto const pythonBegin = duration_cast<duration_t>(TimingServiceBase::pythonStartTime() - beginTime).count();
    auto const pythonEnd = duration_cast<duration_t>(TimingServiceBase::pythonEndTime() - beginTime).count();
    auto const servicesBegin = duration_cast<duration_t>(TimingServiceBase::servicesStartTime() - beginTime).count();

    auto startupTimes = std::make_shared<StartupTimes>();
    auto ptrStartupTimes = startupTimes.get();
    iRegistry.watchPostServicesConstruction([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->endServiceConstruction = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPreEventSetupModulesConstruction([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->beginConstructESModules = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPostEventSetupModulesConstruction([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->endConstructESModules = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPreModulesInitializationFinalized([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->beginFinalizeEDModules = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPostModulesInitializationFinalized([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->endFinalizeEDModules = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPreEventSetupConfigurationFinalized([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->beginFinalizeEventSetupConfiguration =
          duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPostEventSetupConfigurationFinalized([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->endFinalizeEventSetupConfiguration =
          duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPreScheduleConsistencyCheck([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->beginScheduleConsistencyCheck =
          duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPostScheduleConsistencyCheck([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->endScheduleConsistencyCheck = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPrePrincipalsCreation([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->beginCreateRunLumiEvents = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPostPrincipalsCreation([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->endCreateRunLumiEvents = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPreFinishSchedule([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->beginFinishSchedule = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPostFinishSchedule([ptrStartupTimes, beginTime]() mutable {
      ptrStartupTimes->endFinishSchedule = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
    });
    iRegistry.watchPreBeginJob([logFile,
                                moduleLabelsPtr,
                                esModuleLabelsPtr,
                                moduleCtrDtrPtr,
                                esmoduleCtrDtrPtr,
                                sourceCtrPtr,
                                beginTime,
                                pythonBegin,
                                pythonEnd,
                                servicesBegin,
                                startupTimes](auto&) mutable {
      {
        std::ostringstream oss;
        moduleIdToLabel(oss, *moduleLabelsPtr, 'M', "EDModule ID", "Module label");
        logFile->write(oss.str());
        moduleLabelsPtr.reset();
      }
      {
        std::ostringstream oss;
        moduleIdToLabel(oss, *esModuleLabelsPtr, 'N', "ESModule ID", "ESModule label");
        logFile->write(oss.str());
        esModuleLabelsPtr.reset();
      }
      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::processPython), 0, 0, 0, 0, pythonBegin);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::processPython), 0, 0, 0, 0, pythonEnd);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::startServices), 0, 0, 0, 0, servicesBegin);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::startServices),
            0,
            0,
            0,
            0,
            startupTimes->endServiceConstruction);
        logFile->write(std::move(msg));
      }

      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::constructESModules),
            0,
            0,
            0,
            0,
            startupTimes->beginConstructESModules);
        logFile->write(std::move(msg));
      }
      {
        std::vector<int> indices(esmoduleCtrDtrPtr->size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&esmoduleCtrDtrPtr](int l, int r) {
          return (*esmoduleCtrDtrPtr)[l].beginConstruction < (*esmoduleCtrDtrPtr)[r].beginConstruction;
        });
        for (auto index : indices) {
          auto const& ctr = (*esmoduleCtrDtrPtr)[index];
          if (ctr.beginConstruction != 0 and ctr.endConstruction != 0) {
            auto msg = assembleMessage<Step::preESModule>(
                static_cast<std::underlying_type_t<Phase>>(Phase::constructESModules),
                0,
                index,
                0,
                0,
                0,
                0,  //signifies no calling module
                ctr.beginConstruction);
            logFile->write(std::move(msg));
            auto emsg = assembleMessage<Step::postESModule>(
                static_cast<std::underlying_type_t<Phase>>(Phase::constructESModules),
                0,
                index,
                0,
                0,
                0,
                0,
                ctr.endConstruction);
            logFile->write(std::move(emsg));
          }
        }
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::constructESModules),
            0,
            0,
            0,
            0,
            startupTimes->endConstructESModules);
        logFile->write(std::move(msg));
      }

      //NOTE: the source construction can run concurently with module construction so we need to properly
      // interleave its timing in with the modules
      auto srcBeginConstruction = sourceCtrPtr->beginConstruction;
      auto srcEndConstruction = sourceCtrPtr->endConstruction;
      sourceCtrPtr.reset();
      auto handleSource = [&srcBeginConstruction, &srcEndConstruction, &logFile](long long iTime) mutable {
        if (srcBeginConstruction != 0 and srcBeginConstruction < iTime) {
          auto bmsg = assembleMessage<Step::preSourceTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::construction), 0, srcBeginConstruction);
          logFile->write(std::move(bmsg));
          srcBeginConstruction = 0;
        }
        if (srcEndConstruction != 0 and srcEndConstruction < iTime) {
          auto bmsg = assembleMessage<Step::postSourceTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::construction), 0, srcEndConstruction);
          logFile->write(std::move(bmsg));
          srcEndConstruction = 0;
        }
      };
      {
        std::vector<int> indices(moduleCtrDtrPtr->size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&moduleCtrDtrPtr](int l, int r) {
          return (*moduleCtrDtrPtr)[l].beginConstruction < (*moduleCtrDtrPtr)[r].beginConstruction;
        });
        for (auto id : indices) {
          auto const& ctr = (*moduleCtrDtrPtr)[id];
          if (ctr.beginConstruction != 0) {
            handleSource(ctr.beginConstruction);
            auto bmsg = assembleMessage<Step::preModuleTransition>(
                static_cast<std::underlying_type_t<Phase>>(Phase::construction), 0, id, 0, 0, 0, ctr.beginConstruction);
            logFile->write(std::move(bmsg));
            handleSource(ctr.endConstruction);
            auto emsg = assembleMessage<Step::postModuleTransition>(
                static_cast<std::underlying_type_t<Phase>>(Phase::construction), 0, id, 0, 0, 0, ctr.endConstruction);
            logFile->write(std::move(emsg));
          }
          ++id;
        }
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&moduleCtrDtrPtr](int l, int r) {
          return (*moduleCtrDtrPtr)[l].beginDestruction < (*moduleCtrDtrPtr)[r].beginDestruction;
        });
        for (auto id : indices) {
          auto const& dtr = (*moduleCtrDtrPtr)[id];
          if (dtr.beginDestruction != 0) {
            handleSource(dtr.beginDestruction);
            auto bmsg = assembleMessage<Step::preModuleTransition>(
                static_cast<std::underlying_type_t<Phase>>(Phase::destruction), 0, id, 0, 0, 0, dtr.beginDestruction);
            logFile->write(std::move(bmsg));
            handleSource(dtr.endDestruction);
            auto emsg = assembleMessage<Step::postModuleTransition>(
                static_cast<std::underlying_type_t<Phase>>(Phase::destruction), 0, id, 0, 0, 0, dtr.endDestruction);
            logFile->write(std::move(emsg));
          }
          ++id;
        }
        moduleCtrDtrPtr.reset();
      }
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      handleSource(t);

      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::finishSchedule),
            0,
            0,
            0,
            0,
            startupTimes->beginFinishSchedule);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::finishSchedule),
            0,
            0,
            0,
            0,
            startupTimes->endFinishSchedule);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::createRunLumiEvents),
            0,
            0,
            0,
            0,
            startupTimes->beginCreateRunLumiEvents);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::createRunLumiEvents),
            0,
            0,
            0,
            0,
            startupTimes->endCreateRunLumiEvents);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::scheduleConsistencyCheck),
            0,
            0,
            0,
            0,
            startupTimes->beginScheduleConsistencyCheck);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::scheduleConsistencyCheck),
            0,
            0,
            0,
            0,
            startupTimes->endScheduleConsistencyCheck);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::finalizeEventSetupConfiguration),
            0,
            0,
            0,
            0,
            startupTimes->beginFinalizeEventSetupConfiguration);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::finalizeEventSetupConfiguration),
            0,
            0,
            0,
            0,
            startupTimes->endFinalizeEventSetupConfiguration);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::finalizeEDModules),
            0,
            0,
            0,
            0,
            startupTimes->beginFinalizeEDModules);
        logFile->write(std::move(msg));
      }
      {
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::finalizeEDModules),
            0,
            0,
            0,
            0,
            startupTimes->endFinalizeEDModules);
        logFile->write(std::move(msg));
      }

      auto msg = assembleMessage<Step::preFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostBeginJob([logFile, beginTime]() {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::postFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });

    iRegistry.watchPreEndJob([logFile, beginTime]() {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::preFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostEndJob([logFile, beginTime]() {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::postFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });

    iRegistry.watchPreBeginStream([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::preFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::beginStream), stream_id(sc), 0, 0, 0, t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostBeginStream([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::postFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::beginStream), stream_id(sc), 0, 0, 0, t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPreEndStream([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::preFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::endStream), stream_id(sc), 0, 0, 0, t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostEndStream([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::postFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::endStream), stream_id(sc), 0, 0, 0, t);
      logFile->write(std::move(msg));
    });

    iRegistry.watchPreEvent([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg = assembleMessage<Step::preFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::Event),
                                                               stream_id(sc),
                                                               sc.eventID().run(),
                                                               sc.eventID().luminosityBlock(),
                                                               sc.eventID().event(),
                                                               t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostEvent([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg =
          assembleMessage<Step::postFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::Event),
                                                         stream_id(sc),
                                                         sc.eventID().run(),
                                                         sc.eventID().luminosityBlock(),
                                                         sc.eventID().event(),
                                                         t);
      logFile->write(std::move(msg));
    });

    iRegistry.watchPreClearEvent([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg =
          assembleMessage<Step::preFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::clearEvent),
                                                        stream_id(sc),
                                                        sc.eventID().run(),
                                                        sc.eventID().luminosityBlock(),
                                                        sc.eventID().event(),
                                                        t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostClearEvent([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
      auto msg =
          assembleMessage<Step::postFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::clearEvent),
                                                         stream_id(sc),
                                                         sc.eventID().run(),
                                                         sc.eventID().luminosityBlock(),
                                                         sc.eventID().event(),
                                                         t);
      logFile->write(std::move(msg));
    });

    {
      auto preGlobal = [logFile, beginTime](GlobalContext const& gc) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preFrameworkTransition>(toTransition(gc),
                                                                 toTransitionIndex(gc),
                                                                 gc.luminosityBlockID().run(),
                                                                 gc.luminosityBlockID().luminosityBlock(),
                                                                 0,
                                                                 t);
        logFile->write(std::move(msg));
      };
      iRegistry.watchPreBeginProcessBlock(preGlobal);
      iRegistry.watchPreEndProcessBlock(preGlobal);
      iRegistry.watchPreWriteProcessBlock(preGlobal);
      iRegistry.watchPreAccessInputProcessBlock(preGlobal);
      iRegistry.watchPreGlobalBeginRun(preGlobal);
      iRegistry.watchPreGlobalBeginLumi(preGlobal);
      iRegistry.watchPreGlobalEndLumi(preGlobal);
      iRegistry.watchPreGlobalWriteLumi(preGlobal);
      iRegistry.watchPreGlobalEndRun(preGlobal);
      iRegistry.watchPreGlobalWriteRun(preGlobal);
    }
    {
      auto postGlobal = [logFile, beginTime](GlobalContext const& gc) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postFrameworkTransition>(toTransition(gc),
                                                                  toTransitionIndex(gc),
                                                                  gc.luminosityBlockID().run(),
                                                                  gc.luminosityBlockID().luminosityBlock(),
                                                                  0,
                                                                  t);
        logFile->write(std::move(msg));
      };
      iRegistry.watchPostBeginProcessBlock(postGlobal);
      iRegistry.watchPostEndProcessBlock(postGlobal);
      iRegistry.watchPostWriteProcessBlock(postGlobal);
      iRegistry.watchPostAccessInputProcessBlock(postGlobal);
      iRegistry.watchPostGlobalBeginRun(postGlobal);
      iRegistry.watchPostGlobalBeginLumi(postGlobal);
      iRegistry.watchPostGlobalEndLumi(postGlobal);
      iRegistry.watchPostGlobalWriteLumi(postGlobal);
      iRegistry.watchPostGlobalEndRun(postGlobal);
      iRegistry.watchPostGlobalWriteRun(postGlobal);
    }
    {
      auto preStream = [logFile, beginTime](StreamContext const& sc) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            toTransition(sc), stream_id(sc), sc.eventID().run(), sc.eventID().luminosityBlock(), 0, t);
        logFile->write(std::move(msg));
      };
      iRegistry.watchPreStreamBeginRun(preStream);
      iRegistry.watchPreStreamBeginLumi(preStream);
      iRegistry.watchPreStreamEndLumi(preStream);
      iRegistry.watchPreStreamEndRun(preStream);
    }
    {
      auto postStream = [logFile, beginTime](StreamContext const& sc) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postFrameworkTransition>(
            toTransition(sc), stream_id(sc), sc.eventID().run(), sc.eventID().luminosityBlock(), 0, t);
        logFile->write(std::move(msg));
      };
      iRegistry.watchPostStreamBeginRun(postStream);
      iRegistry.watchPostStreamBeginLumi(postStream);
      iRegistry.watchPostStreamEndLumi(postStream);
      iRegistry.watchPostStreamEndRun(postStream);
    }
    {
      iRegistry.watchESSyncIOVQueuing([logFile, beginTime](IOVSyncValue const& sv) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::esSyncEnqueue),
            -1,
            sv.eventID().run(),
            sv.eventID().luminosityBlock(),
            0,
            t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPreESSyncIOV([logFile, beginTime](IOVSyncValue const& sv) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg =
            assembleMessage<Step::preFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::esSync),
                                                          -1,
                                                          sv.eventID().run(),
                                                          sv.eventID().luminosityBlock(),
                                                          0,
                                                          t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostESSyncIOV([logFile, beginTime](IOVSyncValue const& sv) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg =
            assembleMessage<Step::postFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::esSync),
                                                           -1,
                                                           sv.eventID().run(),
                                                           sv.eventID().luminosityBlock(),
                                                           0,
                                                           t);
        logFile->write(std::move(msg));
      });
    }
    {
      iRegistry.watchPreOpenFile([logFile, beginTime](std::string const&) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::openFile), 0, t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostOpenFile([logFile, beginTime](std::string const&) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::openFile), 0, t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPreSourceEvent([logFile, beginTime](StreamID id) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::Event), id.value(), t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostSourceEvent([logFile, beginTime](StreamID id) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::Event), id.value(), t);
        logFile->write(std::move(msg));
      });

      iRegistry.watchPreSourceRun([logFile, beginTime](RunIndex id) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginRun), id.value(), t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostSourceRun([logFile, beginTime](RunIndex id) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginRun), id.value(), t);
        logFile->write(std::move(msg));
      });

      iRegistry.watchPreSourceLumi([logFile, beginTime](auto id) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginLumi), id.value(), t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostSourceLumi([logFile, beginTime](auto id) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginLumi), id.value(), t);
        logFile->write(std::move(msg));
      });

      iRegistry.watchPreSourceNextTransition([logFile, beginTime]() {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::getNextTransition), t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostSourceNextTransition([logFile, beginTime]() {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postSourceTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::getNextTransition), t);
        logFile->write(std::move(msg));
      });

      //ED Modules
      iRegistry.watchPreModuleBeginJob([logFile, beginTime](auto const& md) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preModuleTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, md.id(), 0, 0, 0, t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostModuleBeginJob([logFile, beginTime](auto const& md) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postModuleTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, md.id(), 0, 0, 0, t);
        logFile->write(std::move(msg));
      });

      iRegistry.watchPreModuleBeginStream(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleBeginStream(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleEndStream(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleEndStream(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleEndJob([logFile, beginTime](auto const& md) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::preModuleTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, md.id(), 0, 0, 0, t);
        logFile->write(std::move(msg));
      });
      iRegistry.watchPostModuleEndJob([logFile, beginTime](auto const& md) {
        auto const t = duration_cast<duration_t>(steady_clock::now() - beginTime).count();
        auto msg = assembleMessage<Step::postModuleTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, md.id(), 0, 0, 0, t);
        logFile->write(std::move(msg));
      });

      iRegistry.watchPreModuleEventPrefetching(StreamEDModuleState<Step::preModulePrefetching>(logFile, beginTime));
      iRegistry.watchPostModuleEventPrefetching(StreamEDModuleState<Step::postModulePrefetching>(logFile, beginTime));

      iRegistry.watchPreModuleEvent(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleEvent(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleEventAcquire(StreamEDModuleState<Step::preModuleEventAcquire>(logFile, beginTime));
      iRegistry.watchPostModuleEventAcquire(StreamEDModuleState<Step::postModuleEventAcquire>(logFile, beginTime));
      iRegistry.watchPreModuleEventDelayedGet(StreamEDModuleState<Step::preModuleEventDelayedGet>(logFile, beginTime));
      iRegistry.watchPostModuleEventDelayedGet(
          StreamEDModuleState<Step::postModuleEventDelayedGet>(logFile, beginTime));
      iRegistry.watchPreEventReadFromSource(StreamEDModuleState<Step::preEventReadFromSource>(logFile, beginTime));
      iRegistry.watchPostEventReadFromSource(StreamEDModuleState<Step::postEventReadFromSource>(logFile, beginTime));

      iRegistry.watchPreModuleTransform(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleTransform(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleTransformPrefetching(StreamEDModuleState<Step::preModulePrefetching>(logFile, beginTime));
      iRegistry.watchPostModuleTransformPrefetching(
          StreamEDModuleState<Step::postModulePrefetching>(logFile, beginTime));
      iRegistry.watchPreModuleTransformAcquiring(StreamEDModuleState<Step::preModuleEventAcquire>(logFile, beginTime));
      iRegistry.watchPostModuleTransformAcquiring(
          StreamEDModuleState<Step::postModuleEventAcquire>(logFile, beginTime));

      iRegistry.watchPreModuleStreamPrefetching(StreamEDModuleState<Step::preModulePrefetching>(logFile, beginTime));
      iRegistry.watchPostModuleStreamPrefetching(StreamEDModuleState<Step::postModulePrefetching>(logFile, beginTime));

      iRegistry.watchPreModuleStreamBeginRun(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleStreamBeginRun(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleStreamEndRun(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleStreamEndRun(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleStreamBeginLumi(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleStreamBeginLumi(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleStreamEndLumi(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleStreamEndLumi(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleBeginProcessBlock(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleBeginProcessBlock(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleAccessInputProcessBlock(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleAccessInputProcessBlock(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleEndProcessBlock(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleEndProcessBlock(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleGlobalPrefetching(GlobalEDModuleState<Step::preModulePrefetching>(logFile, beginTime));
      iRegistry.watchPostModuleGlobalPrefetching(GlobalEDModuleState<Step::postModulePrefetching>(logFile, beginTime));

      iRegistry.watchPreModuleGlobalBeginRun(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleGlobalBeginRun(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleGlobalEndRun(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleGlobalEndRun(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleGlobalBeginLumi(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleGlobalBeginLumi(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));
      iRegistry.watchPreModuleGlobalEndLumi(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleGlobalEndLumi(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleWriteProcessBlock(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleWriteProcessBlock(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleWriteRun(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleWriteRun(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      iRegistry.watchPreModuleWriteLumi(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime));
      iRegistry.watchPostModuleWriteLumi(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime));

      //ES Modules
      iRegistry.watchPreESModulePrefetching(
          ESModuleState<Step::preESModulePrefetching>(logFile, beginTime, recordIndices));
      iRegistry.watchPostESModulePrefetching(
          ESModuleState<Step::postESModulePrefetching>(logFile, beginTime, recordIndices));
      iRegistry.watchPreESModule(ESModuleState<Step::preESModule>(logFile, beginTime, recordIndices));
      iRegistry.watchPostESModule(ESModuleState<Step::postESModule>(logFile, beginTime, recordIndices));
      iRegistry.watchPreESModuleAcquire(ESModuleState<Step::preESModuleAcquire>(logFile, beginTime, recordIndices));
      iRegistry.watchPostESModuleAcquire(ESModuleState<Step::postESModuleAcquire>(logFile, beginTime, recordIndices));
    }

    std::ostringstream oss;
    oss << "# Transition               Symbol\n";
    oss << "#------------------------  ------\n";
    oss << "# processPython            " << Phase::processPython << "\n"
        << "# startServices            " << Phase::startServices << "\n"
        << "# constructionESModules    " << Phase::constructESModules << "\n"
        << "# finishSchedule           " << Phase::finishSchedule << "\n"
        << "# createRunLumiEvents      " << Phase::createRunLumiEvents << "\n"
        << "# scheduleConsistencyCheck " << Phase::scheduleConsistencyCheck << "\n"
        << "# finish ES Configuration  " << Phase::finalizeEventSetupConfiguration << "\n"
        << "# finalize EDModules       " << Phase::finalizeEDModules << "\n"
        << "# construction             " << Phase::construction << "\n"
        << "# getNextTransition        " << Phase::getNextTransition << "\n"
        << "# esSyncEnqueue            " << Phase::esSyncEnqueue << "\n"
        << "# esSync                   " << Phase::esSync << "\n"
        << "# beginJob                 " << Phase::beginJob << "\n"
        << "# beginStream              " << Phase::beginStream << "\n"
        << "# openFile                 " << Phase::openFile << "\n"
        << "# beginProcessBlock        " << Phase::beginProcessBlock << "\n"
        << "# accessInputProcessBlock  " << Phase::accessInputProcessBlock << "\n"
        << "# globalBeginRun           " << Phase::globalBeginRun << "\n"
        << "# streamBeginRun           " << Phase::streamBeginRun << "\n"
        << "# globalBeginLumi          " << Phase::globalBeginLumi << "\n"
        << "# streamBeginLumi          " << Phase::streamBeginLumi << "\n"
        << "# Event                    " << Phase::Event << "\n"
        << "# clearEvent               " << Phase::clearEvent << "\n"
        << "# streamEndLumi            " << Phase::streamEndLumi << "\n"
        << "# globalEndLumi            " << Phase::globalEndLumi << "\n"
        << "# globalWriteLumi          " << Phase::globalWriteLumi << "\n"
        << "# streamEndRun             " << Phase::streamEndRun << "\n"
        << "# globalEndRun             " << Phase::globalEndRun << "\n"
        << "# globalWriteRun           " << Phase::globalWriteRun << "\n"
        << "# endProcessBlock          " << Phase::endProcessBlock << "\n"
        << "# writeProcessBlock        " << Phase::writeProcessBlock << "\n"
        << "# endStream                " << Phase::endStream << "\n"
        << "# endJob                   " << Phase::endJob << "\n"
        << "# destruction              " << Phase::destruction << "\n\n";
    oss << "# Step                       Symbol Entries\n"
        << "# -------------------------- ------ ------------------------------------------\n"
        << "# preSourceTransition           " << Step::preSourceTransition
        << "   <Transition type> <Transition ID> <Time since begin of cmsRun (us)>\n"
        << "# postSourceTransition          " << Step::postSourceTransition
        << "   <Transition type> <Transition ID> <Time since begin of cmsRun (us)>\n"
        << "# preModulePrefetching          " << Step::preModulePrefetching
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID or 0> <Requesting "
           "Call ID> <Time since begin of "
           "cmsRun "
           "(us)>\n"
        << "# postModulePrefetching         " << Step::postModulePrefetching
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID or 0> <Requesting "
           "Call ID> <Time since begin of "
           "cmsRun "
           "(us)>\n"
        << "# preModuleEventAcquire         " << Step::preModuleEventAcquire
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID or 0> <Requesting "
           "Call ID> <Time since begin of "
           "cmsRun "
           "(us)>\n"
        << "# postModuleEventAcquire        " << Step::postModuleEventAcquire
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID or 0> <Requesting "
           "Call ID> <Time since begin of "
           "cmsRun "
           "(us)>\n"
        << "# preModuleTransition           " << Step::preModuleTransition
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID or 0> <Requesting "
           "Call ID> <Time since begin of "
           "cmsRun "
           "(us)>\n"
        << "# preEventReadFromSource        " << Step::preEventReadFromSource
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID or 0> <Requesting "
           "Call ID> <Time since begin of "
           "cmsRun "
           "(us)>\n"
        << "# postEventReadFromSource       " << Step::postEventReadFromSource
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID or 0> <Requesting "
           "Call ID> <Time since begin of "
           "cmsRun "
           "(us)>\n"
        << "# postModuleTransition          " << Step::postModuleTransition
        << "   <Transition type> <Transition ID> <EDModule ID> <Call ID> <Requesting EDModule ID> <Requesting Call ID> "
           "<Time since begin of cmsRun "
           "(us)>\n"
        << "# preESModulePrefetching        " << Step::preESModulePrefetching
        << "   <Transition type> <Transition ID> <ESModule ID> <Record ID> <Call ID> <Requesting module ID (+ED, -ES)> "
           "<Requesting Call ID> "
           "<Time "
           "since of cmsRun (us)>\n"
        << "# postESModulePrefetching       " << Step::postESModulePrefetching
        << "   <Transition type> <Transition ID> <ESModule ID> <Record ID> <Call ID> <Requesting module ID (+ED, -ES)> "
           "<Requesting Call ID> "
           "<Time "
           "since of cmsRun (us)>\n"
        << "# preESModuleTransition         " << Step::preESModule
        << "   <TransitionType> <Transition ID> <ESModule ID> <Record ID> <Call ID> <Requesting module ID (+ED, -ES)> "
           "<Requesting Call ID> "
           "<Time "
           "since of cmsRun (us)>\n"
        << "# postESModuleTransition        " << Step::postESModule
        << "   <TransitionType> <Transition ID> <ESModule ID> <Record ID> <Call ID> <Requesting module ID (+ED, -ES)> "
           "<Requesting Call ID> "
           "<Time "
           "since of cmsRun (us)>\n"
        << "# preFrameworkTransition        " << Step::preFrameworkTransition
        << "   <Transition type> <Transition ID> <Run#> <LumiBlock#> <Event #> <Time since begin of cmsRun (ms)>\n"
        << "# postFrameworkTransition       " << Step::postFrameworkTransition
        << "   <Transition type> <Transition ID> <Run#> <LumiBlock#> <Event #> <Time since begin of cmsRun (ms)>\n";
    logFile->write(oss.str());
    return;
  }
}  // namespace edm::service::tracer
