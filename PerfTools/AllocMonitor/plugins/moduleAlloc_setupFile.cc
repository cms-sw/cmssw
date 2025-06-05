#include "moduleAlloc_setupFile.h"
#include "monitor_file_utilities.h"

#include <chrono>

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
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"

#include "FWCore/AbstractServices/interface/TimingServiceBase.h"

#include "ThreadAllocInfo.h"

using namespace edm::moduleAlloc::monitor_file_utilities;

namespace {
  using duration_t = std::chrono::microseconds;
  using clock_t = std::chrono::steady_clock;
  auto const now = clock_t::now;

  enum class Step : char {
    preSourceTransition = 'S',
    postSourceTransition = 's',
    preModuleEventAcquire = 'A',
    postModuleEventAcquire = 'a',
    preModuleTransition = 'M',
    preEventReadFromSource = 'R',
    postEventReadFromSource = 'r',
    preModuleEventDelayedGet = 'D',
    postModuleEventDelayedGet = 'd',
    postModuleTransition = 'm',
    preESModule = 'N',
    postESModule = 'n',
    preESModuleAcquire = 'B',
    postESModuleAcquire = 'b',
    preFrameworkTransition = 'F',
    postFrameworkTransition = 'f'
  };

  constexpr bool isPostTransition(Step s) {
    switch (s) {
      case Step::postSourceTransition:
      case Step::postModuleEventAcquire:
      case Step::postEventReadFromSource:
      case Step::postModuleEventDelayedGet:
      case Step::postModuleTransition:
      case Step::postESModule:
      case Step::postESModuleAcquire:
      case Step::postFrameworkTransition:
        return true;
      default:
        return false;
    }
    return false;
  }

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
    startTracing = 17
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

  template <Step S, typename... ARGS>
  std::string assembleAllocMessage(edm::service::moduleAlloc::ThreadAllocInfo const& info, ARGS const... args) {
    std::ostringstream oss;
    oss << S;
    concatenate(oss, args...);
    concatenate(oss,
                info.nAllocations_,
                info.nDeallocations_,
                info.presentActual_,
                info.minActual_,
                info.maxActual_,
                info.maxSingleAlloc_);
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

  using namespace edm::service::moduleAlloc;

  template <Step S>
  struct ESModuleState {
    ESModuleState(std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile,
                  decltype(now()) beginTime,
                  std::shared_ptr<std::vector<std::type_index>> recordIndices,
                  Filter const* iFilter)
        : logFile_{logFile}, recordIndices_{recordIndices}, beginTime_{beginTime}, filter_(iFilter) {}

    void operator()(edm::eventsetup::EventSetupRecordKey const& iKey,
                    edm::ESModuleCallingContext const& iContext) const {
      using namespace edm;
      auto const t = std::chrono::duration_cast<duration_t>(now() - beginTime_).count();
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
      if constexpr (isPostTransition(S)) {
        auto info = filter_->stopOnThread(-1 * (iContext.componentDescription()->id_ + 1));
        if (info) {
          auto msg = assembleAllocMessage<S>(
              *info, phase, phaseID, iContext.componentDescription()->id_ + 1, recordIndex, iContext.callID(), t);
          logFile_->write(std::move(msg));
        }
      } else {
        if (filter_->startOnThread(-1 * (iContext.componentDescription()->id_ + 1))) {
          auto msg = assembleMessage<S>(
              phase, phaseID, iContext.componentDescription()->id_ + 1, recordIndex, iContext.callID(), t);
          logFile_->write(std::move(msg));
        }
      }
    }

  private:
    int findRecordIndices(edm::eventsetup::EventSetupRecordKey const& iKey) const {
      auto index = std::type_index(iKey.type().value());
      auto itFind = std::find(recordIndices_->begin(), recordIndices_->end(), index);
      return itFind - recordIndices_->begin();
    }

    std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile_;
    std::shared_ptr<std::vector<std::type_index>> recordIndices_;
    decltype(now()) beginTime_;
    Filter const* filter_;
  };

  template <Step S>
  struct GlobalEDModuleState {
    GlobalEDModuleState(std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile,
                        decltype(now()) beginTime,
                        Filter const* iFilter)
        : logFile_{logFile}, beginTime_{beginTime}, filter_(iFilter) {}

    void operator()(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
      using namespace edm;
      auto const t = std::chrono::duration_cast<duration_t>(now() - beginTime_).count();
      if constexpr (isPostTransition(S)) {
        auto id = module_id(mcc);
        auto info = filter_->stopOnThread(id);
        if (info) {
          auto msg = assembleAllocMessage<S>(*info, toTransition(gc), toTransitionIndex(gc), id, module_callid(mcc), t);
          logFile_->write(std::move(msg));
        }
      } else {
        auto id = module_id(mcc);
        if (filter_->startOnThread(id)) {
          auto msg = assembleMessage<S>(toTransition(gc), toTransitionIndex(gc), module_id(mcc), module_callid(mcc), t);
          logFile_->write(std::move(msg));
        }
      }
    }

  private:
    std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile_;
    decltype(now()) beginTime_;
    Filter const* filter_;
  };

  template <Step S>
  struct StreamEDModuleState {
    StreamEDModuleState(std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile,
                        decltype(now()) beginTime,
                        Filter const* iFilter)
        : logFile_{logFile}, beginTime_{beginTime}, filter_(iFilter) {}

    void operator()(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
      using namespace edm;
      auto const t = std::chrono::duration_cast<duration_t>(now() - beginTime_).count();
      if constexpr (isPostTransition(S)) {
        auto id = module_id(mcc);
        auto info = filter_->stopOnThread(id);
        if (info) {
          auto msg =
              assembleAllocMessage<S>(*info, toTransition(sc), stream_id(sc), module_id(mcc), module_callid(mcc), t);
          logFile_->write(std::move(msg));
        }
      } else {
        auto id = module_id(mcc);
        if (filter_->startOnThread(id)) {
          auto msg = assembleMessage<S>(toTransition(sc), stream_id(sc), id, module_callid(mcc), t);
          logFile_->write(std::move(msg));
        }
      }
    }

  private:
    std::shared_ptr<edm::ThreadSafeOutputFileStream> logFile_;
    decltype(now()) beginTime_;
    Filter const* filter_;
  };

  struct ModuleCtrDtr {
    long long beginConstruction = 0;
    long long endConstruction = 0;
    edm::service::moduleAlloc::ThreadAllocInfo constructionAllocInfo;
    long long beginDestruction = 0;
    long long endDestruction = 0;
    edm::service::moduleAlloc::ThreadAllocInfo destructionAllocInfo;
  };
}  // namespace

namespace edm::service::moduleAlloc {
  void setupFile(std::string const& iFileName, edm::ActivityRegistry& iRegistry, Filter const* iFilter) {
    auto beginModuleAlloc = now();
    using namespace std::chrono;

    if (iFileName.empty()) {
      return;
    }

    auto logFile = std::make_shared<edm::ThreadSafeOutputFileStream>(iFileName);

    auto beginTime = TimingServiceBase::jobStartTime();

    auto esModuleLabelsPtr = std::make_shared<std::vector<std::string>>();
    auto& esModuleLabels = *esModuleLabelsPtr;
    //acquire names for all the ED and ES modules
    iRegistry.watchPostESModuleRegistration([&esModuleLabels](auto const& iDescription) {
      if (esModuleLabels.size() <= iDescription.id_ + 1) {
        esModuleLabels.resize(iDescription.id_ + 2);
      }
      //NOTE: we want the id to start at 1 not 0
      if (not iDescription.label_.empty()) {
        esModuleLabels[iDescription.id_ + 1] = iDescription.label_;
      } else {
        esModuleLabels[iDescription.id_ + 1] = iDescription.type_;
      }
    });
    auto moduleCtrDtrPtr = std::make_shared<std::vector<ModuleCtrDtr>>();
    auto& moduleCtrDtr = *moduleCtrDtrPtr;
    auto moduleLabelsPtr = std::make_shared<std::vector<std::string>>();
    auto& moduleLabels = *moduleLabelsPtr;
    iRegistry.watchPreModuleConstruction(
        [&moduleLabels, &moduleCtrDtr, beginTime, iFilter](ModuleDescription const& md) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();

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
          iFilter->startOnThread(mid);
        });
    iRegistry.watchPostModuleConstruction([&moduleCtrDtr, beginTime, iFilter](auto const& md) {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      moduleCtrDtr[md.id()].endConstruction = t;
      auto alloc = iFilter->stopOnThread(md.id());
      if (alloc) {
        moduleCtrDtr[md.id()].constructionAllocInfo = *alloc;
      }
    });

    auto addDataInDtr = std::make_shared<bool>(false);
    iRegistry.watchPreModuleDestruction([&moduleCtrDtr, beginTime, iFilter, addDataInDtr, logFile](auto const& md) {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      if (*addDataInDtr) {
        if (iFilter->keepModuleInfo(md.id())) {
          auto bmsg = assembleMessage<Step::preModuleTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::destruction), 0, md.id(), 0, 0, 0, t);
          logFile->write(std::move(bmsg));
        }
      } else {
        moduleCtrDtr[md.id()].beginDestruction = t;
      }
      iFilter->startOnThread(md.id());
    });
    iRegistry.watchPostModuleDestruction([&moduleCtrDtr, beginTime, iFilter, addDataInDtr, logFile](auto const& md) {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      if (not *addDataInDtr) {
        moduleCtrDtr[md.id()].endDestruction = t;
      }
      auto info = iFilter->stopOnThread(md.id());
      if (info) {
        if (*addDataInDtr) {
          if (iFilter->keepModuleInfo(md.id())) {
            auto emsg = assembleAllocMessage<Step::postModuleTransition>(
                *info, static_cast<std::underlying_type_t<Phase>>(Phase::destruction), 0, md.id(), 0, 0, 0, t);
            logFile->write(std::move(emsg));
          }

        } else {
          moduleCtrDtr[md.id()].destructionAllocInfo = *info;
        }
      }
    });

    auto sourceCtrPtr = std::make_shared<ModuleCtrDtr>();
    auto& sourceCtr = *sourceCtrPtr;
    iRegistry.watchPreSourceConstruction([&sourceCtr, beginTime, iFilter](auto const&) {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      sourceCtr.beginConstruction = t;
      iFilter->startOnThread();
    });
    iRegistry.watchPostSourceConstruction([&sourceCtr, beginTime, iFilter](auto const&) {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      sourceCtr.endConstruction = t;
      auto info = iFilter->stopOnThread();
      if (info) {
        sourceCtr.constructionAllocInfo = *info;
      }
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

    iRegistry.watchPreBeginJob([logFile,
                                iFilter,
                                moduleLabelsPtr,
                                esModuleLabelsPtr,
                                moduleCtrDtrPtr,
                                sourceCtrPtr,
                                beginTime,
                                beginModuleAlloc,
                                addDataInDtr](auto&) mutable {
      *addDataInDtr = true;
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
        auto const moduleAllocStart = duration_cast<duration_t>(beginModuleAlloc - beginTime).count();
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::startTracing), 0, 0, 0, 0, moduleAllocStart);
        logFile->write(std::move(msg));
      }
      if (not iFilter->globalKeep()) {
        auto const t = duration_cast<duration_t>(now() - beginTime).count();
        auto msg = assembleMessage<Step::preFrameworkTransition>(
            static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, 0, 0, 0, t);
        logFile->write(std::move(msg));
        return;
      }
      //NOTE: the source construction can run concurently with module construction so we need to properly
      // interleave its timing in with the modules
      auto srcBeginConstruction = sourceCtrPtr->beginConstruction;
      auto srcEndConstruction = sourceCtrPtr->endConstruction;
      auto srcAllocInfo = sourceCtrPtr->constructionAllocInfo;
      sourceCtrPtr.reset();
      auto handleSource =
          [&srcBeginConstruction, &srcEndConstruction, &logFile, &srcAllocInfo](long long iTime) mutable {
            if (srcBeginConstruction != 0 and srcBeginConstruction < iTime) {
              auto bmsg = assembleMessage<Step::preSourceTransition>(
                  static_cast<std::underlying_type_t<Phase>>(Phase::construction), 0, srcBeginConstruction);
              logFile->write(std::move(bmsg));
              srcBeginConstruction = 0;
            }
            if (srcEndConstruction != 0 and srcEndConstruction < iTime) {
              auto bmsg = assembleAllocMessage<Step::postSourceTransition>(
                  srcAllocInfo, static_cast<std::underlying_type_t<Phase>>(Phase::construction), 0, srcEndConstruction);
              logFile->write(std::move(bmsg));
              srcEndConstruction = 0;
            }
          };
      {
        std::sort(moduleCtrDtrPtr->begin(), moduleCtrDtrPtr->end(), [](auto const& l, auto const& r) {
          return l.beginConstruction < r.beginConstruction;
        });
        int id = 0;
        for (auto const& ctr : *moduleCtrDtrPtr) {
          if (ctr.beginConstruction != 0) {
            handleSource(ctr.beginConstruction);
            if (iFilter->keepModuleInfo(id)) {
              auto bmsg = assembleMessage<Step::preModuleTransition>(
                  static_cast<std::underlying_type_t<Phase>>(Phase::construction), 0, id, 0, ctr.beginConstruction);
              logFile->write(std::move(bmsg));
            }
            handleSource(ctr.endConstruction);
            if (iFilter->keepModuleInfo(id)) {
              auto const& allocInfo = ctr.constructionAllocInfo;
              auto emsg = assembleAllocMessage<Step::postModuleTransition>(
                  allocInfo,
                  static_cast<std::underlying_type_t<Phase>>(Phase::construction),
                  0,
                  id,
                  0,
                  ctr.endConstruction);
              logFile->write(std::move(emsg));
            }
          }
          ++id;
        }
        id = 0;
        std::sort(moduleCtrDtrPtr->begin(), moduleCtrDtrPtr->end(), [](auto const& l, auto const& r) {
          return l.beginDestruction < r.beginDestruction;
        });
        for (auto const& dtr : *moduleCtrDtrPtr) {
          if (dtr.beginDestruction != 0) {
            handleSource(dtr.beginDestruction);
            if (iFilter->keepModuleInfo(id)) {
              auto bmsg = assembleMessage<Step::preModuleTransition>(
                  static_cast<std::underlying_type_t<Phase>>(Phase::destruction), 0, id, 0, 0, 0, dtr.beginDestruction);
              logFile->write(std::move(bmsg));
            }
            handleSource(dtr.endDestruction);
            if (iFilter->keepModuleInfo(id)) {
              auto emsg = assembleAllocMessage<Step::postModuleTransition>(
                  dtr.destructionAllocInfo,
                  static_cast<std::underlying_type_t<Phase>>(Phase::destruction),
                  0,
                  id,
                  0,
                  0,
                  0,
                  dtr.endDestruction);
              logFile->write(std::move(emsg));
            }
          }
          ++id;
        }
        moduleCtrDtrPtr.reset();
      }
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      handleSource(t);
      auto msg = assembleMessage<Step::preFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostBeginJob([logFile, beginTime]() {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      auto msg = assembleMessage<Step::postFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });

    iRegistry.watchPreEndJob([logFile, beginTime]() {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      auto msg = assembleMessage<Step::preFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostEndJob([logFile, beginTime]() {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      auto msg = assembleMessage<Step::postFrameworkTransition>(
          static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, 0, 0, 0, t);
      logFile->write(std::move(msg));
    });

    iRegistry.watchPreEvent([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      auto msg = assembleMessage<Step::preFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::Event),
                                                               stream_id(sc),
                                                               sc.eventID().run(),
                                                               sc.eventID().luminosityBlock(),
                                                               sc.eventID().event(),
                                                               t);
      logFile->write(std::move(msg));
    });
    iRegistry.watchPostEvent([logFile, beginTime](auto const& sc) {
      auto const t = duration_cast<duration_t>(now() - beginTime).count();
      auto msg =
          assembleMessage<Step::postFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::Event),
                                                         stream_id(sc),
                                                         sc.eventID().run(),
                                                         sc.eventID().luminosityBlock(),
                                                         sc.eventID().event(),
                                                         t);
      logFile->write(std::move(msg));
    });

    iRegistry.watchPreClearEvent([logFile, beginTime, iFilter](auto const& sc) {
      if (iFilter->startOnThread()) {
        auto const t = duration_cast<duration_t>(now() - beginTime).count();
        auto msg =
            assembleMessage<Step::preFrameworkTransition>(static_cast<std::underlying_type_t<Phase>>(Phase::clearEvent),
                                                          stream_id(sc),
                                                          sc.eventID().run(),
                                                          sc.eventID().luminosityBlock(),
                                                          sc.eventID().event(),
                                                          t);
        logFile->write(std::move(msg));
      }
    });
    iRegistry.watchPostClearEvent([logFile, beginTime, iFilter](auto const& sc) {
      auto info = iFilter->stopOnThread();
      if (info) {
        auto const t = duration_cast<duration_t>(now() - beginTime).count();
        auto msg = assembleAllocMessage<Step::postFrameworkTransition>(
            *info,
            static_cast<std::underlying_type_t<Phase>>(Phase::clearEvent),
            stream_id(sc),
            sc.eventID().run(),
            sc.eventID().luminosityBlock(),
            sc.eventID().event(),
            t);
        logFile->write(std::move(msg));
      }
    });

    {
      auto preGlobal = [logFile, beginTime](GlobalContext const& gc) {
        auto const t = duration_cast<duration_t>(now() - beginTime).count();
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
        auto const t = duration_cast<duration_t>(now() - beginTime).count();
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
        auto const t = duration_cast<duration_t>(now() - beginTime).count();
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
        auto const t = duration_cast<duration_t>(now() - beginTime).count();
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
      iRegistry.watchPreOpenFile([logFile, beginTime, iFilter](std::string const&) {
        if (iFilter->startOnThread()) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleMessage<Step::preSourceTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::openFile), 0, t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPostOpenFile([logFile, beginTime, iFilter](std::string const&) {
        auto info = iFilter->stopOnThread();
        if (info) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleAllocMessage<Step::postSourceTransition>(
              *info, static_cast<std::underlying_type_t<Phase>>(Phase::openFile), 0, t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPreSourceEvent([logFile, beginTime, iFilter](StreamID id) {
        if (iFilter->startOnThread()) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleMessage<Step::preSourceTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::Event), id.value(), t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPostSourceEvent([logFile, beginTime, iFilter](StreamID id) {
        auto info = iFilter->stopOnThread();
        if (info) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleAllocMessage<Step::postSourceTransition>(
              *info, static_cast<std::underlying_type_t<Phase>>(Phase::Event), id.value(), t);
          logFile->write(std::move(msg));
        }
      });

      iRegistry.watchPreSourceRun([logFile, beginTime, iFilter](RunIndex id) {
        if (iFilter->startOnThread()) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleMessage<Step::preSourceTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginRun), id.value(), t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPostSourceRun([logFile, beginTime, iFilter](RunIndex id) {
        auto info = iFilter->stopOnThread();
        if (info) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleAllocMessage<Step::postSourceTransition>(
              *info, static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginRun), id.value(), t);
          logFile->write(std::move(msg));
        }
      });

      iRegistry.watchPreSourceLumi([logFile, beginTime, iFilter](auto id) {
        if (iFilter->startOnThread()) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleMessage<Step::preSourceTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginLumi), id.value(), t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPostSourceLumi([logFile, beginTime, iFilter](auto id) {
        auto info = iFilter->stopOnThread();
        if (info) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleAllocMessage<Step::postSourceTransition>(
              *info, static_cast<std::underlying_type_t<Phase>>(Phase::globalBeginLumi), id.value(), t);
          logFile->write(std::move(msg));
        }
      });

      iRegistry.watchPreSourceNextTransition([logFile, beginTime, iFilter]() {
        if (iFilter->startOnThread()) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleMessage<Step::preSourceTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::getNextTransition), t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPostSourceNextTransition([logFile, beginTime, iFilter]() {
        auto info = iFilter->stopOnThread();
        if (info) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleAllocMessage<Step::postSourceTransition>(
              *info, static_cast<std::underlying_type_t<Phase>>(Phase::getNextTransition), t);
          logFile->write(std::move(msg));
        }
      });

      //ED Modules
      iRegistry.watchPreModuleBeginJob([logFile, beginTime, iFilter](auto const& md) {
        if (iFilter->startOnThread(md.id())) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleMessage<Step::preModuleTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, md.id(), 0, t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPostModuleBeginJob([logFile, beginTime, iFilter](auto const& md) {
        auto info = iFilter->stopOnThread(md.id());
        if (info) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleAllocMessage<Step::postModuleTransition>(
              *info, static_cast<std::underlying_type_t<Phase>>(Phase::beginJob), 0, md.id(), 0, t);
          logFile->write(std::move(msg));
        }
      });

      iRegistry.watchPreModuleBeginStream(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleBeginStream(
          StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleEndStream(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleEndStream(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleEndJob([logFile, beginTime, iFilter](auto const& md) {
        if (iFilter->startOnThread(md.id())) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleMessage<Step::preModuleTransition>(
              static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, md.id(), 0, t);
          logFile->write(std::move(msg));
        }
      });
      iRegistry.watchPostModuleEndJob([logFile, beginTime, iFilter](auto const& md) {
        auto info = iFilter->stopOnThread(md.id());
        if (info) {
          auto const t = duration_cast<duration_t>(now() - beginTime).count();
          auto msg = assembleAllocMessage<Step::postModuleTransition>(
              *info, static_cast<std::underlying_type_t<Phase>>(Phase::endJob), 0, md.id(), 0, t);
          logFile->write(std::move(msg));
        }
      });

      iRegistry.watchPreModuleEvent(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleEvent(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleEventAcquire(
          StreamEDModuleState<Step::preModuleEventAcquire>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleEventAcquire(
          StreamEDModuleState<Step::postModuleEventAcquire>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleEventDelayedGet(
          StreamEDModuleState<Step::preModuleEventDelayedGet>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleEventDelayedGet(
          StreamEDModuleState<Step::postModuleEventDelayedGet>(logFile, beginTime, iFilter));
      iRegistry.watchPreEventReadFromSource(
          StreamEDModuleState<Step::preEventReadFromSource>(logFile, beginTime, iFilter));
      iRegistry.watchPostEventReadFromSource(
          StreamEDModuleState<Step::postEventReadFromSource>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleTransform(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleTransform(StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleTransformAcquiring(
          StreamEDModuleState<Step::preModuleEventAcquire>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleTransformAcquiring(
          StreamEDModuleState<Step::postModuleEventAcquire>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleStreamBeginRun(
          StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleStreamBeginRun(
          StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleStreamEndRun(StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleStreamEndRun(
          StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleStreamBeginLumi(
          StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleStreamBeginLumi(
          StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleStreamEndLumi(
          StreamEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleStreamEndLumi(
          StreamEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleBeginProcessBlock(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleBeginProcessBlock(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleAccessInputProcessBlock(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleAccessInputProcessBlock(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleEndProcessBlock(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleEndProcessBlock(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleGlobalBeginRun(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleGlobalBeginRun(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleGlobalEndRun(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleGlobalEndRun(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleGlobalBeginLumi(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleGlobalBeginLumi(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPreModuleGlobalEndLumi(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleGlobalEndLumi(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleWriteProcessBlock(
          GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleWriteProcessBlock(
          GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleWriteRun(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleWriteRun(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      iRegistry.watchPreModuleWriteLumi(GlobalEDModuleState<Step::preModuleTransition>(logFile, beginTime, iFilter));
      iRegistry.watchPostModuleWriteLumi(GlobalEDModuleState<Step::postModuleTransition>(logFile, beginTime, iFilter));

      //ES Modules
      iRegistry.watchPreESModule(ESModuleState<Step::preESModule>(logFile, beginTime, recordIndices, iFilter));
      iRegistry.watchPostESModule(ESModuleState<Step::postESModule>(logFile, beginTime, recordIndices, iFilter));
      iRegistry.watchPreESModuleAcquire(
          ESModuleState<Step::preESModuleAcquire>(logFile, beginTime, recordIndices, iFilter));
      iRegistry.watchPostESModuleAcquire(
          ESModuleState<Step::postESModuleAcquire>(logFile, beginTime, recordIndices, iFilter));
    }

    std::ostringstream oss;
    oss << "# Transition Type         Symbol\n";
    oss << "#------------------------ ------\n";
    oss << "# startTracing            " << Phase::startTracing << "\n"
        << "# construction            " << Phase::construction << "\n"
        << "# getNextTransition       " << Phase::getNextTransition << "\n"
        << "# beginJob                " << Phase::beginJob << "\n"
        << "# beginStream             " << Phase::beginStream << "\n"
        << "# openFile                " << Phase::openFile << "\n"
        << "# beginProcessBlock       " << Phase::beginProcessBlock << "\n"
        << "# accessInputProcessBlock " << Phase::accessInputProcessBlock << "\n"
        << "# globalBeginRun          " << Phase::globalBeginRun << "\n"
        << "# streamBeginRun          " << Phase::streamBeginRun << "\n"
        << "# globalBeginLumi         " << Phase::globalBeginLumi << "\n"
        << "# streamBeginLumi         " << Phase::streamBeginLumi << "\n"
        << "# Event                   " << Phase::Event << "\n"
        << "# clearEvent              " << Phase::clearEvent << "\n"
        << "# streamEndLumi           " << Phase::streamEndLumi << "\n"
        << "# globalEndLumi           " << Phase::globalEndLumi << "\n"
        << "# globalWriteLumi         " << Phase::globalWriteLumi << "\n"
        << "# streamEndRun            " << Phase::streamEndRun << "\n"
        << "# globalEndRun            " << Phase::globalEndRun << "\n"
        << "# globalWriteRun          " << Phase::globalWriteRun << "\n"
        << "# endProcessBlock         " << Phase::endProcessBlock << "\n"
        << "# writeProcessBlock       " << Phase::writeProcessBlock << "\n"
        << "# endStream               " << Phase::endStream << "\n"
        << "# endJob                  " << Phase::endJob << "\n"
        << "# destruction             " << Phase::destruction << "\n\n";
    constexpr std::string_view kTransition = "   <Transition type> <Transition ID>";
    constexpr std::string_view kTransitionInfo = " <Run #> <LumiBlock #> <Event #>";
    constexpr std::string_view kTime = " <Time since begin of cmsRun (us)>";
    constexpr std::string_view kEDModule = " <EDModule ID> <Call ID>";
    constexpr std::string_view kESModule = " <ESModule ID> <Record ID> <Call ID>";
    constexpr std::string_view kAllocInfo =
        " <# allocs> <# deallocs> <additional bytes> <min bytes> <max bytes> <max allocation bytes>";

    oss << "# Step                       Symbol Entries\n"
        << "# -------------------------- ------ ------------------------------------------\n"
        << "# preSourceTransition           " << Step::preSourceTransition << kTransition << kTime << "\n"
        << "# postSourceTransition          " << Step::postSourceTransition << kTransition << kTime << " " << kAllocInfo
        << "\n"
        << "# preModuleEventAcquire         " << Step::preModuleEventAcquire << kTransition << kEDModule << kTime
        << "\n"
        << "# postModuleEventAcquire        " << Step::postModuleEventAcquire << kTransition << kEDModule << kTime
        << kAllocInfo << "\n"
        << "# preModuleTransition           " << Step::preModuleTransition << kTransition << kEDModule << kTime << "\n"
        << "# preEventReadFromSource        " << Step::preEventReadFromSource << kTransition << kEDModule << kTime
        << "\n"
        << "# postEventReadFromSource       " << Step::postEventReadFromSource << kTransition << kEDModule << kTime
        << kAllocInfo << "\n"
        << "# postModuleTransition          " << Step::postModuleTransition << kTransition << kEDModule << kTime
        << kAllocInfo << "\n"
        << "# preESModuleTransition         " << Step::preESModule << kTransition << kESModule << kTime << "\n"
        << "# postESModuleTransition        " << Step::postESModule << kTransition << kESModule << kTime << kAllocInfo
        << "\n"
        << "# preFrameworkTransition        " << Step::preFrameworkTransition << kTransition << kTransitionInfo << kTime
        << "\n"
        << "# preFrameworkTransition        " << Step::preFrameworkTransition << "                  "
        << Phase::clearEvent << " <Transition ID>" << kTransitionInfo << kTime << kAllocInfo << "\n"
        << "# postFrameworkTransition       " << Step::postFrameworkTransition << kTransition << kTransitionInfo
        << kTime << "\n";
    logFile->write(oss.str());
    return;
  }
}  // namespace edm::service::moduleAlloc
