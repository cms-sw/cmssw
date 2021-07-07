// -*- C++ -*-
//
// Package:     <package>
// Module:      EDLooperBase
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:44:26 EDT 2006

#include "FWCore/Framework/interface/EDLooperBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Concurrency/interface/include_first_syncWait.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "tbb/global_control.h"

namespace edm {

  EDLooperBase::EDLooperBase()
      : iCounter_(0),
        act_table_(nullptr),
        moduleChanger_(nullptr),
        moduleDescription_("Looper", "looper"),
        moduleCallingContext_(&moduleDescription_) {}
  EDLooperBase::~EDLooperBase() noexcept(false) {}

  void EDLooperBase::doStartingNewLoop() { startingNewLoop(iCounter_); }

  EDLooperBase::Status EDLooperBase::doDuringLoop(edm::EventPrincipal& eventPrincipal,
                                                  const edm::EventSetupImpl& esi,
                                                  edm::ProcessingController& ioController,
                                                  StreamContext* streamContext) {
    streamContext->setTransition(StreamContext::Transition::kEvent);
    streamContext->setEventID(eventPrincipal.id());
    streamContext->setRunIndex(eventPrincipal.luminosityBlockPrincipal().runPrincipal().index());
    streamContext->setLuminosityBlockIndex(eventPrincipal.luminosityBlockPrincipal().index());
    streamContext->setTimestamp(eventPrincipal.time());
    ParentContext parentContext(streamContext);
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    Event event(eventPrincipal, moduleDescription_, &moduleCallingContext_);
    event.setConsumer(this);

    Status status = kContinue;
    try {
      ESParentContext parentC(&moduleCallingContext_);
      const EventSetup es{
          esi, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), parentC, false};
      status = duringLoop(event, es, ioController);
    } catch (cms::Exception& e) {
      e.addContext("Calling the 'duringLoop' method of a looper");
      exception_actions::ActionCodes action = (act_table_->find(e.category()));
      if (action != exception_actions::Rethrow) {
        edm::printCmsExceptionWarning("SkipEvent", e);
      } else {
        throw;
      }
    }
    return status;
  }

  EDLooperBase::Status EDLooperBase::doEndOfLoop(const edm::EventSetupImpl& esi) {
    ESParentContext parentC(&moduleCallingContext_);
    const EventSetup es{
        esi, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC, false};
    return endOfLoop(es, iCounter_);
  }

  void EDLooperBase::prepareForNextLoop(eventsetup::EventSetupProvider* esp) {
    ++iCounter_;

    std::set<edm::eventsetup::EventSetupRecordKey> const& keys = modifyingRecords();
    for_all(keys,
            std::bind(&eventsetup::EventSetupProvider::resetRecordPlusDependentRecords, esp, std::placeholders::_1));
  }

  void EDLooperBase::beginOfJob(const edm::EventSetupImpl& iImpl) {
    ESParentContext parentC(&moduleCallingContext_);
    beginOfJob(EventSetup{iImpl,
                          static_cast<unsigned int>(Transition::BeginRun),
                          esGetTokenIndices(Transition::BeginRun),
                          parentC,
                          false});
  }
  void EDLooperBase::beginOfJob(const edm::EventSetup&) { beginOfJob(); }
  void EDLooperBase::beginOfJob() {}

  void EDLooperBase::endOfJob() {}

  void EDLooperBase::doBeginRun(RunPrincipal& iRP, EventSetupImpl const& iES, ProcessContext* processContext) {
    GlobalContext globalContext(GlobalContext::Transition::kBeginRun,
                                LuminosityBlockID(iRP.run(), 0),
                                iRP.index(),
                                LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                                iRP.beginTime(),
                                processContext);
    ParentContext parentContext(&globalContext);
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    Run run(iRP, moduleDescription_, &moduleCallingContext_, false);
    run.setConsumer(this);
    ESParentContext parentC(&moduleCallingContext_);
    const EventSetup es{
        iES, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), parentC, false};
    beginRun(run, es);
  }

  void EDLooperBase::doEndRun(RunPrincipal& iRP, EventSetupImpl const& iES, ProcessContext* processContext) {
    GlobalContext globalContext(GlobalContext::Transition::kEndRun,
                                LuminosityBlockID(iRP.run(), 0),
                                iRP.index(),
                                LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                                iRP.endTime(),
                                processContext);
    ParentContext parentContext(&globalContext);
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    Run run(iRP, moduleDescription_, &moduleCallingContext_, true);
    run.setConsumer(this);
    ESParentContext parentC(&moduleCallingContext_);
    const EventSetup es{
        iES, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC, false};
    endRun(run, es);
  }
  void EDLooperBase::doBeginLuminosityBlock(LuminosityBlockPrincipal& iLB,
                                            EventSetupImpl const& iES,
                                            ProcessContext* processContext) {
    GlobalContext globalContext(GlobalContext::Transition::kBeginLuminosityBlock,
                                iLB.id(),
                                iLB.runPrincipal().index(),
                                iLB.index(),
                                iLB.beginTime(),
                                processContext);
    ParentContext parentContext(&globalContext);
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    LuminosityBlock luminosityBlock(iLB, moduleDescription_, &moduleCallingContext_, false);
    luminosityBlock.setConsumer(this);
    ESParentContext parentC(&moduleCallingContext_);
    const EventSetup es{iES,
                        static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                        esGetTokenIndices(Transition::BeginLuminosityBlock),
                        parentC,
                        false};
    beginLuminosityBlock(luminosityBlock, es);
  }
  void EDLooperBase::doEndLuminosityBlock(LuminosityBlockPrincipal& iLB,
                                          EventSetupImpl const& iES,
                                          ProcessContext* processContext) {
    GlobalContext globalContext(GlobalContext::Transition::kEndLuminosityBlock,
                                iLB.id(),
                                iLB.runPrincipal().index(),
                                iLB.index(),
                                iLB.beginTime(),
                                processContext);
    ParentContext parentContext(&globalContext);
    ModuleContextSentry moduleContextSentry(&moduleCallingContext_, parentContext);
    LuminosityBlock luminosityBlock(iLB, moduleDescription_, &moduleCallingContext_, true);
    luminosityBlock.setConsumer(this);
    ESParentContext parentC(&moduleCallingContext_);
    const EventSetup es{iES,
                        static_cast<unsigned int>(Transition::EndLuminosityBlock),
                        esGetTokenIndices(Transition::EndLuminosityBlock),
                        parentC,
                        false};
    endLuminosityBlock(luminosityBlock, es);
  }

  void EDLooperBase::beginRun(Run const&, EventSetup const&) {}
  void EDLooperBase::endRun(Run const&, EventSetup const&) {}
  void EDLooperBase::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) {}
  void EDLooperBase::endLuminosityBlock(LuminosityBlock const&, EventSetup const&) {}

  void EDLooperBase::attachTo(ActivityRegistry&) {}

  void EDLooperBase::prefetchAsync(WaitingTaskHolder iTask,
                                   ServiceToken const& token,
                                   Transition iTrans,
                                   Principal const& iPrincipal,
                                   EventSetupImpl const& iImpl) const {
    esPrefetchAsync(iTask, iImpl, iTrans, token);
    edPrefetchAsync(std::move(iTask), token, iPrincipal);
  }

  void EDLooperBase::edPrefetchAsync(WaitingTaskHolder iTask,
                                     ServiceToken const& token,
                                     Principal const& iPrincipal) const {
    //Based on Worker edPrefetchAsync

    // Prefetch products the module declares it consumes
    std::vector<ProductResolverIndexAndSkipBit> const& items = itemsToGetFrom(iPrincipal.branchType());

    for (auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      bool skipCurrentProcess = item.skipCurrentProcess();
      if (productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal.prefetchAsync(iTask, productResolverIndex, skipCurrentProcess, token, &moduleCallingContext_);
      }
    }
  }

  void EDLooperBase::esPrefetchAsync(WaitingTaskHolder iTask,
                                     EventSetupImpl const& iImpl,
                                     Transition iTrans,
                                     ServiceToken const& iToken) const {
    //Based on Worker::esPrefetchAsync
    if (iTrans >= edm::Transition::NumberOfEventSetupTransitions) {
      return;
    }
    auto const& recs = esGetTokenRecordIndicesVector(iTrans);
    auto const& items = esGetTokenIndicesVector(iTrans);

    assert(items.size() == recs.size());
    if (items.empty()) {
      return;
    }

    //Thread case of 1 thread special. The master thread is doing a wait_for_all on the
    // default tbb arena. It will not process any tasks on the es arena. We need to add a
    // task that will synchronously do a wait_for_all in the es arena to be sure prefetching
    // will work.

    if UNLIKELY (tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism) == 1) {
      auto taskGroup = iTask.group();
      ServiceWeakToken weakToken = iToken;
      taskGroup->run([this, task = std::move(iTask), iTrans, &iImpl, weakToken]() {
        std::exception_ptr exceptPtr{};
        iImpl.taskArena()->execute([this, iTrans, &iImpl, weakToken, &exceptPtr]() {
          exceptPtr = syncWait([&](WaitingTaskHolder&& iHolder) {
            auto const& recs = esGetTokenRecordIndicesVector(iTrans);
            auto const& items = esGetTokenIndicesVector(iTrans);
            auto hWaitTask = std::move(iHolder);
            auto token = weakToken.lock();
            for (size_t i = 0; i != items.size(); ++i) {
              if (recs[i] != ESRecordIndex{}) {
                auto rec = iImpl.findImpl(recs[i]);
                if (rec) {
                  rec->prefetchAsync(hWaitTask, items[i], &iImpl, token, ESParentContext(&moduleCallingContext_));
                }
              }
            }
          });  //syncWait
        });    //esTaskArena().execute
        //note use of a copy gets around declaring the lambda as mutable
        auto tempTask = task;
        tempTask.doneWaiting(exceptPtr);
      });  //group.run
    } else {
      auto group = iTask.group();
      //We need iTask to run in the default arena since it is not an ES task
      auto task = make_waiting_task(
          [holder = WaitingTaskWithArenaHolder(std::move(iTask))](std::exception_ptr const* iExcept) mutable {
            if (iExcept) {
              holder.doneWaiting(*iExcept);
            } else {
              holder.doneWaiting(std::exception_ptr{});
            }
          });

      WaitingTaskHolder tempH(*group, task);
      iImpl.taskArena()->execute([&]() {
        for (size_t i = 0; i != items.size(); ++i) {
          if (recs[i] != ESRecordIndex{}) {
            auto rec = iImpl.findImpl(recs[i]);
            if (rec) {
              rec->prefetchAsync(tempH, items[i], &iImpl, iToken, ESParentContext(&moduleCallingContext_));
            }
          }
        }
      });
    }
  }

  std::set<eventsetup::EventSetupRecordKey> EDLooperBase::modifyingRecords() const {
    return std::set<eventsetup::EventSetupRecordKey>();
  }

  void EDLooperBase::copyInfo(const ScheduleInfo& iInfo) { scheduleInfo_ = std::make_unique<ScheduleInfo>(iInfo); }
  void EDLooperBase::setModuleChanger(ModuleChanger* iChanger) { moduleChanger_ = iChanger; }

  ModuleChanger* EDLooperBase::moduleChanger() { return moduleChanger_; }
  const ScheduleInfo* EDLooperBase::scheduleInfo() const { return scheduleInfo_.get(); }

}  // namespace edm
