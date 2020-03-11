#include "FWCore/Framework/interface/SubProcess.h"

#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/SubProcessParentageHelper.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/streamTransitionAsync.h"
#include "FWCore/Framework/src/globalTransitionAsync.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/ParameterSet/interface/IllegalParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"

#include "boost/range/adaptor/reversed.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace edm {

  SubProcess::SubProcess(ParameterSet& parameterSet,
                         ParameterSet const& topLevelParameterSet,
                         std::shared_ptr<ProductRegistry const> parentProductRegistry,
                         std::shared_ptr<BranchIDListHelper const> parentBranchIDListHelper,
                         ThinnedAssociationsHelper const& parentThinnedAssociationsHelper,
                         SubProcessParentageHelper const& parentSubProcessParentageHelper,
                         eventsetup::EventSetupsController& esController,
                         ActivityRegistry& parentActReg,
                         ServiceToken const& token,
                         serviceregistry::ServiceLegacy iLegacy,
                         PreallocationConfiguration const& preallocConfig,
                         ProcessContext const* parentProcessContext)
      : EDConsumerBase(),
        serviceToken_(),
        parentPreg_(parentProductRegistry),
        preg_(),
        branchIDListHelper_(),
        act_table_(),
        processConfiguration_(),
        historyLumiOffset_(preallocConfig.numberOfStreams()),
        historyRunOffset_(historyLumiOffset_ + preallocConfig.numberOfLuminosityBlocks()),
        processHistoryRegistries_(historyRunOffset_ + preallocConfig.numberOfRuns()),
        historyAppenders_(historyRunOffset_ + preallocConfig.numberOfRuns()),
        principalCache_(),
        esp_(),
        schedule_(),
        parentToChildPhID_(),
        subProcesses_(),
        processParameterSet_(),
        productSelectorRules_(parameterSet, "outputCommands", "OutputModule"),
        productSelector_(),
        wantAllEvents_(true) {
    //Setup the event selection
    Service<service::TriggerNamesService> tns;

    ParameterSet selectevents = parameterSet.getUntrackedParameterSet("SelectEvents", ParameterSet());

    selectevents.registerIt();  // Just in case this PSet is not registered
    wantAllEvents_ = detail::configureEventSelector(
        selectevents, tns->getProcessName(), getAllTriggerNames(), selectors_, consumesCollector());
    std::map<std::string, std::vector<std::pair<std::string, int>>> outputModulePathPositions;
    selector_config_id_ = detail::registerProperSelectionInfo(
        selectevents, "", outputModulePathPositions, parentProductRegistry->anyProductProduced());

    std::map<BranchID, bool> keepAssociation;
    selectProducts(*parentProductRegistry, parentThinnedAssociationsHelper, keepAssociation);

    std::string const maxEvents("maxEvents");
    std::string const maxLumis("maxLuminosityBlocks");

    // propagate_const<T> has no reset() function
    processParameterSet_ =
        std::unique_ptr<ParameterSet>(parameterSet.popParameterSet(std::string("process")).release());

    // if this process has a maxEvents or maxLuminosityBlocks parameter set, remove them.
    if (processParameterSet_->exists(maxEvents)) {
      processParameterSet_->popParameterSet(maxEvents);
    }
    if (processParameterSet_->exists(maxLumis)) {
      processParameterSet_->popParameterSet(maxLumis);
    }

    // if the top level process has a maxEvents or maxLuminosityBlocks parameter set, add them to this process.
    if (topLevelParameterSet.exists(maxEvents)) {
      processParameterSet_->addUntrackedParameter<ParameterSet>(
          maxEvents, topLevelParameterSet.getUntrackedParameterSet(maxEvents));
    }
    if (topLevelParameterSet.exists(maxLumis)) {
      processParameterSet_->addUntrackedParameter<ParameterSet>(
          maxLumis, topLevelParameterSet.getUntrackedParameterSet(maxLumis));
    }

    // If there are subprocesses, pop the subprocess parameter sets out of the process parameter set
    auto subProcessVParameterSet = popSubProcessVParameterSet(*processParameterSet_);
    bool hasSubProcesses = subProcessVParameterSet.size() != 0ull;

    // Validates the parameters in the 'options', 'maxEvents', and 'maxLuminosityBlocks'
    // top level parameter sets. Default values are also set in here if the
    // parameters were not explicitly set.
    validateTopLevelParameterSets(processParameterSet_.get());

    ScheduleItems items(*parentProductRegistry, *this);
    actReg_ = items.actReg_;

    //initialize the services
    ServiceToken iToken;

    // get any configured services.
    auto serviceSets = processParameterSet_->popVParameterSet(std::string("services"));

    ServiceToken newToken = items.initServices(serviceSets, *processParameterSet_, token, iLegacy, false);
    parentActReg.connectToSubProcess(*items.actReg_);
    serviceToken_ = items.addCPRandTNS(*processParameterSet_, newToken);

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    // intialize miscellaneous items
    items.initMisc(*processParameterSet_);

    // intialize the event setup provider
    esp_ = esController.makeProvider(*processParameterSet_, actReg_.get());

    branchIDListHelper_ = items.branchIDListHelper();
    updateBranchIDListHelper(parentBranchIDListHelper->branchIDLists());

    thinnedAssociationsHelper_ = items.thinnedAssociationsHelper();
    thinnedAssociationsHelper_->updateFromParentProcess(
        parentThinnedAssociationsHelper, keepAssociation, droppedBranchIDToKeptBranchID_);

    // intialize the Schedule
    schedule_ = items.initSchedule(*processParameterSet_, hasSubProcesses, preallocConfig, &processContext_);

    // set the items
    act_table_ = std::move(items.act_table_);
    preg_ = items.preg();

    subProcessParentageHelper_ = items.subProcessParentageHelper();
    subProcessParentageHelper_->update(parentSubProcessParentageHelper, *parentProductRegistry);

    //CMS-THREADING this only works since Run/Lumis are synchronous so when principalCache asks for
    // the reducedProcessHistoryID from a full ProcessHistoryID that registry will not be in use by
    // another thread. We really need to change how this is done in the PrincipalCache.
    principalCache_.setProcessHistoryRegistry(processHistoryRegistries_[historyRunOffset_]);

    processConfiguration_ = items.processConfiguration();
    processContext_.setProcessConfiguration(processConfiguration_.get());
    processContext_.setParentProcessContext(parentProcessContext);

    principalCache_.setNumberOfConcurrentPrincipals(preallocConfig);
    for (unsigned int index = 0; index < preallocConfig.numberOfStreams(); ++index) {
      auto ep = std::make_shared<EventPrincipal>(preg_,
                                                 branchIDListHelper(),
                                                 thinnedAssociationsHelper(),
                                                 *processConfiguration_,
                                                 &(historyAppenders_[index]),
                                                 index,
                                                 false /*not primary process*/);
      principalCache_.insert(ep);
    }
    for (unsigned int index = 0; index < preallocConfig.numberOfLuminosityBlocks(); ++index) {
      auto lbpp = std::make_unique<LuminosityBlockPrincipal>(
          preg_, *processConfiguration_, &(historyAppenders_[historyLumiOffset_ + index]), index, false);
      principalCache_.insert(std::move(lbpp));
    }

    inUseLumiPrincipals_.resize(preallocConfig.numberOfLuminosityBlocks());

    subProcesses_.reserve(subProcessVParameterSet.size());
    for (auto& subProcessPSet : subProcessVParameterSet) {
      subProcesses_.emplace_back(subProcessPSet,
                                 topLevelParameterSet,
                                 preg_,
                                 branchIDListHelper(),
                                 *thinnedAssociationsHelper_,
                                 *subProcessParentageHelper_,
                                 esController,
                                 *items.actReg_,
                                 newToken,
                                 iLegacy,
                                 preallocConfig,
                                 &processContext_);
    }
  }

  SubProcess::~SubProcess() {}

  void SubProcess::doBeginJob() { this->beginJob(); }

  void SubProcess::doEndJob() { endJob(); }

  void SubProcess::beginJob() {
    // If event selection is being used, the SubProcess class reads TriggerResults
    // object(s) in the parent process from the event. This next call is needed for
    // getByToken to work properly. Normally, this is done by the worker, but since
    // a SubProcess is not a module, it has no worker.
    updateLookup(InEvent, *parentPreg_->productLookup(InEvent), false);

    if (!droppedBranchIDToKeptBranchID().empty()) {
      fixBranchIDListsForEDAliases(droppedBranchIDToKeptBranchID());
    }
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->convertCurrentProcessAlias(processConfiguration_->processName());
    pathsAndConsumesOfModules_.initialize(schedule_.get(), preg_);
    //NOTE: this may throw
    checkForModuleDependencyCorrectness(pathsAndConsumesOfModules_, false);
    actReg_->preBeginJobSignal_(pathsAndConsumesOfModules_, processContext_);
    schedule_->beginJob(*preg_, esp_->recordsToProxyIndices());
    for_all(subProcesses_, [](auto& subProcess) { subProcess.doBeginJob(); });
  }

  void SubProcess::endJob() {
    ServiceRegistry::Operate operate(serviceToken_);
    ExceptionCollector c(
        "Multiple exceptions were thrown while executing endJob. An exception message follows for each.");
    schedule_->endJob(c);
    for (auto& subProcess : subProcesses_) {
      c.call([&subProcess]() { subProcess.doEndJob(); });
    }
    if (c.hasThrown()) {
      c.rethrow();
    }
  }

  void SubProcess::selectProducts(ProductRegistry const& preg,
                                  ThinnedAssociationsHelper const& parentThinnedAssociationsHelper,
                                  std::map<BranchID, bool>& keepAssociation) {
    if (productSelector_.initialized())
      return;
    productSelector_.initialize(productSelectorRules_, preg.allBranchDescriptions());

    // TODO: See if we can collapse keptProducts_ and productSelector_ into a
    // single object. See the notes in the header for ProductSelector
    // for more information.

    std::map<BranchID, BranchDescription const*> trueBranchIDToKeptBranchDesc;
    std::vector<BranchDescription const*> associationDescriptions;
    std::set<BranchID> keptProductsInEvent;

    for (auto const& it : preg.productList()) {
      BranchDescription const& desc = it.second;
      if (desc.transient()) {
        // if the class of the branch is marked transient, output nothing
      } else if (!desc.present() && !desc.produced()) {
        // else if the branch containing the product has been previously dropped,
        // output nothing
      } else if (desc.unwrappedType() == typeid(ThinnedAssociation)) {
        associationDescriptions.push_back(&desc);
      } else if (productSelector_.selected(desc)) {
        keepThisBranch(desc, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
      }
    }

    parentThinnedAssociationsHelper.selectAssociationProducts(
        associationDescriptions, keptProductsInEvent, keepAssociation);

    for (auto association : associationDescriptions) {
      if (keepAssociation[association->branchID()]) {
        keepThisBranch(*association, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
      }
    }

    // Now fill in a mapping needed in the case that a branch was dropped while its EDAlias was kept.
    ProductSelector::fillDroppedToKept(preg, trueBranchIDToKeptBranchDesc, droppedBranchIDToKeptBranchID_);
  }

  void SubProcess::keepThisBranch(BranchDescription const& desc,
                                  std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc,
                                  std::set<BranchID>& keptProductsInEvent) {
    ProductSelector::checkForDuplicateKeptBranch(desc, trueBranchIDToKeptBranchDesc);

    if (desc.branchType() == InEvent) {
      if (desc.produced()) {
        keptProductsInEvent.insert(desc.originalBranchID());
      } else {
        keptProductsInEvent.insert(desc.branchID());
      }
    }
    EDGetToken token = consumes(TypeToGet{desc.unwrappedTypeID(), PRODUCT_TYPE},
                                InputTag{desc.moduleLabel(), desc.productInstanceName(), desc.processName()});

    // Now put it in the list of selected branches.
    keptProducts_[desc.branchType()].push_back(std::make_pair(&desc, token));
  }

  void SubProcess::fixBranchIDListsForEDAliases(
      std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID) {
    // Check for branches dropped while an EDAlias was kept.
    // Replace BranchID of each dropped branch with that of the kept alias.
    for (BranchIDList& branchIDList : branchIDListHelper_->mutableBranchIDLists()) {
      for (BranchID::value_type& branchID : branchIDList) {
        std::map<BranchID::value_type, BranchID::value_type>::const_iterator iter =
            droppedBranchIDToKeptBranchID.find(branchID);
        if (iter != droppedBranchIDToKeptBranchID.end()) {
          branchID = iter->second;
        }
      }
    }
    for_all(subProcesses_, [&droppedBranchIDToKeptBranchID](auto& subProcess) {
      subProcess.fixBranchIDListsForEDAliases(droppedBranchIDToKeptBranchID);
    });
  }

  void SubProcess::doEventAsync(WaitingTaskHolder iHolder,
                                EventPrincipal const& ep,
                                std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    ServiceRegistry::Operate operate(serviceToken_);
    /* BEGIN relevant bits from OutputModule::doEvent */
    if (!wantAllEvents_) {
      EventForOutput e(ep, ModuleDescription(), nullptr);
      e.setConsumer(this);
      if (!selectors_.wantEvent(e)) {
        return;
      }
    }
    processAsync(std::move(iHolder), ep, iEventSetupImpls);
    /* END relevant bits from OutputModule::doEvent */
  }

  void SubProcess::processAsync(WaitingTaskHolder iHolder,
                                EventPrincipal const& principal,
                                std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    EventAuxiliary aux(principal.aux());
    aux.setProcessHistoryID(principal.processHistoryID());

    EventSelectionIDVector esids{principal.eventSelectionIDs()};
    if (principal.productRegistry().anyProductProduced() || !wantAllEvents_) {
      esids.push_back(selector_config_id_);
    }

    EventPrincipal& ep = principalCache_.eventPrincipal(principal.streamID().value());
    auto& processHistoryRegistry = processHistoryRegistries_[principal.streamID().value()];
    processHistoryRegistry.registerProcessHistory(principal.processHistory());
    BranchListIndexes bli(principal.branchListIndexes());
    branchIDListHelper_->fixBranchListIndexes(bli);
    bool deepCopyRetriever = false;
    ep.fillEventPrincipal(
        aux,
        &principal.processHistory(),
        std::move(esids),
        std::move(bli),
        *(principal.productProvenanceRetrieverPtr()),  //NOTE: this transfers the per product provenance
        principal.reader(),
        deepCopyRetriever);
    ep.setLuminosityBlockPrincipal(inUseLumiPrincipals_[principal.luminosityBlockPrincipal().index()].get());
    propagateProducts(InEvent, principal, ep);

    WaitingTaskHolder finalizeEventTask(
        make_waiting_task(tbb::task::allocate_root(), [&ep, iHolder](std::exception_ptr const* iPtr) mutable {
          ep.clearEventPrincipal();
          if (iPtr) {
            iHolder.doneWaiting(*iPtr);
          } else {
            iHolder.doneWaiting(std::exception_ptr());
          }
        }));
    WaitingTaskHolder afterProcessTask;
    if (subProcesses_.empty()) {
      afterProcessTask = std::move(finalizeEventTask);
    } else {
      afterProcessTask = WaitingTaskHolder(
          make_waiting_task(tbb::task::allocate_root(),
                            [this, &ep, finalizeEventTask, iEventSetupImpls](std::exception_ptr const* iPtr) mutable {
                              if (not iPtr) {
                                for (auto& subProcess : boost::adaptors::reverse(subProcesses_)) {
                                  subProcess.doEventAsync(finalizeEventTask, ep, iEventSetupImpls);
                                }
                              } else {
                                finalizeEventTask.doneWaiting(*iPtr);
                              }
                            }));
    }

    schedule_->processOneEventAsync(std::move(afterProcessTask),
                                    ep.streamID().value(),
                                    ep,
                                    *((*iEventSetupImpls)[esp_->subProcessIndex()]),
                                    serviceToken_);
  }

  void SubProcess::doBeginRunAsync(WaitingTaskHolder iHolder,
                                   RunPrincipal const& principal,
                                   IOVSyncValue const& ts,
                                   std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    ServiceRegistry::Operate operate(serviceToken_);

    auto aux = std::make_shared<RunAuxiliary>(principal.aux());
    aux->setProcessHistoryID(principal.processHistoryID());
    auto rpp = std::make_shared<RunPrincipal>(aux,
                                              preg_,
                                              *processConfiguration_,
                                              &(historyAppenders_[historyRunOffset_ + principal.index()]),
                                              principal.index(),
                                              false);
    auto& processHistoryRegistry = processHistoryRegistries_[historyRunOffset_ + principal.index()];
    processHistoryRegistry.registerProcessHistory(principal.processHistory());
    rpp->fillRunPrincipal(processHistoryRegistry, principal.reader());
    principalCache_.insert(rpp);

    ProcessHistoryID const& parentInputReducedPHID = principal.reducedProcessHistoryID();
    ProcessHistoryID const& inputReducedPHID = rpp->reducedProcessHistoryID();

    parentToChildPhID_.insert(std::make_pair(parentInputReducedPHID, inputReducedPHID));

    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    propagateProducts(InRun, principal, rp);
    typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> Traits;
    beginGlobalTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, rp, ts, esp_->eventSetupImpl(), iEventSetupImpls, serviceToken_, subProcesses_);
  }

  void SubProcess::doEndRunAsync(WaitingTaskHolder iHolder,
                                 RunPrincipal const& principal,
                                 IOVSyncValue const& ts,
                                 std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
                                 bool cleaningUpAfterException) {
    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    propagateProducts(InRun, principal, rp);
    typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> Traits;
    endGlobalTransitionAsync<Traits>(std::move(iHolder),
                                     *schedule_,
                                     rp,
                                     ts,
                                     esp_->eventSetupImpl(),
                                     iEventSetupImpls,
                                     serviceToken_,
                                     subProcesses_,
                                     cleaningUpAfterException);
  }

  void SubProcess::writeRunAsync(edm::WaitingTaskHolder task,
                                 ProcessHistoryID const& parentPhID,
                                 int runNumber,
                                 MergeableRunProductMetadata const* mergeableRunProductMetadata) {
    ServiceRegistry::Operate operate(serviceToken_);
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    auto const& childPhID = it->second;

    auto subTasks = edm::make_waiting_task(
        tbb::task::allocate_root(),
        [this, childPhID, runNumber, task, mergeableRunProductMetadata](std::exception_ptr const* iExcept) mutable {
          if (iExcept) {
            task.doneWaiting(*iExcept);
          } else {
            ServiceRegistry::Operate operateWriteRun(serviceToken_);
            for (auto& s : subProcesses_) {
              s.writeRunAsync(task, childPhID, runNumber, mergeableRunProductMetadata);
            }
          }
        });
    schedule_->writeRunAsync(WaitingTaskHolder(subTasks),
                             principalCache_.runPrincipal(childPhID, runNumber),
                             &processContext_,
                             actReg_.get(),
                             mergeableRunProductMetadata);
  }

  void SubProcess::deleteRunFromCache(ProcessHistoryID const& parentPhID, int runNumber) {
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    auto const& childPhID = it->second;
    principalCache_.deleteRun(childPhID, runNumber);
    for_all(subProcesses_,
            [&childPhID, runNumber](auto& subProcess) { subProcess.deleteRunFromCache(childPhID, runNumber); });
  }

  void SubProcess::doBeginLuminosityBlockAsync(
      WaitingTaskHolder iHolder,
      LuminosityBlockPrincipal const& principal,
      IOVSyncValue const& ts,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    ServiceRegistry::Operate operate(serviceToken_);

    auto aux = principal.aux();
    aux.setProcessHistoryID(principal.processHistoryID());
    auto lbpp = principalCache_.getAvailableLumiPrincipalPtr();
    lbpp->setAux(aux);
    auto& processHistoryRegistry = processHistoryRegistries_[historyLumiOffset_ + lbpp->index()];
    inUseLumiPrincipals_[principal.index()] = lbpp;
    processHistoryRegistry.registerProcessHistory(principal.processHistory());
    lbpp->fillLuminosityBlockPrincipal(&principal.processHistory(), principal.reader());
    lbpp->setRunPrincipal(principalCache_.runPrincipalPtr());
    LuminosityBlockPrincipal& lbp = *lbpp;
    propagateProducts(InLumi, principal, lbp);
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Traits;
    beginGlobalTransitionAsync<Traits>(std::move(iHolder),
                                       *schedule_,
                                       lbp,
                                       ts,
                                       *((*iEventSetupImpls)[esp_->subProcessIndex()]),
                                       iEventSetupImpls,
                                       serviceToken_,
                                       subProcesses_);
  }

  void SubProcess::doEndLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                             LuminosityBlockPrincipal const& principal,
                                             IOVSyncValue const& ts,
                                             std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
                                             bool cleaningUpAfterException) {
    LuminosityBlockPrincipal& lbp = *inUseLumiPrincipals_[principal.index()];
    propagateProducts(InLumi, principal, lbp);
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Traits;
    endGlobalTransitionAsync<Traits>(std::move(iHolder),
                                     *schedule_,
                                     lbp,
                                     ts,
                                     *((*iEventSetupImpls)[esp_->subProcessIndex()]),
                                     iEventSetupImpls,
                                     serviceToken_,
                                     subProcesses_,
                                     cleaningUpAfterException);
  }

  void SubProcess::writeLumiAsync(WaitingTaskHolder task, LuminosityBlockPrincipal& principal) {
    ServiceRegistry::Operate operate(serviceToken_);

    auto l = inUseLumiPrincipals_[principal.index()];
    auto subTasks =
        edm::make_waiting_task(tbb::task::allocate_root(), [this, l, task](std::exception_ptr const* iExcept) mutable {
          if (iExcept) {
            task.doneWaiting(*iExcept);
          } else {
            ServiceRegistry::Operate operateWriteLumi(serviceToken_);
            for (auto& s : subProcesses_) {
              s.writeLumiAsync(task, *l);
            }
          }
        });
    schedule_->writeLumiAsync(WaitingTaskHolder(subTasks), *l, &processContext_, actReg_.get());
  }

  void SubProcess::deleteLumiFromCache(LuminosityBlockPrincipal& principal) {
    //release from list but stay around till end of routine
    auto lb = std::move(inUseLumiPrincipals_[principal.index()]);
    for (auto& s : subProcesses_) {
      s.deleteLumiFromCache(*lb);
    }
    lb->clearPrincipal();
  }

  void SubProcess::doBeginStream(unsigned int iID) {
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->beginStream(iID);
    for_all(subProcesses_, [iID](auto& subProcess) { subProcess.doBeginStream(iID); });
  }

  void SubProcess::doEndStream(unsigned int iID) {
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->endStream(iID);
    for_all(subProcesses_, [iID](auto& subProcess) { subProcess.doEndStream(iID); });
  }

  void SubProcess::doStreamBeginRunAsync(WaitingTaskHolder iHolder,
                                         unsigned int id,
                                         RunPrincipal const& principal,
                                         IOVSyncValue const& ts,
                                         std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    typedef OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> Traits;

    RunPrincipal& rp = *principalCache_.runPrincipalPtr();

    beginStreamTransitionAsync<Traits>(std::move(iHolder),
                                       *schedule_,
                                       id,
                                       rp,
                                       ts,
                                       esp_->eventSetupImpl(),
                                       iEventSetupImpls,
                                       serviceToken_,
                                       subProcesses_);
  }

  void SubProcess::doStreamEndRunAsync(WaitingTaskHolder iHolder,
                                       unsigned int id,
                                       RunPrincipal const& principal,
                                       IOVSyncValue const& ts,
                                       std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
                                       bool cleaningUpAfterException) {
    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    typedef OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> Traits;

    endStreamTransitionAsync<Traits>(std::move(iHolder),
                                     *schedule_,
                                     id,
                                     rp,
                                     ts,
                                     esp_->eventSetupImpl(),
                                     iEventSetupImpls,
                                     serviceToken_,
                                     subProcesses_,
                                     cleaningUpAfterException);
  }

  void SubProcess::doStreamBeginLuminosityBlockAsync(
      WaitingTaskHolder iHolder,
      unsigned int id,
      LuminosityBlockPrincipal const& principal,
      IOVSyncValue const& ts,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> Traits;

    LuminosityBlockPrincipal& lbp = *inUseLumiPrincipals_[principal.index()];

    beginStreamTransitionAsync<Traits>(std::move(iHolder),
                                       *schedule_,
                                       id,
                                       lbp,
                                       ts,
                                       *((*iEventSetupImpls)[esp_->subProcessIndex()]),
                                       iEventSetupImpls,
                                       serviceToken_,
                                       subProcesses_);
  }

  void SubProcess::doStreamEndLuminosityBlockAsync(
      WaitingTaskHolder iHolder,
      unsigned int id,
      LuminosityBlockPrincipal const& principal,
      IOVSyncValue const& ts,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
      bool cleaningUpAfterException) {
    LuminosityBlockPrincipal& lbp = *inUseLumiPrincipals_[principal.index()];
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> Traits;
    endStreamTransitionAsync<Traits>(std::move(iHolder),
                                     *schedule_,
                                     id,
                                     lbp,
                                     ts,
                                     *((*iEventSetupImpls)[esp_->subProcessIndex()]),
                                     iEventSetupImpls,
                                     serviceToken_,
                                     subProcesses_,
                                     cleaningUpAfterException);
  }

  void SubProcess::propagateProducts(BranchType type, Principal const& parentPrincipal, Principal& principal) const {
    SelectedProducts const& keptVector = keptProducts()[type];
    for (auto const& item : keptVector) {
      BranchDescription const& desc = *item.first;
      ProductResolverBase const* parentProductResolver = parentPrincipal.getProductResolver(desc.branchID());
      if (parentProductResolver != nullptr) {
        ProductResolverBase* productResolver = principal.getModifiableProductResolver(desc.branchID());
        if (productResolver != nullptr) {
          //Propagate the per event(run)(lumi) data for this product to the subprocess.
          //First, the product itself.
          productResolver->connectTo(*parentProductResolver, &parentPrincipal);
        }
      }
    }
  }

  void SubProcess::updateBranchIDListHelper(BranchIDLists const& branchIDLists) {
    branchIDListHelper_->updateFromParent(branchIDLists);
    for_all(subProcesses_,
            [this](auto& subProcess) { subProcess.updateBranchIDListHelper(branchIDListHelper_->branchIDLists()); });
  }

  // Call respondToOpenInputFile() on all Modules
  void SubProcess::respondToOpenInputFile(FileBlock const& fb) {
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->respondToOpenInputFile(fb);
    for_all(subProcesses_, [&fb](auto& subProcess) { subProcess.respondToOpenInputFile(fb); });
  }

  // free function
  std::vector<ParameterSet> popSubProcessVParameterSet(ParameterSet& parameterSet) {
    std::vector<std::string> subProcesses =
        parameterSet.getUntrackedParameter<std::vector<std::string>>("@all_subprocesses");
    if (!subProcesses.empty()) {
      return parameterSet.popVParameterSet("subProcesses");
    }
    return {};
  }
}  // namespace edm
