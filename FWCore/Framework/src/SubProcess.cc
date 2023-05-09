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
#include "FWCore/Common/interface/SubProcessBlockHelper.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/src/OutputModuleDescription.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/ScheduleItems.h"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/Framework/interface/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/streamTransitionAsync.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/globalTransitionAsync.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/ParameterSet/interface/IllegalParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"

#include "boost/range/adaptor/reversed.hpp"

#include <cassert>
#include <exception>
#include <string>

namespace edm {

  SubProcess::SubProcess(ParameterSet& parameterSet,
                         ParameterSet const& topLevelParameterSet,
                         std::shared_ptr<ProductRegistry const> parentProductRegistry,
                         std::shared_ptr<BranchIDListHelper const> parentBranchIDListHelper,
                         ProcessBlockHelperBase const& parentProcessBlockHelper,
                         ThinnedAssociationsHelper const& parentThinnedAssociationsHelper,
                         SubProcessParentageHelper const& parentSubProcessParentageHelper,
                         eventsetup::EventSetupsController& esController,
                         ActivityRegistry& parentActReg,
                         ServiceToken const& token,
                         serviceregistry::ServiceLegacy iLegacy,
                         PreallocationConfiguration const& preallocConfig,
                         ProcessContext const* parentProcessContext,
                         ModuleTypeResolverMaker const* typeResolverMaker)
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
    bool hasSubProcesses = !subProcessVParameterSet.empty();

    // Validates the parameters in the 'options', 'maxEvents', and 'maxLuminosityBlocks'
    // top level parameter sets. Default values are also set in here if the
    // parameters were not explicitly set.
    validateTopLevelParameterSets(processParameterSet_.get());

    processBlockHelper_ = std::make_shared<SubProcessBlockHelper>();

    ScheduleItems items(*parentProductRegistry, *this, *processBlockHelper_, parentProcessBlockHelper);
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
    schedule_ = items.initSchedule(*processParameterSet_,
                                   hasSubProcesses,
                                   preallocConfig,
                                   &processContext_,
                                   typeResolverMaker,
                                   *processBlockHelper_);

    // set the items
    act_table_ = std::move(items.act_table_);
    preg_ = items.preg();

    subProcessParentageHelper_ = items.subProcessParentageHelper();
    subProcessParentageHelper_->update(parentSubProcessParentageHelper, *parentProductRegistry);

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
                                                 false /*not primary process*/,
                                                 &*processBlockHelper_);
      principalCache_.insert(ep);
    }

    for (unsigned int index = 0; index < preallocConfig.numberOfRuns(); ++index) {
      auto rpp = std::make_unique<RunPrincipal>(
          preg_, *processConfiguration_, &(historyAppenders_[historyRunOffset_ + index]), index, false);
      principalCache_.insert(std::move(rpp));
    }

    for (unsigned int index = 0; index < preallocConfig.numberOfLuminosityBlocks(); ++index) {
      auto lbpp = std::make_unique<LuminosityBlockPrincipal>(
          preg_, *processConfiguration_, &(historyAppenders_[historyLumiOffset_ + index]), index, false);
      principalCache_.insert(std::move(lbpp));
    }

    {
      auto pb = std::make_unique<ProcessBlockPrincipal>(preg_, *processConfiguration_, false);
      principalCache_.insert(std::move(pb));

      auto pbForInput = std::make_unique<ProcessBlockPrincipal>(preg_, *processConfiguration_, false);
      principalCache_.insertForInput(std::move(pbForInput));
    }

    inUseRunPrincipals_.resize(preallocConfig.numberOfRuns());
    inUseLumiPrincipals_.resize(preallocConfig.numberOfLuminosityBlocks());

    subProcesses_.reserve(subProcessVParameterSet.size());
    for (auto& subProcessPSet : subProcessVParameterSet) {
      subProcesses_.emplace_back(subProcessPSet,
                                 topLevelParameterSet,
                                 preg_,
                                 branchIDListHelper(),
                                 *processBlockHelper_,
                                 *thinnedAssociationsHelper_,
                                 *subProcessParentageHelper_,
                                 esController,
                                 *items.actReg_,
                                 newToken,
                                 iLegacy,
                                 preallocConfig,
                                 &processContext_,
                                 typeResolverMaker);
    }
  }

  SubProcess::~SubProcess() {}

  std::vector<ModuleProcessName> SubProcess::keepOnlyConsumedUnscheduledModules(bool deleteModules) {
    schedule_->convertCurrentProcessAlias(processConfiguration_->processName());
    pathsAndConsumesOfModules_.initialize(schedule_.get(), preg_);

    // Note: all these may throw
    checkForModuleDependencyCorrectness(pathsAndConsumesOfModules_, false);

    // Consumes information from the child SubProcesses
    std::vector<ModuleProcessName> consumedByChildren;
    for_all(subProcesses_, [&consumedByChildren, deleteModules](auto& subProcess) {
      auto c = subProcess.keepOnlyConsumedUnscheduledModules(deleteModules);
      if (consumedByChildren.empty()) {
        std::swap(consumedByChildren, c);
      } else if (not c.empty()) {
        std::vector<ModuleProcessName> tmp;
        tmp.reserve(consumedByChildren.size() + c.size());
        std::merge(consumedByChildren.begin(), consumedByChildren.end(), c.begin(), c.end(), std::back_inserter(tmp));
        std::swap(consumedByChildren, tmp);
      }
    });

    // Non-consumed unscheduled modules in this SubProcess, take into account of the consumes from child SubProcesses
    if (deleteModules) {
      if (auto const unusedModules = nonConsumedUnscheduledModules(pathsAndConsumesOfModules_, consumedByChildren);
          not unusedModules.empty()) {
        pathsAndConsumesOfModules_.removeModules(unusedModules);

        edm::LogInfo("DeleteModules").log([&unusedModules, this](auto& l) {
          l << "Following modules are not in any Path or EndPath, nor is their output consumed by any other module, "
               "and "
               "therefore they are deleted from SubProcess "
            << processConfiguration_->processName() << " before beginJob transition.";
          for (auto const& description : unusedModules) {
            l << "\n " << description->moduleLabel();
          }
        });
        for (auto const& description : unusedModules) {
          schedule_->deleteModule(description->moduleLabel(), actReg_.get());
        }
      }
    }

    // Products possibly consumed from the parent (Sub)Process
    for (auto const& description : pathsAndConsumesOfModules_.allModules()) {
      for (auto const& dep :
           pathsAndConsumesOfModules_.modulesInPreviousProcessesWhoseProductsAreConsumedBy(description->id())) {
        auto it = std::lower_bound(consumedByChildren.begin(),
                                   consumedByChildren.end(),
                                   ModuleProcessName{dep.moduleLabel(), dep.processName()});
        consumedByChildren.emplace(it, dep.moduleLabel(), dep.processName());
      }
    }
    return consumedByChildren;
  }

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
    actReg_->preBeginJobSignal_(pathsAndConsumesOfModules_, processContext_);
    schedule_->beginJob(*preg_, esp_->recordsToProxyIndices(), *processBlockHelper_);
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
        principal.eventToProcessBlockIndexes(),
        *(principal.productProvenanceRetrieverPtr()),  //NOTE: this transfers the per product provenance
        principal.reader(),
        deepCopyRetriever);
    ep.setLuminosityBlockPrincipal(inUseLumiPrincipals_[principal.luminosityBlockPrincipal().index()].get());
    propagateProducts(InEvent, principal, ep);

    using namespace edm::waiting_task;
    chain::first([&](auto nextTask) {
      EventTransitionInfo info(ep, *((*iEventSetupImpls)[esp_->subProcessIndex()]));
      schedule_->processOneEventAsync(std::move(nextTask), ep.streamID().value(), info, serviceToken_);
    }) | chain::ifThen(not subProcesses_.empty(), [this, &ep, iEventSetupImpls](auto nextTask) {
      for (auto& subProcess : boost::adaptors::reverse(subProcesses_)) {
        subProcess.doEventAsync(nextTask, ep, iEventSetupImpls);
      }
    }) | chain::then([&ep](std::exception_ptr const* iPtr, auto nextTask) {
      ep.clearEventPrincipal();
      if (iPtr) {
        nextTask.doneWaiting(*iPtr);
      }
    }) | chain::runLast(std::move(iHolder));
  }

  template <>
  void SubProcess::doBeginProcessBlockAsync<OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>>(
      WaitingTaskHolder iHolder, ProcessBlockTransitionInfo const& iTransitionInfo, bool cleaningUpAfterException) {
    ServiceRegistry::Operate operate(serviceToken_);

    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.inputProcessBlockPrincipal();
    ProcessBlockPrincipal const& parentPrincipal = iTransitionInfo.principal();
    processBlockPrincipal.fillProcessBlockPrincipal(parentPrincipal.processName(), parentPrincipal.reader());
    propagateProducts(InProcess, parentPrincipal, processBlockPrincipal);

    ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
    using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>;
    beginGlobalTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
  }

  template <>
  void SubProcess::doBeginProcessBlockAsync<OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin>>(
      WaitingTaskHolder iHolder, ProcessBlockTransitionInfo const& iTransitionInfo, bool) {
    ServiceRegistry::Operate operate(serviceToken_);

    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal();
    ProcessBlockPrincipal const& parentPrincipal = iTransitionInfo.principal();
    processBlockPrincipal.fillProcessBlockPrincipal(processConfiguration_->processName());
    propagateProducts(InProcess, parentPrincipal, processBlockPrincipal);

    ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);
    using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin>;
    beginGlobalTransitionAsync<Traits>(std::move(iHolder), *schedule_, transitionInfo, serviceToken_, subProcesses_);
  }

  void SubProcess::doEndProcessBlockAsync(WaitingTaskHolder iHolder,
                                          ProcessBlockTransitionInfo const& iTransitionInfo,
                                          bool cleaningUpAfterException) {
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal();
    ProcessBlockPrincipal const& parentPrincipal = iTransitionInfo.principal();
    propagateProducts(InProcess, parentPrincipal, processBlockPrincipal);

    ProcessBlockTransitionInfo transitionInfo(processBlockPrincipal);

    if (parentProducedProductIsKept(parentPrincipal, processBlockPrincipal)) {
      ProcessBlockPrincipal& inputProcessBlockPrincipal = principalCache_.inputProcessBlockPrincipal();
      inputProcessBlockPrincipal.fillProcessBlockPrincipal(parentPrincipal.processName(), parentPrincipal.reader());
      propagateProducts(InProcess, parentPrincipal, inputProcessBlockPrincipal);
      ProcessBlockTransitionInfo inputTransitionInfo(inputProcessBlockPrincipal);

      using namespace edm::waiting_task;
      chain::first([&](const std::exception_ptr*, auto nextTask) {
        using TraitsInput = OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>;
        beginGlobalTransitionAsync<TraitsInput>(std::move(nextTask),
                                                *schedule_,
                                                inputTransitionInfo,
                                                serviceToken_,
                                                subProcesses_,
                                                cleaningUpAfterException);
      }) | chain::then([this](auto nextTask) { writeProcessBlockAsync(nextTask, ProcessBlockType::Input); }) |
          chain::then([this, info = transitionInfo, cleaningUpAfterException](std::exception_ptr const* iPtr,
                                                                              auto nextTask) mutable {
            ProcessBlockPrincipal& inputProcessBlockPrincipal = principalCache_.inputProcessBlockPrincipal();
            inputProcessBlockPrincipal.clearPrincipal();
            for (auto& s : subProcesses_) {
              s.clearProcessBlockPrincipal(ProcessBlockType::Input);
            }
            if (iPtr) {
              nextTask.doneWaiting(*iPtr);
            } else {
              using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>;
              endGlobalTransitionAsync<Traits>(
                  std::move(nextTask), *schedule_, info, serviceToken_, subProcesses_, cleaningUpAfterException);
            }
          }) |
          chain::runLast(std::move(iHolder));
    } else {
      using Traits = OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>;
      endGlobalTransitionAsync<Traits>(
          std::move(iHolder), *schedule_, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
    }
  }

  void SubProcess::doBeginRunAsync(WaitingTaskHolder iHolder, RunTransitionInfo const& iTransitionInfo) {
    ServiceRegistry::Operate operate(serviceToken_);

    RunPrincipal const& parentPrincipal = iTransitionInfo.principal();
    auto aux = parentPrincipal.aux();
    aux.setProcessHistoryID(parentPrincipal.processHistoryID());
    auto rpp = principalCache_.getAvailableRunPrincipalPtr();
    rpp->setAux(aux);
    auto& processHistoryRegistry = processHistoryRegistries_[historyRunOffset_ + parentPrincipal.index()];
    inUseRunPrincipals_[parentPrincipal.index()] = rpp;
    processHistoryRegistry.registerProcessHistory(parentPrincipal.processHistory());
    rpp->fillRunPrincipal(processHistoryRegistry, parentPrincipal.reader());

    RunPrincipal& rp = *rpp;
    propagateProducts(InRun, parentPrincipal, rp);

    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    RunTransitionInfo transitionInfo(rp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>;
    beginGlobalTransitionAsync<Traits>(std::move(iHolder), *schedule_, transitionInfo, serviceToken_, subProcesses_);
  }

  void SubProcess::doEndRunAsync(WaitingTaskHolder iHolder,
                                 RunTransitionInfo const& iTransitionInfo,
                                 bool cleaningUpAfterException) {
    RunPrincipal const& parentPrincipal = iTransitionInfo.principal();
    RunPrincipal& rp = *inUseRunPrincipals_[parentPrincipal.index()];
    propagateProducts(InRun, parentPrincipal, rp);

    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    RunTransitionInfo transitionInfo(rp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    using Traits = OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>;
    endGlobalTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
  }

  void SubProcess::writeProcessBlockAsync(edm::WaitingTaskHolder task, ProcessBlockType processBlockType) {
    using namespace edm::waiting_task;
    chain::first([&](std::exception_ptr const*, auto nextTask) {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->writeProcessBlockAsync(
          nextTask, principalCache_.processBlockPrincipal(processBlockType), &processContext_, actReg_.get());
    }) | chain::ifThen(not subProcesses_.empty(), [this, processBlockType](auto nextTask) {
      ServiceRegistry::Operate operate(serviceToken_);
      for (auto& s : subProcesses_) {
        s.writeProcessBlockAsync(nextTask, processBlockType);
      }
    }) | chain::runLast(std::move(task));
  }

  void SubProcess::writeRunAsync(edm::WaitingTaskHolder task,
                                 RunPrincipal const& principal,
                                 MergeableRunProductMetadata const* mergeableRunProductMetadata) {
    using namespace edm::waiting_task;

    auto rp = inUseRunPrincipals_[principal.index()];
    chain::first([&](std::exception_ptr const*, auto nextTask) {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->writeRunAsync(nextTask, *rp, &processContext_, actReg_.get(), mergeableRunProductMetadata);
    }) | chain::ifThen(not subProcesses_.empty(), [this, rp, mergeableRunProductMetadata](auto nextTask) {
      ServiceRegistry::Operate operateWriteRun(serviceToken_);
      for (auto& s : subProcesses_) {
        s.writeRunAsync(nextTask, *rp, mergeableRunProductMetadata);
      }
    }) | chain::runLast(std::move(task));
  }

  void SubProcess::clearRunPrincipal(RunPrincipal& parentPrincipal) {
    //release from list but stay around till end of routine
    auto rp = std::move(inUseRunPrincipals_[parentPrincipal.index()]);
    for (auto& s : subProcesses_) {
      s.clearRunPrincipal(*rp);
    }
    rp->clearPrincipal();
  }

  void SubProcess::clearProcessBlockPrincipal(ProcessBlockType processBlockType) {
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.processBlockPrincipal(processBlockType);
    processBlockPrincipal.clearPrincipal();
    for (auto& s : subProcesses_) {
      s.clearProcessBlockPrincipal(processBlockType);
    }
  }

  void SubProcess::doBeginLuminosityBlockAsync(WaitingTaskHolder iHolder, LumiTransitionInfo const& iTransitionInfo) {
    ServiceRegistry::Operate operate(serviceToken_);

    LuminosityBlockPrincipal const& parentPrincipal = iTransitionInfo.principal();
    auto aux = parentPrincipal.aux();
    aux.setProcessHistoryID(parentPrincipal.processHistoryID());
    auto lbpp = principalCache_.getAvailableLumiPrincipalPtr();
    lbpp->setAux(aux);
    auto& processHistoryRegistry = processHistoryRegistries_[historyLumiOffset_ + lbpp->index()];
    inUseLumiPrincipals_[parentPrincipal.index()] = lbpp;
    processHistoryRegistry.registerProcessHistory(parentPrincipal.processHistory());
    lbpp->fillLuminosityBlockPrincipal(&parentPrincipal.processHistory(), parentPrincipal.reader());
    lbpp->setRunPrincipal(inUseRunPrincipals_[parentPrincipal.runPrincipal().index()]);
    LuminosityBlockPrincipal& lbp = *lbpp;
    propagateProducts(InLumi, parentPrincipal, lbp);

    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    LumiTransitionInfo transitionInfo(lbp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>;
    beginGlobalTransitionAsync<Traits>(std::move(iHolder), *schedule_, transitionInfo, serviceToken_, subProcesses_);
  }

  void SubProcess::doEndLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                             LumiTransitionInfo const& iTransitionInfo,
                                             bool cleaningUpAfterException) {
    LuminosityBlockPrincipal const& parentPrincipal = iTransitionInfo.principal();
    LuminosityBlockPrincipal& lbp = *inUseLumiPrincipals_[parentPrincipal.index()];
    propagateProducts(InLumi, parentPrincipal, lbp);

    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    LumiTransitionInfo transitionInfo(lbp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>;
    endGlobalTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
  }

  void SubProcess::writeLumiAsync(WaitingTaskHolder task, LuminosityBlockPrincipal& principal) {
    using namespace edm::waiting_task;

    auto l = inUseLumiPrincipals_[principal.index()];
    chain::first([&](std::exception_ptr const*, auto nextTask) {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->writeLumiAsync(nextTask, *l, &processContext_, actReg_.get());
    }) | chain::ifThen(not subProcesses_.empty(), [this, l](auto nextTask) {
      ServiceRegistry::Operate operateWriteLumi(serviceToken_);
      for (auto& s : subProcesses_) {
        s.writeLumiAsync(nextTask, *l);
      }
    }) | chain::runLast(std::move(task));
  }

  void SubProcess::clearLumiPrincipal(LuminosityBlockPrincipal& principal) {
    //release from list but stay around till end of routine
    auto lb = std::move(inUseLumiPrincipals_[principal.index()]);
    for (auto& s : subProcesses_) {
      s.clearLumiPrincipal(*lb);
    }
    lb->setRunPrincipal(std::shared_ptr<RunPrincipal>());
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
                                         RunTransitionInfo const& iTransitionInfo) {
    using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>;

    RunPrincipal const& parentPrincipal = iTransitionInfo.principal();
    RunPrincipal& rp = *inUseRunPrincipals_[parentPrincipal.index()];

    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    RunTransitionInfo transitionInfo(rp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    beginStreamTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, id, transitionInfo, serviceToken_, subProcesses_);
  }

  void SubProcess::doStreamEndRunAsync(WaitingTaskHolder iHolder,
                                       unsigned int id,
                                       RunTransitionInfo const& iTransitionInfo,
                                       bool cleaningUpAfterException) {
    using Traits = OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>;

    RunPrincipal const& parentPrincipal = iTransitionInfo.principal();
    RunPrincipal& rp = *inUseRunPrincipals_[parentPrincipal.index()];

    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    RunTransitionInfo transitionInfo(rp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    endStreamTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, id, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
  }

  void SubProcess::doStreamBeginLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                                     unsigned int id,
                                                     LumiTransitionInfo const& iTransitionInfo) {
    using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>;

    LuminosityBlockPrincipal& lbp = *inUseLumiPrincipals_[iTransitionInfo.principal().index()];
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    LumiTransitionInfo transitionInfo(lbp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    beginStreamTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, id, transitionInfo, serviceToken_, subProcesses_);
  }

  void SubProcess::doStreamEndLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                                   unsigned int id,
                                                   LumiTransitionInfo const& iTransitionInfo,
                                                   bool cleaningUpAfterException) {
    LuminosityBlockPrincipal& lbp = *inUseLumiPrincipals_[iTransitionInfo.principal().index()];
    using Traits = OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>;
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls = iTransitionInfo.eventSetupImpls();
    LumiTransitionInfo transitionInfo(lbp, *((*eventSetupImpls)[esp_->subProcessIndex()]), eventSetupImpls);
    endStreamTransitionAsync<Traits>(
        std::move(iHolder), *schedule_, id, transitionInfo, serviceToken_, subProcesses_, cleaningUpAfterException);
  }

  void SubProcess::propagateProducts(BranchType type, Principal const& parentPrincipal, Principal& principal) const {
    SelectedProducts const& keptVector = keptProducts()[type];
    for (auto const& item : keptVector) {
      BranchDescription const& desc = *item.first;
      ProductResolverBase const* parentProductResolver = parentPrincipal.getProductResolver(desc.branchID());
      if (parentProductResolver != nullptr) {
        ProductResolverBase* productResolver = principal.getModifiableProductResolver(desc.branchID());
        if (productResolver != nullptr) {
          //Propagate the per event(run)(lumi)(processBlock) data for this product to the subprocess.
          productResolver->connectTo(*parentProductResolver, &parentPrincipal);
        }
      }
    }
  }

  bool SubProcess::parentProducedProductIsKept(Principal const& parentPrincipal, Principal& principal) const {
    SelectedProducts const& keptVector = keptProducts()[InProcess];
    for (auto const& item : keptVector) {
      BranchDescription const& desc = *item.first;
      assert(desc.branchType() == InProcess);
      ProductResolverBase const* parentProductResolver = parentPrincipal.getProductResolver(desc.branchID());
      if (parentProductResolver != nullptr) {
        ProductResolverBase* productResolver = principal.getModifiableProductResolver(desc.branchID());
        if (productResolver != nullptr) {
          if (parentProductResolver->branchDescription().produced()) {
            return true;
          }
        }
      }
    }
    return false;
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
