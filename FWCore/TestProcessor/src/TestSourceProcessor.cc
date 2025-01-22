#include "FWCore/TestProcessor/interface/TestSourceProcessor.h"

#include "FWCore/Framework/interface/ScheduleItems.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/maker/InputSourceFactory.h"
#include "FWCore/Framework/interface/ProductResolversFactory.h"

#include "FWCore/Common/interface/ProcessBlockHelper.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/ParameterSetReader/interface/ProcessDescImpl.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"

#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include "DataFormats/Provenance/interface/ParentageRegistry.h"

#include "oneTimeInitialization.h"

namespace {
  using namespace edm;

  std::string name(edm::InputSource::ItemType iType) {
    switch (iType) {
      case edm::InputSource::ItemType::IsEvent:
        return "Event";
      case edm::InputSource::ItemType::IsFile:
        return "File";
      case edm::InputSource::ItemType::IsLumi:
        return "LuminosityBlock";
      case edm::InputSource::ItemType::IsRepeat:
        return "Repeat";
      case edm::InputSource::ItemType::IsStop:
        return "Stop";
      case edm::InputSource::ItemType::IsRun:
        return "Run";
      case edm::InputSource::ItemType::IsSynchronize:
        return "Synchronize";
      case edm::InputSource::ItemType::IsInvalid:
        return "Invalid";
    }
    return "Invalid";
  }
  // ---------------------------------------------------------------
  std::unique_ptr<InputSource> makeInput(unsigned int moduleIndex,
                                         ParameterSet& params,
                                         std::shared_ptr<SignallingProductRegistry> preg,
                                         std::shared_ptr<BranchIDListHelper> branchIDListHelper,
                                         std::shared_ptr<ProcessBlockHelper> const& processBlockHelper,
                                         std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper,
                                         std::shared_ptr<ActivityRegistry> areg,
                                         std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                         PreallocationConfiguration const& allocations) {
    ParameterSet* main_input = params.getPSetForUpdate("@main_input");
    if (main_input == nullptr) {
      throw Exception(errors::Configuration)
          << "There must be exactly one source in the configuration.\n"
          << "It is missing (or there are sufficient syntax errors such that it is not recognized as the source)\n";
    }

    std::string modtype(main_input->getParameter<std::string>("@module_type"));

    std::unique_ptr<ParameterSetDescriptionFillerBase> filler(
        ParameterSetDescriptionFillerPluginFactory::get()->create(modtype));
    ConfigurationDescriptions descriptions(filler->baseType(), modtype);
    filler->fill(descriptions);

    descriptions.validate(*main_input, std::string("source"));

    main_input->registerIt();

    // Fill in "ModuleDescription", in case the input source produces
    // any EDProducts, which would be registered in the ProductRegistry.
    // Also fill in the process history item for this process.
    // There is no module label for the unnamed input source, so
    // just use "source".
    // Only the tracked parameters belong in the process configuration.
    ModuleDescription md(main_input->id(),
                         main_input->getParameter<std::string>("@module_type"),
                         "source",
                         processConfiguration.get(),
                         moduleIndex);

    InputSourceDescription isdesc(
        md, preg, branchIDListHelper, processBlockHelper, thinnedAssociationsHelper, areg, -1, -1, 0, allocations);

    return std::unique_ptr<InputSource>(
        InputSourceFactory::get()->makeInputSource(*main_input, *preg, isdesc).release());
  }
}  // namespace

namespace edm::test {

  TestSourceProcessor::TestSourceProcessor(std::string const& iConfig, ServiceToken iToken)
      : globalControl_(oneapi::tbb::global_control::max_allowed_parallelism, 1),
        arena_(1),
        historyAppender_(std::make_unique<HistoryAppender>()) {
    //Setup various singletons
    (void)testprocessor::oneTimeInitialization();

    ProcessDescImpl desc(iConfig, false);

    auto psetPtr = desc.parameterSet();

    validateTopLevelParameterSets(psetPtr.get());

    auto procDesc = desc.processDesc();
    // Now do general initialization
    ScheduleItems items;

    //initialize the services
    auto& serviceSets = procDesc->getServicesPSets();
    ServiceToken token = items.initServices(serviceSets, *psetPtr, iToken, serviceregistry::kOverlapIsError, true);
    serviceToken_ = items.addCPRandTNS(*psetPtr, token);

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    // intialize miscellaneous items
    items.initMisc(*psetPtr);

    auto nThreads = 1U;
    auto nStreams = 1U;
    auto nConcurrentLumis = 1U;
    auto nConcurrentRuns = 1U;
    preallocations_ = PreallocationConfiguration{nThreads, nStreams, nConcurrentLumis, nConcurrentRuns};

    processBlockHelper_ = std::make_shared<ProcessBlockHelper>();

    {
      // initialize the input source
      auto tempReg = std::make_shared<SignallingProductRegistry>();
      auto sourceID = ModuleDescription::getUniqueID();

      ServiceRegistry::Operate operate(serviceToken_);
      source_ = makeInput(sourceID,
                          *psetPtr,
                          /*items.preg(),*/ tempReg,
                          items.branchIDListHelper(),
                          processBlockHelper_,
                          items.thinnedAssociationsHelper(),
                          items.actReg_,
                          items.processConfiguration(),
                          preallocations_);
      items.preg()->addFromInput(*tempReg);
      source_->switchTo(items.preg());
    }

    actReg_ = items.actReg_;
    branchIDListHelper_ = items.branchIDListHelper();
    thinnedAssociationsHelper_ = items.thinnedAssociationsHelper();
    processConfiguration_ = items.processConfiguration();

    processContext_.setProcessConfiguration(processConfiguration_.get());
    preg_ = items.preg();
    principalCache_.setNumberOfConcurrentPrincipals(preallocations_);

    preg_->setFrozen();
    mergeableRunProductProcesses_.setProcessesWithMergeableRunProducts(*preg_);

    for (unsigned int index = 0; index < preallocations_.numberOfStreams(); ++index) {
      // Reusable event principal
      auto ep = std::make_shared<EventPrincipal>(preg_,
                                                 edm::productResolversFactory::makePrimary,
                                                 branchIDListHelper_,
                                                 thinnedAssociationsHelper_,
                                                 *processConfiguration_,
                                                 historyAppender_.get(),
                                                 index);
      principalCache_.insert(std::move(ep));
    }
    for (unsigned int index = 0; index < preallocations_.numberOfRuns(); ++index) {
      auto rp = std::make_unique<RunPrincipal>(preg_,
                                               edm::productResolversFactory::makePrimary,
                                               *processConfiguration_,
                                               historyAppender_.get(),
                                               index,
                                               &mergeableRunProductProcesses_);
      principalCache_.insert(std::move(rp));
    }
    for (unsigned int index = 0; index < preallocations_.numberOfLuminosityBlocks(); ++index) {
      auto lp = std::make_unique<LuminosityBlockPrincipal>(
          preg_, edm::productResolversFactory::makePrimary, *processConfiguration_, historyAppender_.get(), index);
      principalCache_.insert(std::move(lp));
    }
    {
      auto pb = std::make_unique<ProcessBlockPrincipal>(
          preg_, edm::productResolversFactory::makePrimary, *processConfiguration_);
      principalCache_.insert(std::move(pb));
    }

    source_->doBeginJob();
  }

  TestSourceProcessor::~TestSourceProcessor() {
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);
    try {
      source_.reset();
    } catch (std::exception const& iExcept) {
      std::cerr << " caught exception while destroying TestSourceProcessor\n" << iExcept.what();
    }
  }

  edm::InputSource::ItemTypeInfo TestSourceProcessor::findNextTransition() {
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    lastTransition_ = source_->nextItemType();
    return lastTransition_;
  }

  std::shared_ptr<FileBlock> TestSourceProcessor::openFile() {
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    size_t size = preg_->size();
    fb_ = source_->readFile();
    if (size < preg_->size()) {
      principalCache_.adjustIndexesAfterProductRegistryAddition();
    }
    principalCache_.adjustEventsToNewProductRegistry(preg_);

    source_->fillProcessBlockHelper();
    ProcessBlockPrincipal& processBlockPrincipal = principalCache_.inputProcessBlockPrincipal();
    while (source_->nextProcessBlock(processBlockPrincipal)) {
      source_->readProcessBlock(processBlockPrincipal);
      processBlockPrincipal.clearPrincipal();
    }
    return fb_;
  }
  void TestSourceProcessor::closeFile(std::shared_ptr<FileBlock> iBlock) {
    if (iBlock.get() != fb_.get()) {
      throw cms::Exception("IncorrectFileBlock")
          << "closeFile given a FileBlock that does not correspond to the one returned by openFile";
    }
    if (fb_) {
      //make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      source_->closeFile(fb_.get(), false);
    }
  }

  edm::test::RunFromSource TestSourceProcessor::readRun() {
    if (lastTransition_.itemType() != edm::InputSource::ItemType::IsRun) {
      throw cms::Exception("NotARun") << "The last transition is " << name(lastTransition_.itemType()) << " not a Run";
    }
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    //NOTE: should probably handle merging as well
    runPrincipal_ = principalCache_.getAvailableRunPrincipalPtr();
    runPrincipal_->setAux(*source_->runAuxiliary());
    source_->readRun(*runPrincipal_, *historyAppender_);

    return edm::test::RunFromSource(runPrincipal_, serviceToken_);
  }

  edm::test::LuminosityBlockFromSource TestSourceProcessor::readLuminosityBlock() {
    if (lastTransition_.itemType() != edm::InputSource::ItemType::IsLumi) {
      throw cms::Exception("NotALuminosityBlock")
          << "The last transition is " << name(lastTransition_.itemType()) << " not a LuminosityBlock";
    }

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    lumiPrincipal_ = principalCache_.getAvailableLumiPrincipalPtr();
    assert(lumiPrincipal_);
    lumiPrincipal_->setAux(*source_->luminosityBlockAuxiliary());
    source_->readLuminosityBlock(*lumiPrincipal_, *historyAppender_);

    return edm::test::LuminosityBlockFromSource(lumiPrincipal_, serviceToken_);
  }

  edm::test::EventFromSource TestSourceProcessor::readEvent() {
    if (lastTransition_.itemType() != edm::InputSource::ItemType::IsEvent) {
      throw cms::Exception("NotAnEvent") << "The last transition is " << name(lastTransition_.itemType())
                                         << " not a Event";
    }
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    auto& event = principalCache_.eventPrincipal(0);
    StreamContext streamContext(event.streamID(), &processContext_);

    source_->readEvent(event, streamContext);

    return edm::test::EventFromSource(event, serviceToken_);
  }
}  // namespace edm::test