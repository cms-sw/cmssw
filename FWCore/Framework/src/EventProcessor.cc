
#include "FWCore/Framework/interface/EventProcessor.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

#include "FWCore/Framework/interface/CommonParams.h"
#include "FWCore/Framework/interface/EDLooperBase.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/MessageReceiverForSource.h"
#include "FWCore/Framework/interface/ModuleChanger.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/src/Breakpoints.h"
#include "FWCore/Framework/src/EPStates.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/src/InputSourceFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/IllegalParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "MessageForSource.h"
#include "MessageForParent.h"

#include "boost/bind.hpp"
#include "boost/thread/xtime.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <utility>
#include <sstream>

#include <sys/ipc.h>
#include <sys/msg.h>

//Used for forking
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/fcntl.h>
#include <unistd.h>

//Used for CPU affinity
#ifndef __APPLE__
#include <sched.h>
#endif

//Needed for introspection
#include "Cintex/Cintex.h"

namespace edm {

  // ---------------------------------------------------------------
  std::unique_ptr<InputSource>
  makeInput(ParameterSet& params,
            CommonParams const& common,
            ProductRegistry& preg,
            boost::shared_ptr<BranchIDListHelper> branchIDListHelper,
            boost::shared_ptr<ActivityRegistry> areg,
            boost::shared_ptr<ProcessConfiguration const> processConfiguration) {
    ParameterSet* main_input = params.getPSetForUpdate("@main_input");
    if(main_input == 0) {
      throw Exception(errors::Configuration)
        << "There must be exactly one source in the configuration.\n"
        << "It is missing (or there are sufficient syntax errors such that it is not recognized as the source)\n";
    }

    std::string modtype(main_input->getParameter<std::string>("@module_type"));

    std::auto_ptr<ParameterSetDescriptionFillerBase> filler(
      ParameterSetDescriptionFillerPluginFactory::get()->create(modtype));
    ConfigurationDescriptions descriptions(filler->baseType());
    filler->fill(descriptions);

    try {
      try {
        descriptions.validate(*main_input, std::string("source"));
      }
      catch (cms::Exception& e) { throw; }
      catch (std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch (std::string& s) { convertException::stringToEDM(s); }
      catch (char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch (cms::Exception & iException) {
      std::ostringstream ost;
      ost << "Validating configuration of input source of type " << modtype;
      iException.addContext(ost.str());
      throw;
    }

    main_input->registerIt();

    // Fill in "ModuleDescription", in case the input source produces
    // any EDproducts, which would be registered in the ProductRegistry.
    // Also fill in the process history item for this process.
    // There is no module label for the unnamed input source, so
    // just use "source".
    // Only the tracked parameters belong in the process configuration.
    ModuleDescription md(main_input->id(),
                         main_input->getParameter<std::string>("@module_type"),
                         "source",
                         processConfiguration.get(),
                         ModuleDescription::getUniqueID());

    InputSourceDescription isdesc(md, preg, branchIDListHelper, areg, common.maxEventsInput_, common.maxLumisInput_);
    areg->preSourceConstructionSignal_(md);
    std::unique_ptr<InputSource> input;
    try {
      try {
        input = std::unique_ptr<InputSource>(InputSourceFactory::get()->makeInputSource(*main_input, isdesc).release());
      }
      catch (cms::Exception& e) { throw; }
      catch (std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch (std::string& s) { convertException::stringToEDM(s); }
      catch (char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch (cms::Exception& iException) {
      areg->postSourceConstructionSignal_(md);
      std::ostringstream ost;
      ost << "Constructing input source of type " << modtype;
      iException.addContext(ost.str());
      throw;
    }
    areg->postSourceConstructionSignal_(md);
    return input;
  }

  // ---------------------------------------------------------------
  boost::shared_ptr<EDLooperBase>
  fillLooper(eventsetup::EventSetupsController& esController,
             eventsetup::EventSetupProvider& cp,
                         ParameterSet& params) {
    boost::shared_ptr<EDLooperBase> vLooper;

    std::vector<std::string> loopers = params.getParameter<std::vector<std::string> >("@all_loopers");

    if(loopers.size() == 0) {
       return vLooper;
    }

    assert(1 == loopers.size());

    for(std::vector<std::string>::iterator itName = loopers.begin(), itNameEnd = loopers.end();
        itName != itNameEnd;
        ++itName) {

      ParameterSet* providerPSet = params.getPSetForUpdate(*itName);
      providerPSet->registerIt();
      vLooper = eventsetup::LooperFactory::get()->addTo(esController,
                                                        cp,
                                                        *providerPSet);
      }
      return vLooper;

  }

  // ---------------------------------------------------------------
  EventProcessor::EventProcessor(std::string const& config,
                                ServiceToken const& iToken,
                                serviceregistry::ServiceLegacy iLegacy,
                                std::vector<std::string> const& defaultServices,
                                std::vector<std::string> const& forcedServices) :
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcess_(),
    historyAppender_(new HistoryAppender),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    stateMachineWasInErrorState_(false),
    fileMode_(),
    emptyRunLumiMode_(),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    numberOfForkedChildren_(0),
    numberOfSequentialEventsPerChild_(1),
    setCpuAffinity_(false),
    eventSetupDataToExcludeFromPrefetching_() {
    boost::shared_ptr<ParameterSet> parameterSet = PythonProcessDesc(config).parameterSet();
    boost::shared_ptr<ProcessDesc> processDesc(new ProcessDesc(parameterSet));
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, iToken, iLegacy);
  }

  EventProcessor::EventProcessor(std::string const& config,
                                std::vector<std::string> const& defaultServices,
                                std::vector<std::string> const& forcedServices) :
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcess_(),
    historyAppender_(new HistoryAppender),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    stateMachineWasInErrorState_(false),
    fileMode_(),
    emptyRunLumiMode_(),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    numberOfForkedChildren_(0),
    numberOfSequentialEventsPerChild_(1),
    setCpuAffinity_(false),
  asyncStopRequestedWhileProcessingEvents_(false),
  nextItemTypeFromProcessingEvents_(InputSource::IsEvent),
    eventSetupDataToExcludeFromPrefetching_()
  {
    boost::shared_ptr<ParameterSet> parameterSet = PythonProcessDesc(config).parameterSet();
    boost::shared_ptr<ProcessDesc> processDesc(new ProcessDesc(parameterSet));
    processDesc->addServices(defaultServices, forcedServices);
    init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
  }

  EventProcessor::EventProcessor(boost::shared_ptr<ProcessDesc>& processDesc,
                 ServiceToken const& token,
                 serviceregistry::ServiceLegacy legacy) :
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcess_(),
    historyAppender_(new HistoryAppender),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    stateMachineWasInErrorState_(false),
    fileMode_(),
    emptyRunLumiMode_(),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    numberOfForkedChildren_(0),
    numberOfSequentialEventsPerChild_(1),
    setCpuAffinity_(false),
    asyncStopRequestedWhileProcessingEvents_(false),
    nextItemTypeFromProcessingEvents_(InputSource::IsEvent),
    eventSetupDataToExcludeFromPrefetching_()
  {
    init(processDesc, token, legacy);
  }


  EventProcessor::EventProcessor(std::string const& config, bool isPython):
    actReg_(),
    preg_(),
    branchIDListHelper_(),
    serviceToken_(),
    input_(),
    espController_(new eventsetup::EventSetupsController),
    esp_(),
    act_table_(),
    processConfiguration_(),
    schedule_(),
    subProcess_(),
    historyAppender_(new HistoryAppender),
    my_sig_num_(getSigNum()),
    fb_(),
    looper_(),
    principalCache_(),
    beginJobCalled_(false),
    shouldWeStop_(false),
    stateMachineWasInErrorState_(false),
    fileMode_(),
    emptyRunLumiMode_(),
    exceptionMessageFiles_(),
    exceptionMessageRuns_(),
    exceptionMessageLumis_(),
    alreadyHandlingException_(false),
    forceLooperToEnd_(false),
    looperBeginJobRun_(false),
    forceESCacheClearOnNewRun_(false),
    numberOfForkedChildren_(0),
    numberOfSequentialEventsPerChild_(1),
    setCpuAffinity_(false),
    asyncStopRequestedWhileProcessingEvents_(false),
    nextItemTypeFromProcessingEvents_(InputSource::IsEvent),
    eventSetupDataToExcludeFromPrefetching_()
{
    if(isPython) {
      boost::shared_ptr<ParameterSet> parameterSet = PythonProcessDesc(config).parameterSet();
      boost::shared_ptr<ProcessDesc> processDesc(new ProcessDesc(parameterSet));
      init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
    }
    else {
      boost::shared_ptr<ProcessDesc> processDesc(new ProcessDesc(config));
      init(processDesc, ServiceToken(), serviceregistry::kOverlapIsError);
    }
  }

  void
  EventProcessor::init(boost::shared_ptr<ProcessDesc>& processDesc,
                        ServiceToken const& iToken,
                        serviceregistry::ServiceLegacy iLegacy) {

    //std::cerr << processDesc->dump() << std::endl;
   
    ROOT::Cintex::Cintex::Enable();

    // register the empty parentage vector , once and for all
    ParentageRegistry::instance()->insertMapped(Parentage());

    // register the empty parameter set, once and for all.
    ParameterSet().registerIt();

    boost::shared_ptr<ParameterSet> parameterSet = processDesc->getProcessPSet();
    //std::cerr << parameterSet->dump() << std::endl;

    // If there is a subprocess, pop the subprocess parameter set out of the process parameter set
    boost::shared_ptr<ParameterSet> subProcessParameterSet(popSubProcessParameterSet(*parameterSet).release());

    // Now set some parameters specific to the main process.
    ParameterSet const& optionsPset(parameterSet->getUntrackedParameterSet("options", ParameterSet()));
    fileMode_ = optionsPset.getUntrackedParameter<std::string>("fileMode", "");
    emptyRunLumiMode_ = optionsPset.getUntrackedParameter<std::string>("emptyRunLumiMode", "");
    forceESCacheClearOnNewRun_ = optionsPset.getUntrackedParameter<bool>("forceEventSetupCacheClearOnNewRun", false);
    //threading
    unsigned int nThreads=1;
    if(optionsPset.existsAs<unsigned int>("numberOfThreads",false)) {
      nThreads = optionsPset.getUntrackedParameter<unsigned int>("numberOfThreads");
      if(nThreads == 0) {
        nThreads = 1;
      }
    }
    /* TODO: when we support having each stream run in a different thread use this default
    unsigned int nStreams =nThreads;
     */
    unsigned int nStreams =1;
    if(optionsPset.existsAs<unsigned int>("numberOfStreams",false)) {
      nStreams = optionsPset.getUntrackedParameter<unsigned int>("numberOfStreams");
      if(nStreams==0) {
        nStreams = nThreads;
      }
    }
    /*
    bool nRunsSet = false;
     */
    unsigned int nConcurrentRuns =1;
    /*
    if(nRunsSet = optionsPset.existsAs<unsigned int>("numberOfConcurrentRuns",false)) {
    nConcurrentRuns = optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentRuns");
    }
     */
    unsigned int nConcurrentLumis =1;
    /*
    if(optionsPset.existsAs<unsigned int>("numberOfConcurrentLuminosityBlocks",false)) {
    nConcurrentLumis = optionsPset.getUntrackedParameter<unsigned int>("numberOfConcurrentLuminosityBlocks");
    } else {
      nConcurrentLumis = nConcurrentRuns;
    }
     */
    //Check that relationships between threading parameters makes sense
    /*
    if(nThreads<nStreams) {
      //bad
    }
    if(nConcurrentRuns>nStreams) {
      //bad
    }
    if(nConcurrentRuns>nConcurrentLumis) {
      //bad
    }
     */
    //forking
    ParameterSet const& forking = optionsPset.getUntrackedParameterSet("multiProcesses", ParameterSet());
    numberOfForkedChildren_ = forking.getUntrackedParameter<int>("maxChildProcesses", 0);
    numberOfSequentialEventsPerChild_ = forking.getUntrackedParameter<unsigned int>("maxSequentialEventsPerChild", 1);
    setCpuAffinity_ = forking.getUntrackedParameter<bool>("setCpuAffinity", false);
    continueAfterChildFailure_ = forking.getUntrackedParameter<bool>("continueAfterChildFailure",false);
    std::vector<ParameterSet> const& excluded = forking.getUntrackedParameterSetVector("eventSetupDataToExcludeFromPrefetching", std::vector<ParameterSet>());
    for(std::vector<ParameterSet>::const_iterator itPS = excluded.begin(), itPSEnd = excluded.end();
        itPS != itPSEnd;
        ++itPS) {
      eventSetupDataToExcludeFromPrefetching_[itPS->getUntrackedParameter<std::string>("record")].insert(
                                                std::make_pair(itPS->getUntrackedParameter<std::string>("type", "*"),
                                                               itPS->getUntrackedParameter<std::string>("label", "")));
    }
    IllegalParameters::setThrowAnException(optionsPset.getUntrackedParameter<bool>("throwIfIllegalParameter", true));

    // Now do general initialization
    ScheduleItems items;

    //initialize the services
    boost::shared_ptr<std::vector<ParameterSet> > pServiceSets = processDesc->getServicesPSets();
    ServiceToken token = items.initServices(*pServiceSets, *parameterSet, iToken, iLegacy, true);
    serviceToken_ = items.addCPRandTNS(*parameterSet, token);

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    // intialize miscellaneous items
    boost::shared_ptr<CommonParams> common(items.initMisc(*parameterSet));

    // intialize the event setup provider
    esp_ = espController_->makeProvider(*parameterSet);

    // initialize the looper, if any
    looper_ = fillLooper(*espController_, *esp_, *parameterSet);
    if(looper_) {
      looper_->setActionTable(items.act_table_.get());
      looper_->attachTo(*items.actReg_);

      //For now loopers make us run only 1 transition at a time
      nStreams=1;
      nConcurrentLumis=1;
      nConcurrentRuns=1;
    }
    
    preallocations_ = PreallocationConfiguration{nThreads,nStreams,nConcurrentLumis,nConcurrentRuns};

    // initialize the input source
    input_ = makeInput(*parameterSet, *common, *items.preg_, items.branchIDListHelper_, items.actReg_, items.processConfiguration_);

    // intialize the Schedule
    schedule_ = items.initSchedule(*parameterSet,subProcessParameterSet.get(),preallocations_,&processContext_);

    // set the data members
    act_table_ = std::move(items.act_table_);
    actReg_ = items.actReg_;
    preg_.reset(items.preg_.release());
    branchIDListHelper_ = items.branchIDListHelper_;
    processConfiguration_ = items.processConfiguration_;
    processContext_.setProcessConfiguration(processConfiguration_.get());
    principalCache_.setProcessHistoryRegistry(input_->processHistoryRegistry());

    FDEBUG(2) << parameterSet << std::endl;

    principalCache_.setNumberOfConcurrentPrincipals(preallocations_);
    for(unsigned int index = 0; index<preallocations_.numberOfStreams(); ++index ) {
      // Reusable event principal
      boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(preg_,
                                                              branchIDListHelper_,
                                                              *processConfiguration_,
                                                              historyAppender_.get(),
                                                              index));
      principalCache_.insert(ep,index);
    }
    // initialize the subprocess, if there is one
    if(subProcessParameterSet) {
      subProcess_.reset(new SubProcess(*subProcessParameterSet,
                                       *parameterSet,
                                       preg_,
                                       input_->processHistoryRegistryForUpdate(),
                                       branchIDListHelper_,
                                       *espController_,
                                       *actReg_,
                                       token,
                                       serviceregistry::kConfigurationOverrides,
                                       preallocations_,
                                       &processContext_));
    }
  }

  EventProcessor::~EventProcessor() {
    // Make the services available while everything is being deleted.
    ServiceToken token = getToken();
    ServiceRegistry::Operate op(token);

    // manually destroy all these thing that may need the services around
    espController_.reset();
    subProcess_.reset();
    esp_.reset();
    schedule_.reset();
    input_.reset();
    looper_.reset();
    actReg_.reset();

    pset::Registry* psetRegistry = pset::Registry::instance();
    psetRegistry->dataForUpdate().clear();
    psetRegistry->extraForUpdate().setID(ParameterSetID());

    ParentageRegistry::instance()->dataForUpdate().clear();
  }

  void
  EventProcessor::beginJob() {
    if(beginJobCalled_) return;
    beginJobCalled_=true;
    bk::beginJob();

    // StateSentry toerror(this); // should we add this ?
    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    //NOTE:  This implementation assumes 'Job' means one call
    // the EventProcessor::run
    // If it really means once per 'application' then this code will
    // have to be changed.
    // Also have to deal with case where have 'run' then new Module
    // added and do 'run'
    // again.  In that case the newly added Module needs its 'beginJob'
    // to be called.

    //NOTE: in future we should have a beginOfJob for looper that takes no arguments
    //  For now we delay calling beginOfJob until first beginOfRun
    //if(looper_) {
    //   looper_->beginOfJob(es);
    //}
    try {
      try {
        input_->doBeginJob();
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      ex.addContext("Calling beginJob for the source");
      throw;
    }
    schedule_->beginJob(*preg_);
    // toerror.succeeded(); // should we add this?
    if(hasSubProcess()) subProcess_->doBeginJob();
    actReg_->postBeginJobSignal_();
    
    for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
      schedule_->beginStream(i);
      if(hasSubProcess()) subProcess_->doBeginStream(i);
    }
  }

  void
  EventProcessor::endJob() {
    // Collects exceptions, so we don't throw before all operations are performed.
    ExceptionCollector c("Multiple exceptions were thrown while executing endJob. An exception message follows for each.\n");

    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    //NOTE: this really should go elsewhere in the future
    for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
      c.call([this,i](){this->schedule_->endStream(i);});
      if(hasSubProcess()) {
        c.call([this,i](){ this->subProcess_->doEndStream(i); } );
      }
    }
    schedule_->endJob(c);
    if(hasSubProcess()) {
      c.call(boost::bind(&SubProcess::doEndJob, subProcess_.get()));
    }
    c.call(boost::bind(&InputSource::doEndJob, input_.get()));
    if(looper_) {
      c.call(boost::bind(&EDLooperBase::endOfJob, looper_));
    }
    auto actReg = actReg_.get();
    c.call([actReg](){actReg->postEndJobSignal_();});
    if(c.hasThrown()) {
      c.rethrow();
    }
  }

  ServiceToken
  EventProcessor::getToken() {
    return serviceToken_;
  }

  //Setup signal handler to listen for when forked children stop
  namespace {
    //These are volatile since the compiler can not be allowed to optimize them
    // since they can be modified in the signaller handler
    volatile bool child_failed = false;
    volatile unsigned int num_children_done = 0;
    volatile int child_fail_exit_status = 0;
    volatile int child_fail_signal = 0;
    
    //NOTE: We setup the signal handler to run in the main thread which
    // is also the same thread that then reads the above values

    extern "C" {
      void ep_sigchld(int, siginfo_t*, void*) {
        //printf("in sigchld\n");
        //FDEBUG(1) << "in sigchld handler\n";
        int stat_loc;
        pid_t p = waitpid(-1, &stat_loc, WNOHANG);
        while(0<p) {
          //printf("  looping\n");
          if(WIFEXITED(stat_loc)) {
            ++num_children_done;
            if(0 != WEXITSTATUS(stat_loc)) {
              child_fail_exit_status = WEXITSTATUS(stat_loc);
              child_failed = true;
            }
          }
          if(WIFSIGNALED(stat_loc)) {
            ++num_children_done;
            child_fail_signal = WTERMSIG(stat_loc);
            child_failed = true;
          }
          p = waitpid(-1, &stat_loc, WNOHANG);
        }
      }
    }

  }

  enum {
    kChildSucceed,
    kChildExitBadly,
    kChildSegv,
    kMaxChildAction
  };

  namespace {
    unsigned int numberOfDigitsInChildIndex(unsigned int numberOfChildren) {
      unsigned int n = 0;
      while(numberOfChildren != 0) {
        ++n;
        numberOfChildren /= 10;
      }
      if(n == 0) {
        n = 3; // Protect against zero numberOfChildren
      }
      return n;
    }
    
    /*This class embodied the thread which is used to listen to the forked children and
     then tell them which events they should process */
    class MessageSenderToSource {
    public:
      MessageSenderToSource(std::vector<int> const& childrenSockets, std::vector<int> const& childrenPipes, long iNEventsToProcess);
      void operator()();

    private:
      const std::vector<int>& m_childrenPipes;
      long const m_nEventsToProcess;
      fd_set m_socketSet;
      unsigned int m_aliveChildren;
      int m_maxFd;
    };
    
    MessageSenderToSource::MessageSenderToSource(std::vector<int> const& childrenSockets,
                                                 std::vector<int> const& childrenPipes,
                                                 long iNEventsToProcess):
    m_childrenPipes(childrenPipes),
    m_nEventsToProcess(iNEventsToProcess),
    m_aliveChildren(childrenSockets.size()),
    m_maxFd(0)
    {
      FD_ZERO(&m_socketSet);
      for (std::vector<int>::const_iterator it = childrenSockets.begin(), itEnd = childrenSockets.end();
           it != itEnd; it++) {
        FD_SET(*it, &m_socketSet);
        if (*it > m_maxFd) {
          m_maxFd = *it;
        }
      }
      for (std::vector<int>::const_iterator it = childrenPipes.begin(), itEnd = childrenPipes.end();
           it != itEnd; ++it) {
        FD_SET(*it, &m_socketSet);
        if (*it > m_maxFd) {
          m_maxFd = *it;
        }
      }
      m_maxFd++; // select reads [0,m_maxFd).
    }
   
    /* This function is the heart of the communication between parent and child.
     * When ready for more data, the child (see MessageReceiverForSource) requests
     * data through a AF_UNIX socket message.  The parent will then assign the next
     * chunk of data by sending a message back.
     *
     * Additionally, this function also monitors the read-side of the pipe fd from the child.
     * If the child dies unexpectedly, the pipe will be selected as ready for read and
     * will return EPIPE when read from.  Further, if the child thinks the parent has died
     * (defined as waiting more than 1s for a response), it will write a single byte to
     * the pipe.  If the parent has died, the child will get a EPIPE and throw an exception.
     * If still alive, the parent will read the byte and ignore it.
     *
     * Note this function is complemented by the SIGCHLD handler above as currently only the SIGCHLD
     * handler can distinguish between success and failure cases.
     */
 
    void
    MessageSenderToSource::operator()() {
      multicore::MessageForParent childMsg;
      LogInfo("ForkingController") << "I am controller";
      //this is the master and therefore the controller
      
      multicore::MessageForSource sndmsg;
      sndmsg.startIndex = 0;
      sndmsg.nIndices = m_nEventsToProcess;
      do {
        
        fd_set readSockets, errorSockets;
        // Wait for a request from a child for events.
        memcpy(&readSockets, &m_socketSet, sizeof(m_socketSet));
        memcpy(&errorSockets, &m_socketSet, sizeof(m_socketSet));
        // Note that we don't timeout; may be reconsidered in the future.
        ssize_t rc;
        while (((rc = select(m_maxFd, &readSockets, NULL, &errorSockets, NULL)) < 0) && (errno == EINTR)) {}
        if (rc < 0) {
          std::cerr << "select failed; should be impossible due to preconditions.\n";
          abort();
          break;
        }

        // Read the message from the child.
        for (int idx=0; idx<m_maxFd; idx++) {

          // Handle errors
          if (FD_ISSET(idx, &errorSockets)) {
            LogInfo("ForkingController") << "Error on socket " << idx;
            FD_CLR(idx, &m_socketSet);
            close(idx);
            // See if it was the watchdog pipe that died.
            for (std::vector<int>::const_iterator it = m_childrenPipes.begin(); it != m_childrenPipes.end(); it++) {
              if (*it == idx) {
                m_aliveChildren--;
              }
            }
            continue;
          }
          
          if (!FD_ISSET(idx, &readSockets)) {
            continue;
          }

          // See if this FD is a child watchdog pipe.  If so, read from it to prevent
          // writes from blocking.
          bool is_pipe = false;
          for (std::vector<int>::const_iterator it = m_childrenPipes.begin(), itEnd = m_childrenPipes.end(); it != itEnd; it++) {
              if (*it == idx) {
                is_pipe = true;
                char buf;
                while (((rc = read(idx, &buf, 1)) < 0) && (errno == EINTR)) {}
                if (rc <= 0) {
                  m_aliveChildren--;
                  FD_CLR(idx, &m_socketSet);
                  close(idx);
                }
              }
          }

          // Only execute this block if the FD is a socket for sending the child work.
          if (!is_pipe) {
            while (((rc = recv(idx, reinterpret_cast<char*>(&childMsg),childMsg.sizeForBuffer() , 0)) < 0) && (errno == EINTR)) {}
            if (rc < 0) {
              FD_CLR(idx, &m_socketSet);
              close(idx);
              continue;
            }
          
            // Tell the child what events to process.
            // If 'send' fails, then the child process has failed (any other possibilities are
            // eliminated because we are using fixed-size messages with Unix datagram sockets).
            // Thus, the SIGCHLD handler will fire and set child_fail = true.
            while (((rc = send(idx, (char *)(&sndmsg), multicore::MessageForSource::sizeForBuffer(), 0)) < 0) && (errno == EINTR)) {}
            if (rc < 0) {
              FD_CLR(idx, &m_socketSet);
              close(idx);
              continue;
            }
            //std::cout << "Sent chunk starting at " << sndmsg.startIndex << " to child, length " << sndmsg.nIndices << std::endl;
            sndmsg.startIndex += sndmsg.nIndices;
          }
        }
      
      } while (m_aliveChildren > 0);
      
      return;
    }

  }

  
  void EventProcessor::possiblyContinueAfterForkChildFailure() {
    if(child_failed && continueAfterChildFailure_) {
      if (child_fail_signal) {
        LogSystem("ForkedChildFailed") << "child process ended abnormally with signal " << child_fail_signal;
        child_fail_signal=0;
      } else if (child_fail_exit_status) {
        LogSystem("ForkedChildFailed") << "child process ended abnormally with exit code " << child_fail_exit_status;
        child_fail_exit_status=0;
      } else {
        LogSystem("ForkedChildFailed") << "child process ended abnormally for unknown reason";
      }
      child_failed =false;
    }
  }

  bool
  EventProcessor::forkProcess(std::string const& jobReportFile) {

    if(0 == numberOfForkedChildren_) {return true;}
    assert(0<numberOfForkedChildren_);
    //do what we want done in common
    {
      beginJob(); //make sure this was run
      // make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      InputSource::ItemType itemType;
      itemType = input_->nextItemType();

      assert(itemType == InputSource::IsFile);
      {
        readFile();
      }
      itemType = input_->nextItemType();
      assert(itemType == InputSource::IsRun);

      LogSystem("ForkingEventSetupPreFetching") << " prefetching for run " << input_->runAuxiliary()->run();
      IOVSyncValue ts(EventID(input_->runAuxiliary()->run(), 0, 0),
                      input_->runAuxiliary()->beginTime());
      espController_->eventSetupForInstance(ts);
      EventSetup const& es = esp_->eventSetup();

      //now get all the data available in the EventSetup
      std::vector<eventsetup::EventSetupRecordKey> recordKeys;
      es.fillAvailableRecordKeys(recordKeys);
      std::vector<eventsetup::DataKey> dataKeys;
      for(std::vector<eventsetup::EventSetupRecordKey>::const_iterator itKey = recordKeys.begin(), itEnd = recordKeys.end();
          itKey != itEnd;
          ++itKey) {
        eventsetup::EventSetupRecord const* recordPtr = es.find(*itKey);
        //see if this is on our exclusion list
        ExcludedDataMap::const_iterator itExcludeRec = eventSetupDataToExcludeFromPrefetching_.find(itKey->type().name());
        ExcludedData const* excludedData(nullptr);
        if(itExcludeRec != eventSetupDataToExcludeFromPrefetching_.end()) {
          excludedData = &(itExcludeRec->second);
          if(excludedData->size() == 0 || excludedData->begin()->first == "*") {
            //skip all items in this record
            continue;
          }
        }
        if(0 != recordPtr) {
          dataKeys.clear();
          recordPtr->fillRegisteredDataKeys(dataKeys);
          for(std::vector<eventsetup::DataKey>::const_iterator itDataKey = dataKeys.begin(), itDataKeyEnd = dataKeys.end();
              itDataKey != itDataKeyEnd;
              ++itDataKey) {
            //std::cout << "  " << itDataKey->type().name() << " " << itDataKey->name().value() << std::endl;
            if(0 != excludedData && excludedData->find(std::make_pair(itDataKey->type().name(), itDataKey->name().value())) != excludedData->end()) {
              LogInfo("ForkingEventSetupPreFetching") << "   excluding:" << itDataKey->type().name() << " " << itDataKey->name().value() << std::endl;
              continue;
            }
            try {
              recordPtr->doGet(*itDataKey);
            } catch(cms::Exception& e) {
             LogWarning("ForkingEventSetupPreFetching") << e.what();
            }
          }
        }
      }
    }
    LogSystem("ForkingEventSetupPreFetching") <<"  done prefetching";
    {
      // make the services available
      ServiceRegistry::Operate operate(serviceToken_);
      Service<JobReport> jobReport;
      jobReport->parentBeforeFork(jobReportFile, numberOfForkedChildren_);

      //Now actually do the forking
      actReg_->preForkReleaseResourcesSignal_();
      input_->doPreForkReleaseResources();
      schedule_->preForkReleaseResources();
    }
    installCustomHandler(SIGCHLD, ep_sigchld);


    unsigned int childIndex = 0;
    unsigned int const kMaxChildren = numberOfForkedChildren_;
    unsigned int const numberOfDigitsInIndex = numberOfDigitsInChildIndex(kMaxChildren);
    std::vector<pid_t> childrenIds;
    childrenIds.reserve(kMaxChildren);
    std::vector<int> childrenSockets;
    childrenSockets.reserve(kMaxChildren);
    std::vector<int> childrenPipes;
    childrenPipes.reserve(kMaxChildren);
    std::vector<int> childrenSocketsCopy;
    childrenSocketsCopy.reserve(kMaxChildren);
    std::vector<int> childrenPipesCopy;
    childrenPipesCopy.reserve(kMaxChildren);
    int pipes[] {0, 0};

    {
      // make the services available
      ServiceRegistry::Operate operate(serviceToken_);
      Service<JobReport> jobReport;
      int sockets[2], fd_flags;
      for(; childIndex < kMaxChildren; ++childIndex) {
        // Create a UNIX_DGRAM socket pair
        if (socketpair(AF_UNIX, SOCK_DGRAM, 0, sockets)) {
          printf("Error creating communication socket (errno=%d, %s)\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }
        if (pipe(pipes)) {
          printf("Error creating communication pipes (errno=%d, %s)\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }
        // set CLOEXEC so the socket/pipe doesn't get leaked if the child exec's.
        if ((fd_flags = fcntl(sockets[1], F_GETFD, NULL)) == -1) {
          printf("Failed to get fd flags: %d %s\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }
        // Mark socket as non-block.  Child must be careful to do select prior
        // to reading from socket.
        if (fcntl(sockets[1], F_SETFD, fd_flags | FD_CLOEXEC | O_NONBLOCK) == -1) {
          printf("Failed to set new fd flags: %d %s\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }
        if ((fd_flags = fcntl(pipes[1], F_GETFD, NULL)) == -1) {
          printf("Failed to get fd flags: %d %s\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }
        if (fcntl(pipes[1], F_SETFD, fd_flags | FD_CLOEXEC) == -1) {
          printf("Failed to set new fd flags: %d %s\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }
        // Linux man page notes there are some edge cases where reading from a
        // fd can block, even after a select.
        if ((fd_flags = fcntl(pipes[0], F_GETFD, NULL)) == -1) {
          printf("Failed to get fd flags: %d %s\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }
        if (fcntl(pipes[0], F_SETFD, fd_flags | O_NONBLOCK) == -1) {
          printf("Failed to set new fd flags: %d %s\n", errno, strerror(errno));
          exit(EXIT_FAILURE);
        }

        childrenPipesCopy = childrenPipes;
        childrenSocketsCopy = childrenSockets;

        pid_t value = fork();
        if(value == 0) {
          // Close the parent's side of the socket and pipe which will talk to us.
          close(pipes[0]);
          close(sockets[0]);
          // Close our copies of the parent's other communication pipes.
          for(std::vector<int>::const_iterator it=childrenPipesCopy.begin(); it != childrenPipesCopy.end(); it++) {
            close(*it);
          }
          for(std::vector<int>::const_iterator it=childrenSocketsCopy.begin(); it != childrenSocketsCopy.end(); it++) {
            close(*it);
          }

          // this is the child process, redirect stdout and stderr to a log file
          fflush(stdout);
          fflush(stderr);
          std::stringstream stout;
          stout << "redirectout_" << getpgrp() << "_" << std::setw(numberOfDigitsInIndex) << std::setfill('0') << childIndex << ".log";
          if(0 == freopen(stout.str().c_str(), "w", stdout)) {
            LogError("ForkingStdOutRedirect") << "Error during freopen of child process "<< childIndex;
          }
          if(dup2(fileno(stdout), fileno(stderr)) < 0) {
            LogError("ForkingStdOutRedirect") << "Error during dup2 of child process"<< childIndex;
          }

          LogInfo("ForkingChild") << "I am child " << childIndex << " with pgid " << getpgrp();
          if(setCpuAffinity_) {
            // CPU affinity is handled differently on macosx.
            // We disable it and print a message until someone reads:
            //
            // http://developer.apple.com/mac/library/releasenotes/Performance/RN-AffinityAPI/index.html
            //
            // and implements it.
#ifdef __APPLE__
            LogInfo("ForkingChildAffinity") << "Architecture support for CPU affinity not implemented.";
#else
            LogInfo("ForkingChildAffinity") << "Setting CPU affinity, setting this child to cpu " << childIndex;
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(childIndex, &mask);
            if(sched_setaffinity(0, sizeof(mask), &mask) != 0) {
              LogError("ForkingChildAffinity") << "Failed to set the cpu affinity, errno " << errno;
              exit(-1);
            }
#endif
          }
          break;
        } else {
          //this is the parent
          close(pipes[1]);
          close(sockets[1]);
        }
        if(value < 0) {
          LogError("ForkingChild") << "failed to create a child";
          exit(-1);
        }
        childrenIds.push_back(value);
        childrenSockets.push_back(sockets[0]);
        childrenPipes.push_back(pipes[0]);
      }

      if(childIndex < kMaxChildren) {
        jobReport->childAfterFork(jobReportFile, childIndex, kMaxChildren);
        actReg_->postForkReacquireResourcesSignal_(childIndex, kMaxChildren);

        boost::shared_ptr<multicore::MessageReceiverForSource> receiver(new multicore::MessageReceiverForSource(sockets[1], pipes[1]));
        input_->doPostForkReacquireResources(receiver);
        schedule_->postForkReacquireResources(childIndex, kMaxChildren);
        //NOTE: sources have to reset themselves by listening to the post fork message
        //rewindInput();
        return true;
      }
      jobReport->parentAfterFork(jobReportFile);
    }

    //this is the original, which is now the master for all the children

    //Need to wait for signals from the children or externally
    // To wait we must
    // 1) block the signals we want to wait on so we do not have a race condition
    // 2) check that we haven't already meet our ending criteria
    // 3) call sigsuspend, which unblocks the signals and waits until a signal is caught
    sigset_t blockingSigSet;
    sigset_t unblockingSigSet;
    sigset_t oldSigSet;
    pthread_sigmask(SIG_SETMASK, NULL, &unblockingSigSet);
    pthread_sigmask(SIG_SETMASK, NULL, &blockingSigSet);
    sigaddset(&blockingSigSet, SIGCHLD);
    sigaddset(&blockingSigSet, SIGUSR2);
    sigaddset(&blockingSigSet, SIGINT);
    sigdelset(&unblockingSigSet, SIGCHLD);
    sigdelset(&unblockingSigSet, SIGUSR2);
    sigdelset(&unblockingSigSet, SIGINT);
    pthread_sigmask(SIG_BLOCK, &blockingSigSet, &oldSigSet);

    // If there are too many fd's (unlikely, but possible) for select, denote this 
    // because the sender will fail.
    bool too_many_fds = false;
    if (pipes[1]+1 > FD_SETSIZE) {
      LogError("ForkingFileDescriptors") << "too many file descriptors for multicore job";
      too_many_fds = true;
    }

    //create a thread that sends the units of work to workers
    // we create it after all signals were blocked so that this
    // thread is never interupted by a signal
    MessageSenderToSource sender(childrenSockets, childrenPipes, numberOfSequentialEventsPerChild_);
    boost::thread senderThread(sender);

    if(not too_many_fds) {
      //NOTE: a child could have failed before we got here and even after this call
      // which is why the 'if' is conditional on continueAfterChildFailure_
      possiblyContinueAfterForkChildFailure();
      while(!shutdown_flag && (!child_failed or continueAfterChildFailure_) && (childrenIds.size() != num_children_done)) {
        sigsuspend(&unblockingSigSet);
        possiblyContinueAfterForkChildFailure();
        LogInfo("ForkingAwake") << "woke from sigwait" << std::endl;
      }
    }
    pthread_sigmask(SIG_SETMASK, &oldSigSet, NULL);

    LogInfo("ForkingStopping") << "num children who have already stopped " << num_children_done;
    if(child_failed) {
      LogError("ForkingStopping") << "child failed";
    }
    if(shutdown_flag) {
      LogSystem("ForkingStopping") << "asked to shutdown";
    }

    if(too_many_fds || shutdown_flag || (child_failed && (num_children_done != childrenIds.size()))) {
      LogInfo("ForkingStopping") << "must stop children" << std::endl;
      for(std::vector<pid_t>::iterator it = childrenIds.begin(), itEnd = childrenIds.end();
          it != itEnd; ++it) {
        /* int result = */ kill(*it, SIGUSR2);
      }
      pthread_sigmask(SIG_BLOCK, &blockingSigSet, &oldSigSet);
      while(num_children_done != kMaxChildren) {
        sigsuspend(&unblockingSigSet);
      }
      pthread_sigmask(SIG_SETMASK, &oldSigSet, NULL);
    }
    // The senderThread will notice the pipes die off, one by one.  Once all children are gone, it will exit.
    senderThread.join();
    if(child_failed && !continueAfterChildFailure_) {
      if (child_fail_signal) {
        throw cms::Exception("ForkedChildFailed") << "child process ended abnormally with signal " << child_fail_signal;
      } else if (child_fail_exit_status) {
        throw cms::Exception("ForkedChildFailed") << "child process ended abnormally with exit code " << child_fail_exit_status;
      } else {
        throw cms::Exception("ForkedChildFailed") << "child process ended abnormally for unknown reason";
      }
    }
    if(too_many_fds) {
      throw cms::Exception("ForkedParentFailed") << "hit select limit for number of fds";
    }
    return false;
  }

  std::vector<ModuleDescription const*>
  EventProcessor::getAllModuleDescriptions() const {
    return schedule_->getAllModuleDescriptions();
  }

  int
  EventProcessor::totalEvents() const {
    return schedule_->totalEvents();
  }

  int
  EventProcessor::totalEventsPassed() const {
    return schedule_->totalEventsPassed();
  }

  int
  EventProcessor::totalEventsFailed() const {
    return schedule_->totalEventsFailed();
  }

  void
  EventProcessor::enableEndPaths(bool active) {
    schedule_->enableEndPaths(active);
  }

  bool
  EventProcessor::endPathsEnabled() const {
    return schedule_->endPathsEnabled();
  }

  void
  EventProcessor::getTriggerReport(TriggerReport& rep) const {
    schedule_->getTriggerReport(rep);
  }

  void
  EventProcessor::clearCounters() {
    schedule_->clearCounters();
  }


  std::auto_ptr<statemachine::Machine>
  EventProcessor::createStateMachine() {
    statemachine::FileMode fileMode;
    if(fileMode_.empty()) fileMode = statemachine::FULLMERGE;
    else if(fileMode_ == std::string("NOMERGE")) fileMode = statemachine::NOMERGE;
    else if(fileMode_ == std::string("FULLMERGE")) fileMode = statemachine::FULLMERGE;
    else {
      throw Exception(errors::Configuration, "Illegal fileMode parameter value: ")
      << fileMode_ << ".\n"
      << "Legal values are 'NOMERGE' and 'FULLMERGE'.\n";
    }
    
    statemachine::EmptyRunLumiMode emptyRunLumiMode;
    if(emptyRunLumiMode_.empty()) emptyRunLumiMode = statemachine::handleEmptyRunsAndLumis;
    else if(emptyRunLumiMode_ == std::string("handleEmptyRunsAndLumis")) emptyRunLumiMode = statemachine::handleEmptyRunsAndLumis;
    else if(emptyRunLumiMode_ == std::string("handleEmptyRuns")) emptyRunLumiMode = statemachine::handleEmptyRuns;
    else if(emptyRunLumiMode_ == std::string("doNotHandleEmptyRunsAndLumis")) emptyRunLumiMode = statemachine::doNotHandleEmptyRunsAndLumis;
    else {
      throw Exception(errors::Configuration, "Illegal emptyMode parameter value: ")
      << emptyRunLumiMode_ << ".\n"
      << "Legal values are 'handleEmptyRunsAndLumis', 'handleEmptyRuns', and 'doNotHandleEmptyRunsAndLumis'.\n";
    }
    
    std::auto_ptr<statemachine::Machine> machine(new statemachine::Machine(this,
                                             fileMode,
                                             emptyRunLumiMode));
    
    machine->initiate();
    return machine;
  }

  bool
  EventProcessor::checkForAsyncStopRequest(StatusCode& returnCode) {
    bool returnValue = false;
    
    // Look for a shutdown signal
    {
      boost::mutex::scoped_lock sl(usr2_lock);
      if(shutdown_flag) {
        returnValue = true;
        returnCode = epSignal;
      }
    }
    return returnValue;
  }


  EventProcessor::StatusCode
  EventProcessor::runToCompletion() {

    StatusCode returnCode=epSuccess;
    asyncStopStatusCodeFromProcessingEvents_=epSuccess;
    std::auto_ptr<statemachine::Machine> machine;
    {
      beginJob(); //make sure this was called
      
      //StatusCode returnCode = epSuccess;
      stateMachineWasInErrorState_ = false;
      
      // make the services available
      ServiceRegistry::Operate operate(serviceToken_);

      machine = createStateMachine();
      nextItemTypeFromProcessingEvents_=InputSource::IsEvent;
      asyncStopRequestedWhileProcessingEvents_=false;
      try {
        try {
          
          InputSource::ItemType itemType;
          
          while(true) {
            
            bool more = true;
            if(numberOfForkedChildren_ > 0) {
              size_t size = preg_->size();
              more = input_->skipForForking();
              if(more) {
                if(size < preg_->size()) {
                  principalCache_.adjustIndexesAfterProductRegistryAddition();
                }
                principalCache_.adjustEventsToNewProductRegistry(preg_);
              }
            } 
            itemType = (more ? input_->nextItemType() : InputSource::IsStop);
            
            FDEBUG(1) << "itemType = " << itemType << "\n";
            
            if(checkForAsyncStopRequest(returnCode)) {
              forceLooperToEnd_ = true;
              machine->process_event(statemachine::Stop());
              forceLooperToEnd_ = false;
              break;
            }
            
            if(itemType == InputSource::IsEvent) {
              machine->process_event(statemachine::Event());
              if(asyncStopRequestedWhileProcessingEvents_) {
                forceLooperToEnd_ = true;
                machine->process_event(statemachine::Stop());
                forceLooperToEnd_ = false;
                returnCode = asyncStopStatusCodeFromProcessingEvents_;
                break;
              }
              itemType = nextItemTypeFromProcessingEvents_;
            }
            
            if(itemType == InputSource::IsEvent) {
            }
            else if(itemType == InputSource::IsStop) {
              machine->process_event(statemachine::Stop());
            }
            else if(itemType == InputSource::IsFile) {
              machine->process_event(statemachine::File());
            }
            else if(itemType == InputSource::IsRun) {
              machine->process_event(statemachine::Run(input_->reducedProcessHistoryID(), input_->run()));
            }
            else if(itemType == InputSource::IsLumi) {
              machine->process_event(statemachine::Lumi(input_->luminosityBlock()));
            }
            else if(itemType == InputSource::IsSynchronize) {
              //For now, we don't have to do anything
            }
            // This should be impossible
            else {
              throw Exception(errors::LogicError)
              << "Unknown next item type passed to EventProcessor\n"
              << "Please report this error to the Framework group\n";
            }
            if(machine->terminated()) {
              break;
            }
          }  // End of loop over state machine events
        } // Try block
        catch (cms::Exception& e) { throw; }
        catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
        catch (std::exception& e) { convertException::stdToEDM(e); }
        catch(std::string& s) { convertException::stringToEDM(s); }
        catch(char const* c) { convertException::charPtrToEDM(c); }
        catch (...) { convertException::unknownToEDM(); }
      } // Try block
      // Some comments on exception handling related to the boost state machine:
      //
      // Some states used in the machine are special because they
      // perform actions while the machine is being terminated, actions
      // such as close files, call endRun, call endLumi etc ...  Each of these
      // states has two functions that perform these actions.  The functions
      // are almost identical.  The major difference is that one version
      // catches all exceptions and the other lets exceptions pass through.
      // The destructor catches them and the other function named "exit" lets
      // them pass through.  On a normal termination, boost will always call
      // "exit" and then the state destructor.  In our state classes, the
      // the destructors do nothing if the exit function already took
      // care of things.  Here's the interesting part.  When boost is
      // handling an exception the "exit" function is not called (a boost
      // feature).
      //
      // If an exception occurs while the boost machine is in control
      // (which usually means inside a process_event call), then
      // the boost state machine destroys its states and "terminates" itself.
      // This already done before we hit the catch blocks below. In this case
      // the call to terminateMachine below only destroys an already
      // terminated state machine.  Because exit is not called, the state destructors
      // handle cleaning up lumis, runs, and files.  The destructors swallow
      // all exceptions and only pass through the exceptions messages, which
      // are tacked onto the original exception below.
      //
      // If an exception occurs when the boost state machine is not
      // in control (outside the process_event functions), then boost
      // cannot destroy its own states.  The terminateMachine function
      // below takes care of that.  The flag "alreadyHandlingException"
      // is set true so that the state exit functions do nothing (and
      // cannot throw more exceptions while handling the first).  Then the
      // state destructors take care of this because exit did nothing.
      //
      // In both cases above, the EventProcessor::endOfLoop function is
      // not called because it can throw exceptions.
      //
      // One tricky aspect of the state machine is that things that can
      // throw should not be invoked by the state machine while another
      // exception is being handled.
      // Another tricky aspect is that it appears to be important to
      // terminate the state machine before invoking its destructor.
      // We've seen crashes that are not understood when that is not
      // done.  Maintainers of this code should be careful about this.
      
      catch (cms::Exception & e) {
        alreadyHandlingException_ = true;
        terminateMachine(machine);
        alreadyHandlingException_ = false;
        if (!exceptionMessageLumis_.empty()) {
          e.addAdditionalInfo(exceptionMessageLumis_);
          if (e.alreadyPrinted()) {
            LogAbsolute("Additional Exceptions") << exceptionMessageLumis_;
          }
        }
        if (!exceptionMessageRuns_.empty()) {
          e.addAdditionalInfo(exceptionMessageRuns_);
          if (e.alreadyPrinted()) {
            LogAbsolute("Additional Exceptions") << exceptionMessageRuns_;
          }
        }
        if (!exceptionMessageFiles_.empty()) {
          e.addAdditionalInfo(exceptionMessageFiles_);
          if (e.alreadyPrinted()) {
            LogAbsolute("Additional Exceptions") << exceptionMessageFiles_;
          }
        }
        throw;
      }
      
      if(machine->terminated()) {
        FDEBUG(1) << "The state machine reports it has been terminated\n";
        machine.reset();
      }
      
      if(stateMachineWasInErrorState_) {
        throw cms::Exception("BadState")
        << "The boost state machine in the EventProcessor exited after\n"
        << "entering the Error state.\n";
      }
      
    }
    if(machine.get() != 0) {
      terminateMachine(machine);
      throw Exception(errors::LogicError)
        << "State machine not destroyed on exit from EventProcessor::runToCompletion\n"
        << "Please report this error to the Framework group\n";
    }

    return returnCode;
  }

  void EventProcessor::readFile() {
    FDEBUG(1) << " \treadFile\n";
    size_t size = preg_->size();
    fb_ = input_->readFile();
    if(size < preg_->size()) {
      principalCache_.adjustIndexesAfterProductRegistryAddition();
    }
    principalCache_.adjustEventsToNewProductRegistry(preg_);
    if(numberOfForkedChildren_ > 0) {
        fb_->setNotFastClonable(FileBlock::ParallelProcesses);
    }
  }

  void EventProcessor::closeInputFile(bool cleaningUpAfterException) {
    if (fb_.get() != nullptr) {
      input_->closeFile(fb_.get(), cleaningUpAfterException);
    }
    FDEBUG(1) << "\tcloseInputFile\n";
  }

  void EventProcessor::openOutputFiles() {
    if (fb_.get() != nullptr) {
      schedule_->openOutputFiles(*fb_);
      if(hasSubProcess()) subProcess_->openOutputFiles(*fb_);
    }
    FDEBUG(1) << "\topenOutputFiles\n";
  }

  void EventProcessor::closeOutputFiles() {
    if (fb_.get() != nullptr) {
      schedule_->closeOutputFiles();
      if(hasSubProcess()) subProcess_->closeOutputFiles();
    }
    FDEBUG(1) << "\tcloseOutputFiles\n";
  }

  void EventProcessor::respondToOpenInputFile() {
    if (fb_.get() != nullptr) {
      schedule_->respondToOpenInputFile(*fb_);
      if(hasSubProcess()) subProcess_->respondToOpenInputFile(*fb_);
    }
    FDEBUG(1) << "\trespondToOpenInputFile\n";
  }

  void EventProcessor::respondToCloseInputFile() {
    if (fb_.get() != nullptr) {
      schedule_->respondToCloseInputFile(*fb_);
      if(hasSubProcess()) subProcess_->respondToCloseInputFile(*fb_);
    }
    FDEBUG(1) << "\trespondToCloseInputFile\n";
  }

  void EventProcessor::startingNewLoop() {
    shouldWeStop_ = false;
    //NOTE: for first loop, need to delay calling 'doStartingNewLoop'
    // until after we've called beginOfJob
    if(looper_ && looperBeginJobRun_) {
      looper_->doStartingNewLoop();
    }
    FDEBUG(1) << "\tstartingNewLoop\n";
  }

  bool EventProcessor::endOfLoop() {
    if(looper_) {
      ModuleChanger changer(schedule_.get());
      looper_->setModuleChanger(&changer);
      EDLooperBase::Status status = looper_->doEndOfLoop(esp_->eventSetup());
      looper_->setModuleChanger(nullptr);
      if(status != EDLooperBase::kContinue || forceLooperToEnd_) return true;
      else return false;
    }
    FDEBUG(1) << "\tendOfLoop\n";
    return true;
  }

  void EventProcessor::rewindInput() {
    input_->repeat();
    input_->rewind();
    FDEBUG(1) << "\trewind\n";
  }

  void EventProcessor::prepareForNextLoop() {
    looper_->prepareForNextLoop(esp_.get());
    FDEBUG(1) << "\tprepareForNextLoop\n";
  }

  bool EventProcessor::shouldWeCloseOutput() const {
    FDEBUG(1) << "\tshouldWeCloseOutput\n";
    return hasSubProcess() ? subProcess_->shouldWeCloseOutput() : schedule_->shouldWeCloseOutput();
  }

  void EventProcessor::doErrorStuff() {
    FDEBUG(1) << "\tdoErrorStuff\n";
    LogError("StateMachine")
      << "The EventProcessor state machine encountered an unexpected event\n"
      << "and went to the error state\n"
      << "Will attempt to terminate processing normally\n"
      << "(IF using the looper the next loop will be attempted)\n"
      << "This likely indicates a bug in an input module or corrupted input or both\n";
    stateMachineWasInErrorState_ = true;
  }

  void EventProcessor::beginRun(statemachine::Run const& run) {
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(run.processHistoryID(), run.runNumber());
    input_->doBeginRun(runPrincipal, &processContext_);
    IOVSyncValue ts(EventID(runPrincipal.run(), 0, 0),
                    runPrincipal.beginTime());
    if(forceESCacheClearOnNewRun_){
      espController_->forceCacheClear();
    }
    espController_->eventSetupForInstance(ts);
    EventSetup const& es = esp_->eventSetup();
    if(looper_ && looperBeginJobRun_== false) {
      looper_->copyInfo(ScheduleInfo(schedule_.get()));
      looper_->beginOfJob(es);
      looperBeginJobRun_ = true;
      looper_->doStartingNewLoop();
    }
    {
      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> Traits;
      schedule_->processOneGlobal<Traits>(runPrincipal, es);
      if(hasSubProcess()) {
        subProcess_->doBeginRun(runPrincipal, ts);
      }
    }
    FDEBUG(1) << "\tbeginRun " << run.runNumber() << "\n";
    if(looper_) {
      looper_->doBeginRun(runPrincipal, es, &processContext_);
    }
    {
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> Traits;
      for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
        schedule_->processOneStream<Traits>(i,runPrincipal, es);
        if(hasSubProcess()) {
          subProcess_->doStreamBeginRun(i, runPrincipal, ts);
        }
      }
    }
    FDEBUG(1) << "\tstreamBeginRun " << run.runNumber() << "\n";
    if(looper_) {
      //looper_->doStreamBeginRun(schedule_->streamID(),runPrincipal, es);
    }
  }

  void EventProcessor::endRun(statemachine::Run const& run, bool cleaningUpAfterException) {
    RunPrincipal& runPrincipal = principalCache_.runPrincipal(run.processHistoryID(), run.runNumber());
    input_->doEndRun(runPrincipal, cleaningUpAfterException, &processContext_);
    IOVSyncValue ts(EventID(runPrincipal.run(), LuminosityBlockID::maxLuminosityBlockNumber(), EventID::maxEventNumber()),
                    runPrincipal.endTime());
    espController_->eventSetupForInstance(ts);
    EventSetup const& es = esp_->eventSetup();
    {
      for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
        typedef OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> Traits;
        schedule_->processOneStream<Traits>(i,runPrincipal, es, cleaningUpAfterException);
        if(hasSubProcess()) {
          subProcess_->doStreamEndRun(i,runPrincipal, ts, cleaningUpAfterException);
        }
      }
    }
    FDEBUG(1) << "\tstreamEndRun " << run.runNumber() << "\n";
    if(looper_) {
      //looper_->doStreamEndRun(schedule_->streamID(),runPrincipal, es);
    }
    {
      typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> Traits;
      schedule_->processOneGlobal<Traits>(runPrincipal, es, cleaningUpAfterException);
      if(hasSubProcess()) {
        subProcess_->doEndRun(runPrincipal, ts, cleaningUpAfterException);
      }
    }
    FDEBUG(1) << "\tendRun " << run.runNumber() << "\n";
    if(looper_) {
      looper_->doEndRun(runPrincipal, es, &processContext_);
    }
  }

  void EventProcessor::beginLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) {
    LuminosityBlockPrincipal& lumiPrincipal = principalCache_.lumiPrincipal(phid, run, lumi);
    input_->doBeginLumi(lumiPrincipal, &processContext_);

    Service<RandomNumberGenerator> rng;
    if(rng.isAvailable()) {
      LuminosityBlock lb(lumiPrincipal, ModuleDescription(), nullptr);
      rng->preBeginLumi(lb);
    }

    // NOTE: Using 0 as the event number for the begin of a lumi block is a bad idea
    // lumi blocks know their start and end times why not also start and end events?
    IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), 0), lumiPrincipal.beginTime());
    espController_->eventSetupForInstance(ts);
    EventSetup const& es = esp_->eventSetup();
    {
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Traits;
      schedule_->processOneGlobal<Traits>(lumiPrincipal, es);
      if(hasSubProcess()) {
        subProcess_->doBeginLuminosityBlock(lumiPrincipal, ts);
      }
    }
    FDEBUG(1) << "\tbeginLumi " << run << "/" << lumi << "\n";
    if(looper_) {
      looper_->doBeginLuminosityBlock(lumiPrincipal, es, &processContext_);
    }
    {
      for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
        typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> Traits;
        schedule_->processOneStream<Traits>(i,lumiPrincipal, es);
        if(hasSubProcess()) {
          subProcess_->doStreamBeginLuminosityBlock(i,lumiPrincipal, ts);
        }
      }
    }
    FDEBUG(1) << "\tstreamBeginLumi " << run << "/" << lumi << "\n";
    if(looper_) {
      //looper_->doStreamBeginLuminosityBlock(schedule_->streamID(),lumiPrincipal, es);
    }
  }

  void EventProcessor::endLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi, bool cleaningUpAfterException) {
    LuminosityBlockPrincipal& lumiPrincipal = principalCache_.lumiPrincipal(phid, run, lumi);
    input_->doEndLumi(lumiPrincipal, cleaningUpAfterException, &processContext_);
    //NOTE: Using the max event number for the end of a lumi block is a bad idea
    // lumi blocks know their start and end times why not also start and end events?
    IOVSyncValue ts(EventID(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), EventID::maxEventNumber()),
                    lumiPrincipal.endTime());
    espController_->eventSetupForInstance(ts);
    EventSetup const& es = esp_->eventSetup();
    {
      for(unsigned int i=0; i<preallocations_.numberOfStreams();++i) {
        typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> Traits;
        schedule_->processOneStream<Traits>(i,lumiPrincipal, es, cleaningUpAfterException);
        if(hasSubProcess()) {
          subProcess_->doStreamEndLuminosityBlock(i,lumiPrincipal, ts, cleaningUpAfterException);
        }
      }
    }
    FDEBUG(1) << "\tendLumi " << run << "/" << lumi << "\n";
    if(looper_) {
      //looper_->doStreamEndLuminosityBlock(schedule_->streamID(),lumiPrincipal, es);
    }
    {
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Traits;
      schedule_->processOneGlobal<Traits>(lumiPrincipal, es, cleaningUpAfterException);
      if(hasSubProcess()) {
        subProcess_->doEndLuminosityBlock(lumiPrincipal, ts, cleaningUpAfterException);
      }
    }
    FDEBUG(1) << "\tendLumi " << run << "/" << lumi << "\n";
    if(looper_) {
      looper_->doEndLuminosityBlock(lumiPrincipal, es, &processContext_);
    }
  }

  statemachine::Run EventProcessor::readRun() {
    if (principalCache_.hasRunPrincipal()) {
      throw edm::Exception(edm::errors::LogicError)
        << "EventProcessor::readRun\n"
        << "Illegal attempt to insert run into cache\n"
        << "Contact a Framework Developer\n";
    }
    boost::shared_ptr<RunPrincipal> rp(new RunPrincipal(input_->runAuxiliary(),
                                                        preg_,
                                                        *processConfiguration_,
                                                        historyAppender_.get(),
                                                        0));
    input_->readRun(*rp, *historyAppender_);
    principalCache_.insert(rp);
    return statemachine::Run(input_->reducedProcessHistoryID(), input_->run());
  }

  statemachine::Run EventProcessor::readAndMergeRun() {
    principalCache_.merge(input_->runAuxiliary(), preg_);
    input_->readAndMergeRun(*principalCache_.runPrincipalPtr());
    return statemachine::Run(input_->reducedProcessHistoryID(), input_->run());
  }

  int EventProcessor::readLuminosityBlock() {
    if (principalCache_.hasLumiPrincipal()) {
      throw edm::Exception(edm::errors::LogicError)
        << "EventProcessor::readRun\n"
        << "Illegal attempt to insert lumi into cache\n"
        << "Contact a Framework Developer\n";
    }
    if (!principalCache_.hasRunPrincipal()) {
      throw edm::Exception(edm::errors::LogicError)
        << "EventProcessor::readRun\n"
        << "Illegal attempt to insert lumi into cache\n"
        << "Run is invalid\n"
        << "Contact a Framework Developer\n";
    }
    boost::shared_ptr<LuminosityBlockPrincipal> lbp(new LuminosityBlockPrincipal(input_->luminosityBlockAuxiliary(),
                                                                                 preg_,
                                                                                 *processConfiguration_,
                                                                                 historyAppender_.get(),
                                                                                 0));
    input_->readLuminosityBlock(*lbp, *historyAppender_);
    lbp->setRunPrincipal(principalCache_.runPrincipalPtr());
    principalCache_.insert(lbp);
    return input_->luminosityBlock();
  }

  int EventProcessor::readAndMergeLumi() {
    principalCache_.merge(input_->luminosityBlockAuxiliary(), preg_);
    input_->readAndMergeLumi(*principalCache_.lumiPrincipalPtr());
    return input_->luminosityBlock();
  }

  void EventProcessor::writeRun(statemachine::Run const& run) {
    schedule_->writeRun(principalCache_.runPrincipal(run.processHistoryID(), run.runNumber()), &processContext_);
    if(hasSubProcess()) subProcess_->writeRun(run.processHistoryID(), run.runNumber());
    FDEBUG(1) << "\twriteRun " << run.runNumber() << "\n";
  }

  void EventProcessor::deleteRunFromCache(statemachine::Run const& run) {
    principalCache_.deleteRun(run.processHistoryID(), run.runNumber());
    if(hasSubProcess()) subProcess_->deleteRunFromCache(run.processHistoryID(), run.runNumber());
    FDEBUG(1) << "\tdeleteRunFromCache " << run.runNumber() << "\n";
  }

  void EventProcessor::writeLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) {
    schedule_->writeLumi(principalCache_.lumiPrincipal(phid, run, lumi), &processContext_);
    if(hasSubProcess()) subProcess_->writeLumi(phid, run, lumi);
    FDEBUG(1) << "\twriteLumi " << run << "/" << lumi << "\n";
  }

  void EventProcessor::deleteLumiFromCache(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) {
    principalCache_.deleteLumi(phid, run, lumi);
    if(hasSubProcess()) subProcess_->deleteLumiFromCache(phid, run, lumi);
    FDEBUG(1) << "\tdeleteLumiFromCache " << run << "/" << lumi << "\n";
  }

  void EventProcessor::readAndProcessEvent() {
    if(numberOfForkedChildren_>0) {
      readEvent(0);
      processEvent(0);
      return;
    }
    InputSource::ItemType itemType = InputSource::IsEvent;

    //While all the following item types are isEvent, process them right here
    asyncStopRequestedWhileProcessingEvents_ = false;
    
    //We will round-robin which stream to use
    unsigned int nextStreamIndex=0;
    const unsigned int kNumStreams = preallocations_.numberOfStreams();
    do {
      readEvent(nextStreamIndex);
      processEvent(nextStreamIndex);
      nextStreamIndex = (nextStreamIndex+1) % kNumStreams;
      
      if(shouldWeStop()) {
        break;
      }
      itemType = input_->nextItemType();
      if((asyncStopRequestedWhileProcessingEvents_=checkForAsyncStopRequest(asyncStopStatusCodeFromProcessingEvents_))) {
        break;
      }
    } while (itemType == InputSource::IsEvent);
    nextItemTypeFromProcessingEvents_ = itemType;
  }
  void EventProcessor::readEvent(unsigned int iStreamIndex) {
    //TODO this will have to become per stream
    auto& event = principalCache_.eventPrincipal(iStreamIndex);
    StreamContext streamContext(event.streamID(), &processContext_);
    input_->readEvent(event, &streamContext);
    FDEBUG(1) << "\treadEvent\n";
  }
  void EventProcessor::processEvent(unsigned int iStreamIndex) {
    auto pep = &(principalCache_.eventPrincipal(iStreamIndex));
    pep->setLuminosityBlockPrincipal(principalCache_.lumiPrincipalPtr());
    assert(pep->luminosityBlockPrincipalPtrValid());
    assert(principalCache_.lumiPrincipalPtr()->run() == pep->run());
    assert(principalCache_.lumiPrincipalPtr()->luminosityBlock() == pep->luminosityBlock());

    //We can only update IOVs on Lumi boundaries
    //IOVSyncValue ts(pep->id(), pep->time());
    //espController_->eventSetupForInstance(ts);
    EventSetup const& es = esp_->eventSetup();
    {
      typedef OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> Traits;
      schedule_->processOneEvent<Traits>(iStreamIndex,*pep, es);
      if(hasSubProcess()) {
        subProcess_->doEvent(*pep);
      }
    }

    //NOTE: If we have a looper we only have one Stream
    if(looper_) {
      bool randomAccess = input_->randomAccess();
      ProcessingController::ForwardState forwardState = input_->forwardState();
      ProcessingController::ReverseState reverseState = input_->reverseState();
      ProcessingController pc(forwardState, reverseState, randomAccess);

      EDLooperBase::Status status = EDLooperBase::kContinue;
      do {

        StreamContext streamContext(pep->streamID(), &processContext_);
        status = looper_->doDuringLoop(*pep, esp_->eventSetup(), pc, &streamContext);

        bool succeeded = true;
        if(randomAccess) {
          if(pc.requestedTransition() == ProcessingController::kToPreviousEvent) {
            input_->skipEvents(-2);
          }
          else if(pc.requestedTransition() == ProcessingController::kToSpecifiedEvent) {
            succeeded = input_->goToEvent(pc.specifiedEventTransition());
          }
        }
        pc.setLastOperationSucceeded(succeeded);
      } while(!pc.lastOperationSucceeded());
      if(status != EDLooperBase::kContinue) shouldWeStop_ = true;

    }

    FDEBUG(1) << "\tprocessEvent\n";
    pep->clearEventPrincipal();
  }

  bool EventProcessor::shouldWeStop() const {
    FDEBUG(1) << "\tshouldWeStop\n";
    if(shouldWeStop_) return true;
    return (schedule_->terminate() || (hasSubProcess() && subProcess_->terminate()));
  }

  void EventProcessor::setExceptionMessageFiles(std::string& message) {
    exceptionMessageFiles_ = message;
  }

  void EventProcessor::setExceptionMessageRuns(std::string& message) {
    exceptionMessageRuns_ = message;
  }

  void EventProcessor::setExceptionMessageLumis(std::string& message) {
    exceptionMessageLumis_ = message;
  }

  bool EventProcessor::alreadyHandlingException() const {
    return alreadyHandlingException_;
  }

  void EventProcessor::terminateMachine(std::auto_ptr<statemachine::Machine>& iMachine) {
    if(iMachine.get() != 0) {
      if(!iMachine->terminated()) {
        forceLooperToEnd_ = true;
        iMachine->process_event(statemachine::Stop());
        forceLooperToEnd_ = false;
      }
      else {
        FDEBUG(1) << "EventProcess::terminateMachine  The state machine was already terminated \n";
      }
      if(iMachine->terminated()) {
        FDEBUG(1) << "The state machine reports it has been terminated (3)\n";
      }
      iMachine.reset();
    }
  }
}
