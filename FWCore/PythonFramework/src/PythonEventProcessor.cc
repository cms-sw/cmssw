// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     PythonEventProcessor
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Fri, 20 Jan 2017 16:36:41 GMT
//

// system include files
#include <mutex>
#include "tbb/task_arena.h"

// user include files
#include "FWCore/PythonFramework/interface/PythonEventProcessor.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"

#include "FWCore/Framework/interface/defaultCmsRunServices.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ThreadsInfo.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "FWCore/MessageLogger/interface/JobReport.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Concurrency/interface/setNThreads.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace {
  std::once_flag pluginFlag;
  int setupPluginSystem() {
    std::call_once(pluginFlag, []() { edmplugin::PluginManager::configure(edmplugin::standard::config()); });
    return 0;
  }

  std::shared_ptr<edm::ProcessDesc> addDefaultServicesToProcessDesc(std::shared_ptr<edm::ProcessDesc> iDesc) {
    iDesc->addServices(edm::defaultCmsRunServices());
    return iDesc;
  }

  //Only one ThreadsController can be active at a time
  CMS_THREAD_SAFE std::unique_ptr<edm::ThreadsController> tsiPtr;
  CMS_THREAD_SAFE int nThreads;

  std::shared_ptr<edm::ProcessDesc> setupThreading(std::shared_ptr<edm::ProcessDesc> iDesc) {
    // check the "options" ParameterSet
    std::shared_ptr<edm::ParameterSet> pset = iDesc->getProcessPSet();
    auto threadsInfo = threadOptions(*pset);

    threadsInfo.nThreads_ = edm::setNThreads(threadsInfo.nThreads_, threadsInfo.stackSize_, tsiPtr);
    nThreads = threadsInfo.nThreads_;

    // update the numberOfThreads and sizeOfStackForThreadsInKB in the "options" ParameterSet
    setThreadOptions(threadsInfo, *pset);

    return iDesc;
  }

  edm::ServiceToken createJobReport() {
    return edm::ServiceRegistry::createContaining(
        std::make_shared<edm::serviceregistry::ServiceWrapper<edm::JobReport>>(
            std::make_unique<edm::JobReport>(nullptr)));
  }
}  // namespace

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PythonEventProcessor::PythonEventProcessor(PyBind11ProcessDesc const& iDesc)
    : forcePluginSetupFirst_(setupPluginSystem()),
      processor_(addDefaultServicesToProcessDesc(setupThreading(iDesc.processDesc())),
                 createJobReport(),
                 edm::serviceregistry::kOverlapIsError) {}

namespace {
  class TaskCleanupSentry {
  public:
    TaskCleanupSentry(edm::EventProcessor* ep) : ep_(ep) {}
    ~TaskCleanupSentry() { ep_->taskCleanup(); }

  private:
    edm::EventProcessor* ep_;
  };
}  // namespace

PythonEventProcessor::~PythonEventProcessor() {
  auto gil = PyEval_SaveThread();
  // Protects the destructor from throwing exceptions.
  CMS_SA_ALLOW try {
    tbb::task_arena{nThreads}.execute([this]() {
      TaskCleanupSentry s(&processor_);
      processor_.endJob();
    });
  } catch (...) {
  }
  PyEval_RestoreThread(gil);
}

void PythonEventProcessor::run() {
  auto gil = PyEval_SaveThread();
  try {
    tbb::task_arena{nThreads}.execute([this]() { (void)processor_.runToCompletion(); });
  } catch (...) {
  }
  PyEval_RestoreThread(gil);
}
