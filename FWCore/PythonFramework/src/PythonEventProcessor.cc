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

  //TBB only allows 1 task_scheduler_init active on a thread.
  CMS_THREAD_SAFE std::unique_ptr<tbb::task_scheduler_init> tsiPtr;

  std::shared_ptr<edm::ProcessDesc> setupThreading(std::shared_ptr<edm::ProcessDesc> iDesc) {
    // check the "options" ParameterSet
    std::shared_ptr<edm::ParameterSet> pset = iDesc->getProcessPSet();
    auto threadsInfo = threadOptions(*pset);

    threadsInfo.nThreads_ = edm::setNThreads(threadsInfo.nThreads_, threadsInfo.stackSize_, tsiPtr);

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

PythonEventProcessor::~PythonEventProcessor() {
  auto gil = PyEval_SaveThread();
  // Protects the destructor from throwing exceptions.
  CMS_SA_ALLOW try { processor_.endJob(); } catch (...) {
  }
  PyEval_RestoreThread(gil);
}

void PythonEventProcessor::run() {
  auto gil = PyEval_SaveThread();
  try {
    (void)processor_.runToCompletion();
  } catch (...) {
  }
  PyEval_RestoreThread(gil);
}
