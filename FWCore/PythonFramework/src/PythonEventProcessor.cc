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

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "FWCore/MessageLogger/interface/JobReport.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
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
      processor_(addDefaultServicesToProcessDesc(iDesc.processDesc()),
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
