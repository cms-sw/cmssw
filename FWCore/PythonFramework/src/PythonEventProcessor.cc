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
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

namespace {
   std::once_flag pluginFlag;
   int setupPluginSystem() {
      std::call_once(pluginFlag, []() {
         edmplugin::PluginManager::configure(edmplugin::standard::config());
         });
      return 0;
   }
}

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PythonEventProcessor::PythonEventProcessor(PythonProcessDesc const& iDesc)
: forcePluginSetupFirst_(setupPluginSystem())
  ,processor_(iDesc.processDesc(),edm::ServiceToken(),edm::serviceregistry::kOverlapIsError)
{
}

PythonEventProcessor::~PythonEventProcessor()
{
   try {
      processor_.endJob();
   }catch(...) {
      
   }
}

void
PythonEventProcessor::run()
{
   (void) processor_.runToCompletion();
}
