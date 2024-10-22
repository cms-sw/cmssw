// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Wed, 31 Jul 2013 15:59:19 GMT
//

// system include files
#include <cassert>

// user include files
#include "FWCore/Framework/interface/one/OutputModuleBase.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Common/interface/EndPathStatus.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/src/insertSelectedProcesses.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/ProcessBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/src/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"

namespace edm {
  namespace one {

    // -------------------------------------------------------
    OutputModuleBase::OutputModuleBase(ParameterSet const& pset) : core::OutputModuleCore(pset) {}

    void OutputModuleBase::configure(OutputModuleDescription const& desc) {
      core::OutputModuleCore::configure(desc);
      subProcessParentageHelper_ = desc.subProcessParentageHelper_;
    }

    SharedResourcesAcquirer OutputModuleBase::createAcquirer() {
      return SharedResourcesAcquirer{
          std::vector<std::shared_ptr<SerialTaskQueue>>(1, std::make_shared<SerialTaskQueue>())};
    }

    void OutputModuleBase::doPreallocate(PreallocationConfiguration const& iPC) {
      core::OutputModuleCore::doPreallocate_(iPC);
    }

    void OutputModuleBase::doBeginJob() {
      resourcesAcquirer_ = createAcquirer();
      core::OutputModuleCore::doBeginJob_();
    }

    bool OutputModuleBase::doEvent(EventTransitionInfo const& info,
                                   ActivityRegistry* act,
                                   ModuleCallingContext const* mcc) {
      { core::OutputModuleCore::doEvent_(info, act, mcc); }
      if (remainingEvents_ > 0) {
        --remainingEvents_;
      }
      return true;
    }
  }  // namespace one
}  // namespace edm
