#ifndef FWCore_Framework_ModuleContextSentry_h
#define FWCore_Framework_ModuleContextSentry_h

#include "FWCore/ServiceRegistry/interface/CurrentModuleOnThread.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {

  class ModuleContextSentry {
  public:
    ModuleContextSentry(ModuleCallingContext* moduleCallingContext, ParentContext const& parentContext)
        : moduleCallingContext_(moduleCallingContext) {
      moduleCallingContext_->setContext(
          ModuleCallingContext::State::kRunning, parentContext, CurrentModuleOnThread::getCurrentModuleOnThread());
      CurrentModuleOnThread::setCurrentModuleOnThread(moduleCallingContext_);
    }
    ~ModuleContextSentry() {
      CurrentModuleOnThread::setCurrentModuleOnThread(moduleCallingContext_->previousModuleOnThread());
      moduleCallingContext_->setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }

  private:
    edm::propagate_const<ModuleCallingContext*> moduleCallingContext_;
  };
}  // namespace edm
#endif
