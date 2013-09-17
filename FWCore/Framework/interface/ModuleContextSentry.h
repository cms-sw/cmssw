#ifndef FWCore_Framework_ModuleContextSentry_h
#define FWCore_Framework_ModuleContextSentry_h

#include "FWCore/Framework/interface/CurrentModuleOnThread.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"

namespace edm {

  class ModuleContextSentry {
  public:
    ModuleContextSentry(ModuleCallingContext* moduleCallingContext,
                        ParentContext const& parentContext) :
      moduleCallingContext_(moduleCallingContext) {
      moduleCallingContext_->setContext(ModuleCallingContext::State::kRunning, parentContext,
                                        CurrentModuleOnThread::getCurrentModuleOnThread());
      CurrentModuleOnThread::setCurrentModuleOnThread(moduleCallingContext_);
    }
    ~ModuleContextSentry() {
      CurrentModuleOnThread::setCurrentModuleOnThread(moduleCallingContext_->previousModuleOnThread());
      moduleCallingContext_->setContext(ModuleCallingContext::State::kInvalid, ParentContext(), nullptr);
    }
  private:
    ModuleCallingContext* moduleCallingContext_;
  };
}
#endif
