#ifndef FWCore_Framework_ModuleSignalSentry_h
#define FWCore_Framework_ModuleSignalSentry_h

#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
namespace edm {
  class ActivityRegistry;
  class ModuleCallingContext;

  template <typename T>
  class ModuleSignalSentry {
  public:
    ModuleSignalSentry(ActivityRegistry* a,
                       typename T::Context const* context,
                       ModuleCallingContext const* moduleCallingContext)
        : a_(a), context_(context), moduleCallingContext_(moduleCallingContext) {}

    ~ModuleSignalSentry() {
      // This destructor does nothing unless we are unwinding the
      // the stack from an earlier exception (a_ will be null if we are
      // are not). We want to report the earlier exception and ignore any
      // addition exceptions from the post module signal.
      CMS_SA_ALLOW try {
        if (a_) {
          T::postModuleSignal(a_, context_, moduleCallingContext_);
        }
      } catch (...) {
      }
    }
    void preModuleSignal() {
      if (a_) {
        try {
          convertException::wrap([this]() { T::preModuleSignal(a_, context_, moduleCallingContext_); });
        } catch (cms::Exception& ex) {
          ex.addContext("Handling pre module signal, likely in a service function immediately before module method");
          throw;
        }
      }
    }
    void postModuleSignal() {
      if (a_) {
        auto temp = a_;
        // Setting a_ to null informs the destructor that the signal
        // was already run and that it should do nothing.
        a_ = nullptr;
        try {
          convertException::wrap([this, temp]() { T::postModuleSignal(temp, context_, moduleCallingContext_); });
        } catch (cms::Exception& ex) {
          ex.addContext("Handling post module signal, likely in a service function immediately after module method");
          throw;
        }
      }
    }

  private:
    ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
    typename T::Context const* context_;
    ModuleCallingContext const* moduleCallingContext_;
  };
}  // namespace edm

#endif
