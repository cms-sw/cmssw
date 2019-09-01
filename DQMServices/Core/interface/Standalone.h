#ifndef DQMSERVICES_CORE_STANDALONE_H
#define DQMSERVICES_CORE_STANDALONE_H
#if !WITHOUT_CMS_FRAMEWORK
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#else
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace edm {
  std::string getReleaseVersion() { return "CMSSW_STANDALONE"; }

  class ParameterSet {
  public:
    template <class T>
    static const T &getUntrackedParameter(const char * /* key */, const T &value) {
      return value;
    }
  };

  struct ServiceToken {
    ServiceToken(int) {}
  };

  class ServiceRegistry {
  public:
    struct Operate {
      Operate(const ServiceToken &) {}
    };

    static int createSet(const std::vector<ParameterSet> &) { return 0; }
  };

  template <class T>
  class Service {
  public:
    bool isAvailable() { return false; }
    T *operator->() {
      static char buf[sizeof(T)];
      static T *x;
      if (!x)
        x = new (buf) T(ParameterSet());
      return x;
    }
    T &operator*() { return *operator->(); }
  };

  namespace service {
    struct SystemBounds {
      unsigned int maxNumberOfStreams() const { return 0; }
    };
  }  // namespace service

  struct PreallocationSignal {
    template <typename T>
    void connect(T &&){};
  };

  class ActivityRegistry {
  public:
    template <typename T>
    void watchPostSourceRun(void *, T) {}

    template <typename T>
    void watchPostSourceLumi(void *, T) {}

    template <typename F>
    void watchPostSourceRun(F) {}

    template <typename F>
    void watchPostSourceLumi(F) {}

    template <typename T>
    void watchPostGlobalBeginRun(void *, T) {}

    template <typename T>
    void watchPostGlobalBeginLumi(void *, T) {}

    template <typename T>
    void watchPostGlobalEndRun(void *, T) {}

    template <typename T>
    void watchPostGlobalEndLumi(void *, T) {}

    template <typename T>
    void watchPostModuleGlobalEndLumi(void *, T) {}

    template <typename F>
    void watchPostModuleGlobalEndLumi(F) {}

    template <typename T>
    void watchPostModuleGlobalEndRun(void *, T) {}

    template <typename F>
    void watchPostModuleGlobalEndRun(F) {}

    PreallocationSignal preallocateSignal_;
  };

  class LuminosityBlockID {
  public:
    unsigned int run() const { return 0; }
    unsigned int luminosityBlock() const { return 0; }
  };

  class GlobalContext {
  public:
    LuminosityBlockID luminosityBlockID() const { return LuminosityBlockID(); }
  };

  class ModuleDescription {
  public:
    unsigned int id() const { return 0; }
  };

  class ModuleCallingContext {
  public:
    ModuleDescription const *moduleDescription() const {
      static ModuleDescription md;
      return &md;
    }
  };

  class JobReport {
  public:
    JobReport(const edm::ParameterSet &) {}
    void reportAnalysisFile(const std::string &, const std::map<std::string, std::string> &) {}
  };

  class LuminosityBlockIndex {};

  class RunIndex {};

}  // namespace edm
#endif  // WITHOUT_CMS_FRAMEWORK
#endif  // DQMSERVICES_CORE_STANDALONE_H
