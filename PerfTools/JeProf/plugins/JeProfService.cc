
//
//  Description: FWK service to implement hook for jemalloc heap profile
//               dump functionality
//

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include <string>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>

extern "C" {
typedef int (*mallctl_t)(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
}

namespace {
  bool initialize_prof();

  mallctl_t mallctl = nullptr;
  const bool have_jemalloc_and_prof = initialize_prof();

  bool initialize_prof() {
    // check if mallctl and friends are available, if we are using jemalloc
    mallctl = (mallctl_t)::dlsym(RTLD_DEFAULT, "mallctl");
    if (mallctl == nullptr)
      return false;
    // check if heap profiling available, if --enable-prof was specified at build time
    bool enable_prof = false;
    size_t bool_s = sizeof(bool);
    mallctl("prof.active", &enable_prof, &bool_s, nullptr, 0);
    return enable_prof;
  }

}  // namespace

namespace edm {
  class GlobalContext;
  class StreamContext;

  namespace service {
    class JeProfService {
    public:
      JeProfService(const ParameterSet &, ActivityRegistry &);

      void postBeginJob();

      void postBeginRun(GlobalContext const &gc);

      void postBeginLumi(GlobalContext const &gc);

      void preEvent(StreamContext const &sc);
      void postEvent(StreamContext const &sc);

      void preModuleEvent(StreamContext const &sc, ModuleCallingContext const &mcc);
      void postModuleEvent(StreamContext const &sc, ModuleCallingContext const &mcc);

      void postEndLumi(GlobalContext const &gc);

      void preEndRun(GlobalContext const &gc);
      void postEndRun(GlobalContext const &gc);

      void preEndProcessBlock(GlobalContext const &gc);
      void postEndProcessBlock(GlobalContext const &gc);

      void preEndJob();
      void postEndJob();

      void postOpenFile(std::string const &);

      void postCloseFile(std::string const &);

    private:
      void makeDump(const std::string &format, std::string_view moduleLabel = "");
      static std::string replace(const std::string &s, const char *pat, int val);
      static std::string replaceU64(const std::string &s, const char *pat, unsigned long long val);
      static std::string replace(const std::string &s, const char *pat, std::string_view val);

      std::string atPostBeginJob_;
      std::string atPostBeginRun_;
      std::string atPostBeginLumi_;

      std::string atPreEvent_;
      std::string atPostEvent_;

      std::vector<std::string> modules_;
      std::vector<std::string> moduleTypes_;
      std::string atPreModuleEvent_;
      std::string atPostModuleEvent_;

      std::string atPostEndLumi_;
      std::string atPreEndRun_;
      std::string atPostEndRun_;
      std::string atPreEndProcessBlock_;
      std::string atPostEndProcessBlock_;
      std::string atPreEndJob_;
      std::string atPostEndJob_;

      std::string atPostOpenFile_;
      std::string atPostCloseFile_;

      int mineventrecord_;
      int prescale_;
      int nrecord_;  // counter
      edm::EventNumber_t nevent_;
      edm::RunNumber_t nrun_;
      edm::LuminosityBlockNumber_t nlumi_;
      int nfileopened_;  // counter of files opened thus far
      int nfileclosed_;  // counter of files closed thus far
    };
    inline bool isProcessWideService(JeProfService const *) { return true; }
  }  // namespace service
}  // namespace edm

using namespace edm::service;

JeProfService::JeProfService(ParameterSet const &ps, ActivityRegistry &iRegistry)
    : mineventrecord_(1), prescale_(1), nrecord_(0), nevent_(0), nrun_(0), nlumi_(0), nfileopened_(0), nfileclosed_(0) {
  if (!have_jemalloc_and_prof) {
    edm::LogWarning("JeProfModule") << "JeProfModule requested but application is not"
                                    << " currently being profiled with jemalloc profiling\n";
  }
  // Get the configuration
  prescale_ = ps.getUntrackedParameter<int>("reportEventInterval", prescale_);
  mineventrecord_ = ps.getUntrackedParameter<int>("reportFirstEvent", mineventrecord_);

  atPostBeginJob_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostBeginJob", atPostBeginJob_);
  atPostBeginRun_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostBeginRun", atPostBeginRun_);
  atPostBeginLumi_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostBeginLumi", atPostBeginLumi_);

  atPreEvent_ = ps.getUntrackedParameter<std::string>("reportToFileAtPreEvent", atPreEvent_);
  atPostEvent_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostEvent", atPostEvent_);

  modules_ = ps.getUntrackedParameter<std::vector<std::string>>("reportModules", modules_);
  moduleTypes_ = ps.getUntrackedParameter<std::vector<std::string>>("reportModuleTypes", moduleTypes_);
  std::sort(modules_.begin(), modules_.end());
  std::sort(moduleTypes_.begin(), moduleTypes_.end());
  atPreModuleEvent_ = ps.getUntrackedParameter<std::string>("reportToFileAtPreModuleEvent", atPreModuleEvent_);
  atPostModuleEvent_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostModuleEvent", atPostModuleEvent_);

  atPostEndLumi_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostEndLumi", atPostEndLumi_);
  atPreEndRun_ = ps.getUntrackedParameter<std::string>("reportToFileAtPreEndRun", atPreEndRun_);
  atPostEndRun_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostEndRun", atPostEndRun_);
  atPreEndProcessBlock_ =
      ps.getUntrackedParameter<std::string>("reportToFileAtPreEndProcessBlock", atPreEndProcessBlock_);
  atPostEndProcessBlock_ =
      ps.getUntrackedParameter<std::string>("reportToFileAtPostEndProcessBlock", atPostEndProcessBlock_);
  atPreEndJob_ = ps.getUntrackedParameter<std::string>("reportToFileAtPreEndJob", atPreEndJob_);
  atPostEndJob_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostEndJob", atPostEndJob_);

  atPostOpenFile_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostOpenFile", atPostOpenFile_);
  atPostCloseFile_ = ps.getUntrackedParameter<std::string>("reportToFileAtPostCloseFile", atPostCloseFile_);

  // Register for the framework signals
  iRegistry.watchPostBeginJob(this, &JeProfService::postBeginJob);
  iRegistry.watchPostGlobalBeginRun(this, &JeProfService::postBeginRun);
  iRegistry.watchPostGlobalBeginLumi(this, &JeProfService::postBeginLumi);

  iRegistry.watchPreEvent(this, &JeProfService::preEvent);
  iRegistry.watchPostEvent(this, &JeProfService::postEvent);

  if (not modules_.empty() or not moduleTypes_.empty()) {
    iRegistry.watchPreModuleEvent(this, &JeProfService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &JeProfService::postModuleEvent);
  }

  iRegistry.watchPostGlobalEndLumi(this, &JeProfService::postEndLumi);
  iRegistry.watchPreGlobalEndRun(this, &JeProfService::preEndRun);
  iRegistry.watchPostGlobalEndRun(this, &JeProfService::postEndRun);
  iRegistry.watchPreEndProcessBlock(this, &JeProfService::preEndProcessBlock);
  iRegistry.watchPostEndProcessBlock(this, &JeProfService::postEndProcessBlock);
  iRegistry.watchPreEndJob(this, &JeProfService::preEndJob);
  iRegistry.watchPostEndJob(this, &JeProfService::postEndJob);

  iRegistry.watchPostOpenFile(this, &JeProfService::postOpenFile);
  iRegistry.watchPostCloseFile(this, &JeProfService::postCloseFile);
}

void JeProfService::postBeginJob() { makeDump(atPostBeginJob_); }

void JeProfService::postBeginRun(GlobalContext const &gc) {
  nrun_ = gc.luminosityBlockID().run();
  makeDump(atPostBeginRun_);
}

void JeProfService::postBeginLumi(GlobalContext const &gc) {
  nlumi_ = gc.luminosityBlockID().luminosityBlock();
  makeDump(atPostBeginLumi_);
}

void JeProfService::preEvent(StreamContext const &iStream) {
  ++nrecord_;  // count before events
  nevent_ = iStream.eventID().event();
  if ((prescale_ > 0) && (nrecord_ >= mineventrecord_) && (((nrecord_ - mineventrecord_) % prescale_) == 0))
    makeDump(atPreEvent_);
}

void JeProfService::postEvent(StreamContext const &iStream) {
  nevent_ = iStream.eventID().event();
  if ((prescale_ > 0) && (nrecord_ >= mineventrecord_) && (((nrecord_ - mineventrecord_) % prescale_) == 0))
    makeDump(atPostEvent_);
}

void JeProfService::preModuleEvent(StreamContext const &iStream, ModuleCallingContext const &mcc) {
  nevent_ = iStream.eventID().event();
  if ((prescale_ > 0) && (nrecord_ >= mineventrecord_) && (((nrecord_ - mineventrecord_) % prescale_) == 0)) {
    auto const &moduleLabel = mcc.moduleDescription()->moduleLabel();
    auto const &moduleType = mcc.moduleDescription()->moduleName();
    if (std::binary_search(modules_.begin(), modules_.end(), moduleLabel) or
        std::binary_search(moduleTypes_.begin(), moduleTypes_.end(), moduleType)) {
      makeDump(atPreModuleEvent_, moduleLabel);
    }
  }
}

void JeProfService::postModuleEvent(StreamContext const &iStream, ModuleCallingContext const &mcc) {
  nevent_ = iStream.eventID().event();
  if ((prescale_ > 0) && (nrecord_ >= mineventrecord_) && (((nrecord_ - mineventrecord_) % prescale_) == 0)) {
    auto const &moduleLabel = mcc.moduleDescription()->moduleLabel();
    auto const &moduleType = mcc.moduleDescription()->moduleName();
    if (std::binary_search(modules_.begin(), modules_.end(), moduleLabel) or
        std::binary_search(moduleTypes_.begin(), moduleTypes_.end(), moduleType)) {
      makeDump(atPostModuleEvent_, moduleLabel);
    }
  }
}

void JeProfService::postEndLumi(GlobalContext const &gc) {
  nlumi_ = gc.luminosityBlockID().luminosityBlock();
  makeDump(atPostEndLumi_);
}

void JeProfService::preEndRun(GlobalContext const &gc) {
  nrun_ = gc.luminosityBlockID().run();
  makeDump(atPreEndRun_);
}

void JeProfService::postEndRun(GlobalContext const &gc) {
  nrun_ = gc.luminosityBlockID().run();
  makeDump(atPostEndRun_);
}

void JeProfService::preEndProcessBlock(GlobalContext const &gc) { makeDump(atPreEndProcessBlock_); }

void JeProfService::postEndProcessBlock(GlobalContext const &gc) { makeDump(atPostEndProcessBlock_); }

void JeProfService::preEndJob() { makeDump(atPreEndJob_); }

void JeProfService::postEndJob() { makeDump(atPostEndJob_); }

void JeProfService::postOpenFile(std::string const &) {
  ++nfileopened_;
  makeDump(atPostOpenFile_);
}

void JeProfService::postCloseFile(std::string const &) {
  ++nfileclosed_;
  makeDump(atPostCloseFile_);
}

void JeProfService::makeDump(const std::string &format, std::string_view moduleLabel) {
  if (!have_jemalloc_and_prof || format.empty())
    return;

  std::string final(format);
  final = replace(final, "%I", nrecord_);
  final = replaceU64(final, "%E", nevent_);
  final = replaceU64(final, "%R", nrun_);
  final = replaceU64(final, "%L", nlumi_);
  final = replace(final, "%F", nfileopened_);
  final = replace(final, "%C", nfileclosed_);
  final = replace(final, "%M", moduleLabel);
  const char *fileName = final.c_str();
  mallctl("prof.dump", nullptr, nullptr, &fileName, sizeof(const char *));
}

std::string JeProfService::replace(const std::string &s, const char *pat, int val) {
  size_t pos = 0;
  size_t patlen = strlen(pat);
  std::string result = s;
  while ((pos = result.find(pat, pos)) != std::string::npos) {
    char buf[64];
    int n = sprintf(buf, "%d", val);
    result.replace(pos, patlen, buf);
    pos = pos - patlen + n;
  }

  return result;
}

std::string JeProfService::replaceU64(const std::string &s, const char *pat, unsigned long long val) {
  size_t pos = 0;
  size_t patlen = strlen(pat);
  std::string result = s;
  while ((pos = result.find(pat, pos)) != std::string::npos) {
    char buf[64];
    int n = sprintf(buf, "%llu", val);
    result.replace(pos, patlen, buf);
    pos = pos - patlen + n;
  }

  return result;
}

std::string JeProfService::replace(const std::string &s, const char *pat, std::string_view val) {
  size_t pos = 0;
  size_t patlen = strlen(pat);
  std::string result = s;
  while ((pos = result.find(pat, pos)) != std::string::npos) {
    result.replace(pos, patlen, val.data());
    pos = pos - patlen + val.size();
  }

  return result;
}

DEFINE_FWK_SERVICE(JeProfService);
