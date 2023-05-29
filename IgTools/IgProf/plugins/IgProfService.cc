
//
//  Description: FWK service to implement hook for igprof memory profile
//               dump functionality
//
//  Peter Elmer, Princeton University                        18 Nov, 2008
//

#include "IgTools/IgProf/plugins/IgProfService.h"
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

using namespace edm::service;

IgProfService::IgProfService(ParameterSet const &ps, ActivityRegistry &iRegistry)
    : dump_(nullptr),
      mineventrecord_(1),
      prescale_(1),
      nrecord_(0),
      nevent_(0),
      nrun_(0),
      nlumi_(0),
      nfileopened_(0),
      nfileclosed_(0) {
  // Removing the __extension__ gives a warning which
  // is acknowledged as a language problem in the C++ Standard Core
  // Language Defect Report
  //
  // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#195
  //
  // since the suggested decision seems to be that the syntax should
  // actually be "Conditionally-Supported Behavior" in some
  // future C++ standard I simply silence the warning.
  if (void *sym = dlsym(nullptr, "igprof_dump_now")) {
    dump_ = __extension__(void (*)(const char *)) sym;
  } else
    edm::LogWarning("IgProfModule") << "IgProfModule requested but application is not"
                                    << " currently being profiled with igprof\n";

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
  iRegistry.watchPostBeginJob(this, &IgProfService::postBeginJob);
  iRegistry.watchPostGlobalBeginRun(this, &IgProfService::postBeginRun);
  iRegistry.watchPostGlobalBeginLumi(this, &IgProfService::postBeginLumi);

  iRegistry.watchPreEvent(this, &IgProfService::preEvent);
  iRegistry.watchPostEvent(this, &IgProfService::postEvent);

  if (not modules_.empty() or not moduleTypes_.empty()) {
    iRegistry.watchPreModuleEvent(this, &IgProfService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &IgProfService::postModuleEvent);
  }

  iRegistry.watchPostGlobalEndLumi(this, &IgProfService::postEndLumi);
  iRegistry.watchPreGlobalEndRun(this, &IgProfService::preEndRun);
  iRegistry.watchPostGlobalEndRun(this, &IgProfService::postEndRun);
  iRegistry.watchPreEndProcessBlock(this, &IgProfService::preEndProcessBlock);
  iRegistry.watchPostEndProcessBlock(this, &IgProfService::postEndProcessBlock);
  iRegistry.watchPreEndJob(this, &IgProfService::preEndJob);
  iRegistry.watchPostEndJob(this, &IgProfService::postEndJob);

  iRegistry.watchPostOpenFile(this, &IgProfService::postOpenFile);
  iRegistry.watchPostCloseFile(this, &IgProfService::postCloseFile);
}

void IgProfService::postBeginJob() { makeDump(atPostBeginJob_); }

void IgProfService::postBeginRun(GlobalContext const &gc) {
  nrun_ = gc.luminosityBlockID().run();
  makeDump(atPostBeginRun_);
}

void IgProfService::postBeginLumi(GlobalContext const &gc) {
  nlumi_ = gc.luminosityBlockID().luminosityBlock();
  makeDump(atPostBeginLumi_);
}

void IgProfService::preEvent(StreamContext const &iStream) {
  ++nrecord_;  // count before events
  nevent_ = iStream.eventID().event();
  if ((prescale_ > 0) && (nrecord_ >= mineventrecord_) && (((nrecord_ - mineventrecord_) % prescale_) == 0))
    makeDump(atPreEvent_);
}

void IgProfService::postEvent(StreamContext const &iStream) {
  nevent_ = iStream.eventID().event();
  if ((prescale_ > 0) && (nrecord_ >= mineventrecord_) && (((nrecord_ - mineventrecord_) % prescale_) == 0))
    makeDump(atPostEvent_);
}

void IgProfService::preModuleEvent(StreamContext const &iStream, ModuleCallingContext const &mcc) {
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

void IgProfService::postModuleEvent(StreamContext const &iStream, ModuleCallingContext const &mcc) {
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

void IgProfService::postEndLumi(GlobalContext const &gc) {
  nlumi_ = gc.luminosityBlockID().luminosityBlock();
  makeDump(atPostEndLumi_);
}

void IgProfService::preEndRun(GlobalContext const &gc) {
  nrun_ = gc.luminosityBlockID().run();
  makeDump(atPreEndRun_);
}

void IgProfService::postEndRun(GlobalContext const &gc) {
  nrun_ = gc.luminosityBlockID().run();
  makeDump(atPostEndRun_);
}

void IgProfService::preEndProcessBlock(GlobalContext const &gc) { makeDump(atPreEndProcessBlock_); }

void IgProfService::postEndProcessBlock(GlobalContext const &gc) { makeDump(atPostEndProcessBlock_); }

void IgProfService::preEndJob() { makeDump(atPreEndJob_); }

void IgProfService::postEndJob() { makeDump(atPostEndJob_); }

void IgProfService::postOpenFile(std::string const &) {
  ++nfileopened_;
  makeDump(atPostOpenFile_);
}

void IgProfService::postCloseFile(std::string const &) {
  ++nfileclosed_;
  makeDump(atPostCloseFile_);
}

void IgProfService::makeDump(const std::string &format, std::string_view moduleLabel) {
  if (!dump_ || format.empty())
    return;

  std::string final(format);
  final = replace(final, "%I", nrecord_);
  final = replaceU64(final, "%E", nevent_);
  final = replaceU64(final, "%R", nrun_);
  final = replaceU64(final, "%L", nlumi_);
  final = replace(final, "%F", nfileopened_);
  final = replace(final, "%C", nfileclosed_);
  final = replace(final, "%M", moduleLabel);
  dump_(final.c_str());
}

std::string IgProfService::replace(const std::string &s, const char *pat, int val) {
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

std::string IgProfService::replaceU64(const std::string &s, const char *pat, unsigned long long val) {
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

std::string IgProfService::replace(const std::string &s, const char *pat, std::string_view val) {
  size_t pos = 0;
  size_t patlen = strlen(pat);
  std::string result = s;
  while ((pos = result.find(pat, pos)) != std::string::npos) {
    result.replace(pos, patlen, val.data());
    pos = pos - patlen + val.size();
  }

  return result;
}

DEFINE_FWK_SERVICE(IgProfService);
