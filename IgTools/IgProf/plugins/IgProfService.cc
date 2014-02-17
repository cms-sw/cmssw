
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

IgProfService::IgProfService(ParameterSet const& ps, 
                             ActivityRegistry&iRegistry)
  : dump_(0),
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
    if (void *sym = dlsym(0, "igprof_dump_now")) {
      dump_ = __extension__ (void(*)(const char *)) sym;
    } else
      edm::LogWarning("IgProfModule")
        << "IgProfModule requested but application is not"
        << " currently being profiled with igprof\n";

  // Get the configuration
  prescale_    
    = ps.getUntrackedParameter<int>("reportEventInterval", prescale_);
  mineventrecord_    
    = ps.getUntrackedParameter<int>("reportFirstEvent", mineventrecord_);

  atPostBeginJob_  
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostBeginJob", atPostBeginJob_);
  atPostBeginRun_ 
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostBeginRun", atPostBeginRun_);
  atPostBeginLumi_ 
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostBeginLumi", atPostBeginLumi_);

  atPreEvent_     
    = ps.getUntrackedParameter<std::string>("reportToFileAtPreEvent", atPreEvent_);
  atPostEvent_     
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostEvent", atPostEvent_);

  atPostEndLumi_ 
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostEndLumi", atPostEndLumi_);
  atPostEndRun_ 
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostEndRun", atPostEndRun_);
  atPostEndJob_  
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostEndJob", atPostEndJob_);

  atPostOpenFile_ 
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostOpenFile", atPostOpenFile_);
  atPostCloseFile_ 
    = ps.getUntrackedParameter<std::string>("reportToFileAtPostCloseFile", atPostCloseFile_);


  // Register for the framework signals
  iRegistry.watchPostBeginJob(this, &IgProfService::postBeginJob);
  iRegistry.watchPostBeginRun(this, &IgProfService::postBeginRun);
  iRegistry.watchPostBeginLumi(this, &IgProfService::postBeginLumi);

  iRegistry.watchPreProcessEvent(this, &IgProfService::preEvent);
  iRegistry.watchPostProcessEvent(this, &IgProfService::postEvent);

  iRegistry.watchPostEndLumi(this, &IgProfService::postEndLumi);
  iRegistry.watchPostEndRun(this, &IgProfService::postEndRun);
  iRegistry.watchPostEndJob(this, &IgProfService::postEndJob);

  iRegistry.watchPostOpenFile(this, &IgProfService::postOpenFile);
  iRegistry.watchPostCloseFile(this, &IgProfService::postCloseFile);

}

void IgProfService::postBeginJob() { 
  makeDump(atPostBeginJob_); 
}

void IgProfService::postBeginRun(const edm::Run &r, const edm::EventSetup &) { 
  nrun_ = r.run(); makeDump(atPostBeginRun_); 
}

void IgProfService::postBeginLumi(const edm::LuminosityBlock &l, 
                                  const edm::EventSetup &) { 
  nlumi_ = l.luminosityBlock(); makeDump(atPostBeginLumi_); 
}

void IgProfService::preEvent(const edm::EventID &id, 
                             const edm::Timestamp & iTime) {
  ++nrecord_; // count before events
  nevent_ = id.event();
  if ((prescale_ > 0) && 
      (nrecord_ >= mineventrecord_) &&
      (((nrecord_ - mineventrecord_)% prescale_) == 0)) makeDump(atPreEvent_);
}

void IgProfService::postEvent(const edm::Event &e, const edm::EventSetup &) {
  nevent_ = e.id().event();
  if ((prescale_ > 0) && 
      (nrecord_ >= mineventrecord_) &&
      (((nrecord_ - mineventrecord_)% prescale_) == 0)) makeDump(atPostEvent_);
}

void IgProfService::postEndLumi(const edm::LuminosityBlock &l, 
                                const edm::EventSetup &) { 
  makeDump(atPostEndLumi_); 
}

void IgProfService::postEndRun(const edm::Run &, const edm::EventSetup &) { 
  makeDump(atPostEndRun_); 
}

void IgProfService::postEndJob() { 
  makeDump(atPostEndJob_); 
}

void IgProfService::postOpenFile () {
  ++nfileopened_; 
  makeDump(atPostOpenFile_);
}  

void IgProfService::postCloseFile () {
  ++nfileclosed_; 
  makeDump(atPostCloseFile_);
}  

void IgProfService::makeDump(const std::string &format) {
  if (! dump_ || format.empty())
    return;

  std::string final(format);
  final = replace(final, "%I", nrecord_);
  final = replace(final, "%E", nevent_);
  final = replace(final, "%R", nrun_);
  final = replace(final, "%L", nlumi_);
  final = replace(final, "%F", nfileopened_);
  final = replace(final, "%C", nfileclosed_);
  dump_(final.c_str());
}

std::string 
IgProfService::replace(const std::string &s, const char *pat, int val) {
  size_t pos = 0;
  size_t patlen = strlen(pat);
  std::string result = s;
  while ((pos = result.find(pat, pos)) != std::string::npos)
  {
    char buf[64];
    int n = sprintf(buf, "%d", val);
    result.replace(pos, patlen, buf);
    pos = pos - patlen + n;
  }

  return result;
}

DEFINE_FWK_SERVICE(IgProfService);

