#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include <string>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>

class IgProfModule : public edm::EDAnalyzer
{
public:
  IgProfModule(const edm::ParameterSet &ps)
    : dump_(0),
      prescale_(0),
      nrecord_(0),
      nevent_(0),
      nrun_(0),
      nlumi_(0),
      nfile_(0)
    {
      // Removing the __extension__ gives a warning which
      // is acknowledged as a language problem in the C++ Standard Core 
      // Language Defect Report
      //
      // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#195
      //
      // since the suggested decision seems to be that the syntax should
      // actually be "Conditionally-Supported Behavior" in some 
      // future C++ standard I simply silence the warning.
      if (void *sym = dlsym(0, "igprof_dump_now"))
        dump_ = __extension__ (void(*)(const char *)) sym;
      else
	edm::LogWarning("IgProfModule")
	  << "IgProfModule requested but application is not"
	  << " currently being profiled with igprof\n";

      prescale_    = ps.getUntrackedParameter<int>("reportEventInterval", prescale_);
      atBeginJob_  = ps.getUntrackedParameter<std::string>("reportToFileAtBeginJob", atBeginJob_);
      atEndJob_    = ps.getUntrackedParameter<std::string>("reportToFileAtEndJob", atEndJob_);
      atBeginLumi_ = ps.getUntrackedParameter<std::string>("reportToFileAtBeginLumi", atBeginLumi_);
      atEndLumi_   = ps.getUntrackedParameter<std::string>("reportToFileAtEndLumi", atEndLumi_);
      atInputFile_ = ps.getUntrackedParameter<std::string>("reportToFileAtInputFile", atInputFile_);
      atEvent_     = ps.getUntrackedParameter<std::string>("reportToFileAtEvent", atEvent_);
    }

  virtual void beginJob()
    { makeDump(atBeginJob_); }

  virtual void endJob(void)
    { makeDump(atEndJob_); }

  virtual void analyze(const edm::Event &e, const edm::EventSetup &)
    {
      nevent_ = e.id().event();
      if (prescale_ > 0 && (++nrecord_ % prescale_) == 1)
        makeDump(atEvent_);
    }

  virtual void beginRun(const edm::Run &r, const edm::EventSetup &)
    { nrun_ = r.run(); makeDump(atBeginRun_); }

  virtual void endRun(const edm::Run &, const edm::EventSetup &)
    { makeDump(atEndRun_); }

  virtual void beginLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &)
    { nlumi_ = l.luminosityBlock(); makeDump(atBeginLumi_); }

  virtual void endLuminosityBlock(const edm::LuminosityBlock &l, const edm::EventSetup &)
    { makeDump(atEndLumi_); }

  virtual void respondToOpenInputFile(const edm::FileBlock &)
    { ++nfile_; makeDump(atInputFile_); }

private:
  void makeDump(const std::string &format)
    {
      if (! dump_ || format.empty())
	return;

      std::string final(format);
      final = replace(final, "%I", nrecord_);
      final = replace(final, "%E", nevent_);
      final = replace(final, "%R", nrun_);
      final = replace(final, "%L", nlumi_);
      final = replace(final, "%F", nfile_);
      dump_(final.c_str());
    }

  static std::string replace(const std::string &s, const char *pat, int val)
    {
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

  void (*dump_)(const char *);
  std::string atBeginJob_;
  std::string atEndJob_;
  std::string atBeginRun_;
  std::string atEndRun_;
  std::string atBeginLumi_;
  std::string atEndLumi_;
  std::string atInputFile_;
  std::string atEvent_;
  int prescale_;
  int nrecord_;
  int nevent_;
  int nrun_;
  int nlumi_;
  int nfile_;
};

DEFINE_FWK_MODULE(IgProfModule);
