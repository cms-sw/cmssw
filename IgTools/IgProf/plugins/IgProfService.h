#ifndef IgTools_IgProf_IgProfService_h
#define IgTools_IgProf_IgProfService_h

//
//  Description: FWK service to implement hook for igprof memory profile
//               dump functionality
//
//  Peter Elmer, Princeton University                        18 Nov, 2008
//

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  class GlobalContext;
  class StreamContext;

  namespace service {
    class IgProfService {
    public:
      IgProfService(const ParameterSet &, ActivityRegistry &);

      void postBeginJob();

      void postBeginRun(GlobalContext const &gc);

      void postBeginLumi(GlobalContext const &gc);

      void preEvent(StreamContext const &sc);
      void postEvent(StreamContext const &sc);

      void preModuleEvent(StreamContext const &sc, ModuleCallingContext const &mcc);
      void postModuleEvent(StreamContext const &sc, ModuleCallingContext const &mcc);

      void postEndLumi(GlobalContext const &gc);

      void postEndRun(GlobalContext const &gc);

      void postEndJob();

      void postOpenFile(std::string const &);

      void postCloseFile(std::string const &);

      inline bool isProcessWideService(IgProfService const *) { return true; }

    private:
      void makeDump(const std::string &format, std::string_view moduleLabel = "");
      static std::string replace(const std::string &s, const char *pat, int val);
      static std::string replaceU64(const std::string &s, const char *pat, unsigned long long val);
      static std::string replace(const std::string &s, const char *pat, std::string_view val);

      void (*dump_)(const char *);

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
      std::string atPostEndRun_;
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
  }  // namespace service
}  // namespace edm

#endif
