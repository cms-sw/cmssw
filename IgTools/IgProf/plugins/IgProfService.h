#ifndef IgTools_IgProf_IgProfService_h
#define IgTools_IgProf_IgProfService_h

//
//  Description: FWK service to implement hook for igprof memory profile 
//               dump functionality
//
//  Peter Elmer, Princeton University                        18 Nov, 2008
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  namespace service {
    class IgProfService {

    public:

      IgProfService(const ParameterSet&,ActivityRegistry&);

      void postBeginJob();

      void postBeginRun(Run const& run, EventSetup const& es);

      void postBeginLumi(LuminosityBlock const& run, EventSetup const& es);

      void preEvent(EventID const& id, Timestamp const& ts);
      void postEvent(Event const& ev, EventSetup const& es);

      void postEndLumi(LuminosityBlock const& run, EventSetup const& es);

      void postEndRun(Run const& run, EventSetup const& es);

      void postEndJob();

      void postOpenFile();

      void postCloseFile();

      inline
      bool isProcessWideService(IgProfService const*) {
        return true;
      }

    private:

      void makeDump(const std::string &format);
      static std::string replace(const std::string &s, 
                                 const char *pat, int val);

      void (*dump_)(const char *);

      std::string atPostBeginJob_;
      std::string atPostBeginRun_;
      std::string atPostBeginLumi_;

      std::string atPreEvent_;
      std::string atPostEvent_;

      std::string atPostEndLumi_;
      std::string atPostEndRun_;
      std::string atPostEndJob_;

      std::string atPostOpenFile_;
      std::string atPostCloseFile_;

      int mineventrecord_;
      int prescale_;
      int nrecord_;      // counter
      int nevent_;
      int nrun_;
      int nlumi_;
      int nfileopened_;  // counter of files opened thus far
      int nfileclosed_;  // counter of files closed thus far

    };
  }
}


#endif
