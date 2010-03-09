#ifndef FWCore_Services_UnixSignalService_h
#define FWCore_Services_UnixSignalService_h

/*----------------------------------------------------------------------

UnixSignalService: At present, this defines a SIGUSR2 handler and
sets the shutdown flag when that signal has been raised.

This service is instantiated at job startup.

----------------------------------------------------------------------*/

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class Event;
  class EventSetup;
  class ConfigurationDescriptions;

  namespace service {

  class UnixSignalService
  {
  private:
    bool enableSigInt_;

  public:
    UnixSignalService(edm::ParameterSet const& ps, edm::ActivityRegistry& ac); 
    ~UnixSignalService();

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  }; // class UnixSignalService
  }  // end of namespace service
}    // end of namespace edm
#endif
