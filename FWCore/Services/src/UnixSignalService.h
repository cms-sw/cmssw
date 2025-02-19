#ifndef FWCore_Services_UnixSignalService_h
#define FWCore_Services_UnixSignalService_h

/*----------------------------------------------------------------------

UnixSignalService: At present, this defines a SIGUSR2 handler and
sets the shutdown flag when that signal has been raised.

This service is instantiated at job startup.

----------------------------------------------------------------------*/

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConfigurationDescriptions;

  namespace service {
    class UnixSignalService {
    public:
      explicit UnixSignalService(ParameterSet const& ps);
      ~UnixSignalService();

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

    private:
      bool enableSigInt_;
    }; // class UnixSignalService
  }  // end of namespace service
}    // end of namespace edm
#endif
