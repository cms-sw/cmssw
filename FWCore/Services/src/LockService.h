#ifndef FWCore_Services_LockService_h
#define FWCore_Services_LockService_h

/*
  This class exists to only allow because ROOT is not thread-safe in its
  serialization/deserialization process at the moment.  It will hopefully
  be so very soon.  Here we have serialized access to some resource
  i.e. the one locking the mutex, where modules (using the module labels
  to identify them) are prevented from accessing that resource while running.

  There is only one mutex for the entire system.  We count one service
  object being constructed per thread that an event processor runs in.
  The current system only makes one event processor in one thread.
  If more than one event processor in more than one thread in the future,
  then I need to ensure that the service registry is not shared between
  any two threads running event processors.
 */

#include <string>
#include <vector>
#include "boost/thread/mutex.hpp"
namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;
  class ActivityRegistry;
  class ModuleDescription;
  class StreamContext;
  class ModuleCallingContext;

  namespace rootfix {
  
    class LockService
    {
    public:
      LockService(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~LockService();
      
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      boost::mutex& getLock() { return lock_; }
      
      void preSourceConstruction(const edm::ModuleDescription&);
      void postSourceConstruction(const edm::ModuleDescription&);
      
      void preSource();
      void postSource();
      
      
      void preModule(StreamContext const&, ModuleCallingContext const&);
      void postModule(StreamContext const&, ModuleCallingContext const&);
      
    private:
      boost::mutex& lock_;
      typedef std::vector<std::string> Labels;
      Labels labels_;
      bool lockSources_;
    };
  }
}

#endif
