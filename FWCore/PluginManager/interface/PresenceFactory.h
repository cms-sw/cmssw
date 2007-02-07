#ifndef FWCore_PluginManager_PresenceFactory_h
#define FWCore_PluginManager_PresenceFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Utilities/interface/Presence.h"

#include <string>
#include <memory>

namespace edm {

  typedef Presence* (PresenceFunc)();

  class PresenceFactory :
    public seal::PluginFactory<PresenceFunc> {
  public:
    ~PresenceFactory();

    static PresenceFactory* get();

    std::auto_ptr<Presence>
      makePresence(std::string const & presence_type) const;

  private:
    PresenceFactory();
    static PresenceFactory singleInstance_;
  };
}
#endif // FWCore_PluginManager_PresenceFactory_h
