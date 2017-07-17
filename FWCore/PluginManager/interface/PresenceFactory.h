#ifndef FWCore_PluginManager_PresenceFactory_h
#define FWCore_PluginManager_PresenceFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Utilities/interface/Presence.h"

#include <memory>
#include <string>

namespace edm {
typedef edmplugin::PluginFactory<Presence*()> PresencePluginFactory;

typedef Presence*(PresenceFunc)();

class PresenceFactory {
 public:
  ~PresenceFactory();

  static PresenceFactory* get();

  std::unique_ptr<Presence> makePresence(
      std::string const& presence_type) const;

 private:
  PresenceFactory();
  // static PresenceFactory singleInstance_;
};
}
#endif  // FWCore_PluginManager_PresenceFactory_h
