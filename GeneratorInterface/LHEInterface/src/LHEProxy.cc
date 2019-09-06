#include <map>

#include <boost/thread.hpp>

#include "GeneratorInterface/LHEInterface/interface/LHEProxy.h"

using namespace lhef;

static boost::mutex mutex;

typedef std::map<LHEProxy::ProxyID, std::weak_ptr<LHEProxy> > ProxyMap;

static ProxyMap *getProxyMapInstance() {
  static struct Sentinel {
    Sentinel() : instance(new ProxyMap) {}
    ~Sentinel() {
      delete instance;
      instance = nullptr;
    }

    ProxyMap *instance;
  } sentinel;

  return sentinel.instance;
}

LHEProxy::LHEProxy(ProxyID id) : id(id) {}

LHEProxy::~LHEProxy() {
  boost::mutex::scoped_lock scoped_lock(mutex);

  ProxyMap *map = getProxyMapInstance();
  if (map)
    map->erase(id);
}

std::shared_ptr<LHEProxy> LHEProxy::create() {
  static LHEProxy::ProxyID nextProxyID = 0;

  boost::mutex::scoped_lock scoped_lock(mutex);

  std::shared_ptr<LHEProxy> proxy(new LHEProxy(++nextProxyID));

  ProxyMap *map = getProxyMapInstance();
  if (map)
    map->insert(ProxyMap::value_type(proxy->getID(), proxy));

  return proxy;
}

std::shared_ptr<LHEProxy> LHEProxy::find(ProxyID id) {
  boost::mutex::scoped_lock scoped_lock(mutex);

  ProxyMap *map = getProxyMapInstance();
  if (!map)
    return std::shared_ptr<LHEProxy>();

  ProxyMap::const_iterator pos = map->find(id);
  if (pos == map->end())
    return std::shared_ptr<LHEProxy>();

  return std::shared_ptr<LHEProxy>(pos->second);
}
