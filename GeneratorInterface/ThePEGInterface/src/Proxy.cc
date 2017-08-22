#include <map>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

#include "GeneratorInterface/ThePEGInterface/interface/Proxy.h"

using namespace ThePEG;

static boost::mutex mutex;

typedef std::map<ProxyBase::ProxyID, boost::weak_ptr<ProxyBase> > ProxyMap;

static ProxyMap *getProxyMapInstance()
{
	static struct Sentinel {
		Sentinel() : instance(new ProxyMap) {}
		~Sentinel() { delete instance; instance = nullptr; }

		ProxyMap	*instance;
	} sentinel;

	return sentinel.instance;
}

ProxyBase::ProxyBase(ProxyID id) :
	id(id)
{
}

ProxyBase::~ProxyBase()
{
	boost::mutex::scoped_lock scoped_lock(mutex);

	ProxyMap *map = getProxyMapInstance();
	if (map)
		map->erase(id);
}

boost::shared_ptr<ProxyBase> ProxyBase::create(ctor_t ctor)
{
	static ProxyBase::ProxyID nextProxyID = 0;

	boost::mutex::scoped_lock scoped_lock(mutex);

	boost::shared_ptr<ProxyBase> proxy(ctor(++nextProxyID));

	ProxyMap *map = getProxyMapInstance();
	if (map)
		map->insert(ProxyMap::value_type(proxy->getID(), proxy));

	return proxy;
}

boost::shared_ptr<ProxyBase> ProxyBase::find(ProxyID id)
{
	boost::mutex::scoped_lock scoped_lock(mutex);

	ProxyMap *map = getProxyMapInstance();
	if (!map)
		return boost::shared_ptr<ProxyBase>();

	ProxyMap::const_iterator pos = map->find(id);
	if (pos == map->end())
		return boost::shared_ptr<ProxyBase>();

	return boost::shared_ptr<ProxyBase>(pos->second);
}
