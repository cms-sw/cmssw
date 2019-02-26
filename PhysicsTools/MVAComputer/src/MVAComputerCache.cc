#include <memory>

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

namespace PhysicsTools {

MVAComputerCache::MVAComputerCache() :
	containerCacheId(Calibration::MVAComputerContainer::CacheId()),
	computerCacheId(Calibration::MVAComputer::CacheId())
{
}

MVAComputerCache::~MVAComputerCache()
{
}

bool MVAComputerCache::update(const Calibration::MVAComputer *computer)
{
	if (!computer && !*this)
		return false;
	if (computer && !computer->changed(computerCacheId))
		return false;

	if (computer) {
		this->computer.reset(new MVAComputer(computer));
		computerCacheId = computer->getCacheId();
	} else {
		this->computer.reset();
		computerCacheId = Calibration::MVAComputer::CacheId();
	}

	containerCacheId = Calibration::MVAComputerContainer::CacheId();
	return true;
}

bool MVAComputerCache::update(
			const Calibration::MVAComputerContainer *container,
			const char *calib)
{
	if (!container && !*this)
		return false;
	if (container && !container->changed(containerCacheId))
		return false;

	if (container) {
		const Calibration::MVAComputer *computer =
						&container->find(calib);
		bool result = update(computer);
		containerCacheId = container->getCacheId();
		return result;
	}

	this->computer.reset();

	computerCacheId = Calibration::MVAComputer::CacheId();
	containerCacheId = Calibration::MVAComputerContainer::CacheId();
	return true;
}

std::unique_ptr<MVAComputer> MVAComputerCache::release()
{
	computerCacheId = Calibration::MVAComputer::CacheId();
	containerCacheId = Calibration::MVAComputerContainer::CacheId();
	return std::move(computer);
}

} // namespace PhysicsTools
