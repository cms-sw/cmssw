#include <string>
#include <vector>
#include <memory>

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace PhysicsTools::Calibration;

inline GenericMVAComputerCache::IndividualComputer::IndividualComputer()
{
}

inline GenericMVAComputerCache::IndividualComputer::IndividualComputer(
					const IndividualComputer &orig) :
	label(orig.label)
{
}

inline GenericMVAComputerCache::IndividualComputer::~IndividualComputer()
{
}

GenericMVAComputerCache::GenericMVAComputerCache(
				const std::vector<std::string> &labels) :
	computers(labels.size()),
	cacheId(MVAComputerContainer::CacheId()),
	initialized(false),
	empty(true),
        errorUpdatingLabel()
{
	std::vector<IndividualComputer>::iterator computer = computers.begin();
	for(std::vector<std::string>::const_iterator iter = labels.begin();
	    iter != labels.end(); iter++) {
		computer->label = *iter;
		computer->cacheId = MVAComputer::CacheId();
		computer++;
	}
}

GenericMVAComputerCache::~GenericMVAComputerCache()
{
}

GenericMVAComputer const* GenericMVAComputerCache::getComputer(int index) const
{
  if(!errorUpdatingLabel.empty()) {
    throw cms::Exception("MVAComputerCalibration")
      << "GenericMVAComputerCache::getComputer Error occurred during update.\n"
      << "Calibration record " << errorUpdatingLabel
      << " not found in MVAComputerContainer." << std::endl;
  }
  return index >= 0 ? computers[index].computer.get() : nullptr;
}

bool GenericMVAComputerCache::isEmpty() const {
  if(!errorUpdatingLabel.empty()) {
    throw cms::Exception("MVAComputerCalibration")
      << "GenericMVAComputerCache::isEmpty Error occurred during update.\n"
      << "Calibration record " << errorUpdatingLabel
      << " not found in MVAComputerContainer." << std::endl;
  }
  return empty;
}

bool GenericMVAComputerCache::update(const MVAComputerContainer *calib)
{
	// check container for changes
	if (initialized && !calib->changed(cacheId))
		return false;

	empty = true;

	bool changed = false;
	for(std::vector<IndividualComputer>::iterator iter = computers.begin();
	    iter != computers.end(); iter++) {
		// empty labels means we don't want a computer
		if (iter->label.empty())
			continue;

                // Delay throwing if the label cannot be found until getComputer is called
                // Sometimes this cache is updated and never used.
		if (!calib->contains(iter->label)) {
                  errorUpdatingLabel = iter->label;
                  continue;
                }

		const MVAComputer *computerCalib = &calib->find(iter->label);
		if (!computerCalib) {
			iter->computer.reset();
			continue;
		}

		// check container content for changes
		if (iter->computer.get() &&
		    !computerCalib->changed(iter->cacheId)) {
			empty = false;
			continue;
		}

		// drop old computer
		iter->computer.reset();

		if (!computerCalib)
			continue;

		// instantiate new MVAComputer with uptodate calibration
		iter->computer = std::unique_ptr<GenericMVAComputer>(
					new GenericMVAComputer(computerCalib));

		iter->cacheId = computerCalib->getCacheId();

		changed = true;
		empty = false;
	}

	cacheId = calib->getCacheId();
	initialized = true;

	return changed;
}
