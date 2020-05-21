#include <string>
#include <vector>
#include <memory>

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace PhysicsTools::Calibration;

inline GenericMVAComputerCache::IndividualComputer::IndividualComputer() {}

inline GenericMVAComputerCache::IndividualComputer::IndividualComputer(const IndividualComputer &orig)
    : label(orig.label) {}

inline GenericMVAComputerCache::IndividualComputer::~IndividualComputer() {}

GenericMVAComputerCache::GenericMVAComputerCache(const std::vector<std::string> &labels)
    : computers(labels.size()),
      cacheId(MVAComputerContainer::CacheId()),
      initialized(false),
      empty(true),
      errorUpdatingLabel() {
  std::vector<IndividualComputer>::iterator computer = computers.begin();
  for (const auto &label : labels) {
    computer->label = label;
    computer->cacheId = MVAComputer::CacheId();
    computer++;
  }
}

GenericMVAComputerCache::~GenericMVAComputerCache() {}

GenericMVAComputer const *GenericMVAComputerCache::getComputer(int index) const {
  if (!errorUpdatingLabel.empty()) {
    throw cms::Exception("MVAComputerCalibration")
        << "GenericMVAComputerCache::getComputer Error occurred during update.\n"
        << "Calibration record " << errorUpdatingLabel << " not found in MVAComputerContainer." << std::endl;
  }
  return index >= 0 ? computers[index].computer.get() : nullptr;
}

bool GenericMVAComputerCache::isEmpty() const {
  if (!errorUpdatingLabel.empty()) {
    throw cms::Exception("MVAComputerCalibration")
        << "GenericMVAComputerCache::isEmpty Error occurred during update.\n"
        << "Calibration record " << errorUpdatingLabel << " not found in MVAComputerContainer." << std::endl;
  }
  return empty;
}

bool GenericMVAComputerCache::update(const MVAComputerContainer *calib) {
  // check container for changes
  if (initialized && !calib->changed(cacheId))
    return false;

  empty = true;

  bool changed = false;
  for (auto &computer : computers) {
    // empty labels means we don't want a computer
    if (computer.label.empty())
      continue;

    // Delay throwing if the label cannot be found until getComputer is called
    // Sometimes this cache is updated and never used.
    if (!calib->contains(computer.label)) {
      errorUpdatingLabel = computer.label;
      continue;
    }

    const MVAComputer *computerCalib = &calib->find(computer.label);
    if (!computerCalib) {
      computer.computer.reset();
      continue;
    }

    // check container content for changes
    if (computer.computer.get() && !computerCalib->changed(computer.cacheId)) {
      empty = false;
      continue;
    }

    // drop old computer
    computer.computer.reset();

    if (!computerCalib)
      continue;

    // instantiate new MVAComputer with uptodate calibration
    computer.computer = std::unique_ptr<GenericMVAComputer>(new GenericMVAComputer(computerCalib));

    computer.cacheId = computerCalib->getCacheId();

    changed = true;
    empty = false;
  }

  cacheId = calib->getCacheId();
  initialized = true;

  return changed;
}
