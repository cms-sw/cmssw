#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include <algorithm>

EcalBaseNumber::EcalBaseNumber() : _theLevels(0) {}

EcalBaseNumber::EcalBaseNumber(const EcalBaseNumber& aBaseNumber)
    : _sortedName(aBaseNumber._sortedName),
      _sortedCopyNumber(aBaseNumber._sortedCopyNumber),
      _theLevels(aBaseNumber._theLevels) {}

void EcalBaseNumber::setSize(const int& size) {
  _sortedName.resize(size);
  _sortedCopyNumber.resize(size);
}

void EcalBaseNumber::addLevel(const std::string& name, const int& copyNumber) {
  _sortedName[_theLevels] = name;
  _sortedCopyNumber[_theLevels] = copyNumber;
  _theLevels++;
}

int EcalBaseNumber::getLevels() const { return _theLevels; }

int EcalBaseNumber::getCopyNumber(int level) const { return _sortedCopyNumber[level]; }

int EcalBaseNumber::getCopyNumber(const std::string& levelName) const {
  for (int iLevel = 0; iLevel < _theLevels; iLevel++) {
    if (_sortedName[iLevel].find(levelName) != std::string::npos) {
      return _sortedCopyNumber[iLevel];
    }
  }
  return 0;
}

std::string const& EcalBaseNumber::getLevelName(int level) const { return _sortedName[level]; }

int EcalBaseNumber::getCapacity() { return _sortedName.capacity(); }

void EcalBaseNumber::reset() { _theLevels = 0; }

void EcalBaseNumber::reverse() {
  std::reverse(std::begin(_sortedName), std::end(_sortedName));
  std::reverse(std::begin(_sortedCopyNumber), std::end(_sortedCopyNumber));
}
