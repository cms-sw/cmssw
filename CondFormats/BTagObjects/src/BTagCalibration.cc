#include "CondFormats/BTagObjects/interface/BTagCalibration.h"

void BTagCalibration::addEntry(BTagEntry entry)
{
  data_[entry.params.token()].push_back(entry)
}

const std::vector<BTagEntry>& BTagCalibration::getEntries(
  BTagEntry::Parameters par) const
{
  return data_.at(par.token());  // throws exception if key unavailable
}
