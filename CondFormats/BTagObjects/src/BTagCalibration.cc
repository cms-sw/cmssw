#include "CondFormats/BTagObjects/interface/BTagCalibration.h"

void BTagCalibration::addEntry(BTagEntry entry)
{
  data_[token(entry.params)].push_back(entry);
}

const std::vector<BTagEntry>& BTagCalibration::getEntries(
  BTagEntry::Parameters par) const
{
  return data_.at(token(par));  // throws exception if key unavailable
}

std::string BTagCalibration::token(const BTagEntry::Parameters &par) const
{
  std::stringstream buff;
  buff << par.operatingPoint << ", "
       << par.measurementType << ", "
       << par.sysType;
  return buff.str();
}