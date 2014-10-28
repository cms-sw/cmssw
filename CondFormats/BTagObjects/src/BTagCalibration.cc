#include <iostream>

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

void BTagCalibration::readCSV(istream &s)
{
  std::string line;
  while (getline(s,line)) {
    addEntry(BTagEntry(line));
  }
}

void BTagCalibration::makeCSV(ostream &s) const
{
  s << BTagEntry::makeCSVHeader();
  for (auto i = data_.cbegin(); i != data_.cend(); ++i) {
    auto vec = i->second;
    for (auto j = vec.cbegin(); j != vec.cend(); ++j) {
      s << j->makeCSVLine();
    }
  }
}


std::string BTagCalibration::token(const BTagEntry::Parameters &par) const
{
  std::stringstream buff;
  buff << par.operatingPoint << ", "
       << par.measurementType << ", "
       << par.sysType;
  return buff.str();
}