#include <iostream>

#include "CondFormats/BTagObjects/interface/BTagCalibration.h"

void BTagCalibration::addEntry(BTagEntry entry)
{
  data_[token(entry.params)].push_back(entry);
}

const std::vector<BTagEntry>& BTagCalibration::getEntries(
  BTagEntry::Parameters par) const
{
  auto tok = token(par);
  if (!data_.count(tok)) {
    std::cerr << "(OperatingPoint, measurementType, sysType) not available: "
              << tok << std::endl;
  }
  return data_.at(tok);
}

void BTagCalibration::readCSV(const std::string &s)
{
  std::stringstream buff(s);
  readCSV(buff);
}

void BTagCalibration::readCSV(istream &s)
{
  std::string line;

  // firstline might be the header
  getline(s,line);
  if (line.find("OperatingPoint") == std::string::npos) {
    addEntry(BTagEntry(line));
  }

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

std::string BTagCalibration::makeCSV() const
{
  std::stringstream buff;
  makeCSV(buff);
  return buff.str();
}

std::string BTagCalibration::token(const BTagEntry::Parameters &par)
{
  std::stringstream buff;
  buff << par.operatingPoint << ", "
       << par.measurementType << ", "
       << par.sysType;
  return buff.str();
}