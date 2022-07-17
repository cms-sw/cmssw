#include "CondFormats/L1TObjects/interface/LUT.h"

#include <sstream>
#include <string>
#include <algorithm>

//reads in the file
//the format is "address payload"
//all commments are ignored (start with '#') except for the header comment (starts with #<header>)
//currently ignores anything else on the line after the "address payload" and assumes they come first
int l1t::LUT::read(std::istream& stream) {
  data_.clear();

  int readHeaderCode = readHeader_(stream);
  if (readHeaderCode != SUCCESS)
    return readHeaderCode;

  std::vector<std::pair<unsigned int, int> > entries;
  unsigned int maxAddress = addressMask_;
  std::string line;

  while (std::getline(stream, line)) {
    line.erase(std::find(line.begin(), line.end(), '#'), line.end());  //ignore comments
    std::istringstream lineStream(line);
    std::pair<unsigned int, int> entry;
    while (lineStream >> entry.first >> entry.second) {
      entry.first &= addressMask_;
      entry.second &= dataMask_;
      entries.push_back(entry);
      if (entry.first > maxAddress || maxAddress == addressMask_)
        maxAddress = entry.first;
    }
  }
  std::sort(entries.begin(), entries.end());
  if (entries.empty()) {
    //log the error we read nothing
    return NO_ENTRIES;
  }
  //this check is redundant as dups are also picked up by the next check but might make for easier debugging
  if (std::adjacent_find(entries.begin(), entries.end(), [](auto const& a, auto const& b) {
        return a.first == b.first;
      }) != entries.end()) {
    //log the error that we have duplicate addresses once masked
    return DUP_ENTRIES;
  }
  if (entries.front().first != 0 ||
      std::adjacent_find(entries.begin(), entries.end(), [](auto const& a, auto const& b) {
        return a.first + 1 != b.first;
      }) != entries.end()) {
    //log the error that we have a missing entry
    return MISS_ENTRIES;
  }

  if (maxAddress != std::numeric_limits<unsigned int>::max())
    data_.resize(maxAddress + 1, 0);
  else {
    //log the error that we have more addresses than we can deal with (which is 4gb so something probably has gone wrong anyways)
    return MAX_ADDRESS_OUTOFRANGE;
  }

  std::transform(entries.begin(), entries.end(), data_.begin(), [](auto const& x) { return x.second; });
  return SUCCESS;
}

void l1t::LUT::write(std::ostream& stream) const {
  stream << "#<header> V1 " << nrBitsAddress_ << " " << nrBitsData_ << " </header> " << std::endl;
  for (unsigned int address = 0; address < data_.size(); address++) {
    stream << (address & addressMask_) << " " << data(address) << std::endl;
  }
}

int l1t::LUT::readHeader_(std::istream& stream) {
  int startPos = stream.tellg();  //we are going to reset to this position before we exit
  std::string line;
  while (std::getline(stream, line)) {
    if (line.find("#<header>") == 0) {  //line
      std::istringstream lineStream(line);

      std::string version;      //currently not doing anything with this
      std::string headerField;  //currently not doing anything with this
      if (lineStream >> headerField >> version >> nrBitsAddress_ >> nrBitsData_) {
        addressMask_ = nrBitsAddress_ != 32 ? (0x1U << nrBitsAddress_) - 1 : ~0x0;
        dataMask_ = nrBitsData_ != 32 ? (0x1U << nrBitsData_) - 1 : ~0x0;
        stream.seekg(startPos);
        return SUCCESS;
      }
    }
  }

  nrBitsAddress_ = 0;
  nrBitsData_ = 0;
  addressMask_ = (0x1 << nrBitsAddress_) - 1;
  dataMask_ = (0x1 << nrBitsData_) - 1;

  stream.seekg(startPos);
  return NO_HEADER;
}
