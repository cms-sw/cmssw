#include "L1Trigger/CSCTriggerPrimitives/interface/CSCLUTReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

CSCLUTReader::CSCLUTReader(const std::string& fname)
    : nrBitsAddress_(0), nrBitsData_(0), addressMask_(0), dataMask_(0), data_(), m_codeInWidth(12), m_outWidth(32) {
  if (fname != std::string("")) {
    load(fname);
  } else {
    initialize();
  }
}

// I/O functions
void CSCLUTReader::save(std::ofstream& output) { write(output); }

float CSCLUTReader::data(unsigned int address) const {
  return (address & addressMask_) < data_.size() ? data_[address] : 0;
}

int CSCLUTReader::load(const std::string& inFileName) {
  std::ifstream fstream;
  fstream.open(edm::FileInPath(inFileName.c_str()).fullPath());
  if (!fstream.good()) {
    fstream.close();
    throw cms::Exception("FileOpenError") << "Failed to open LUT file: " << inFileName;
  }
  int readCode = read(fstream);

  m_initialized = true;
  fstream.close();

  return readCode;
}

float CSCLUTReader::lookup(int code) const {
  if (m_initialized) {
    return lookupPacked(code);
  }
  return 0;
}

float CSCLUTReader::lookupPacked(const int input) const {
  if (m_initialized) {
    return data((unsigned int)input);
  }
  throw cms::Exception("Uninitialized") << "If you're not loading a LUT from file you need to implement lookupPacked.";
  return 0;
}

void CSCLUTReader::initialize() {
  if (empty()) {
    std::stringstream stream;
    stream << "#<header> V1 " << m_codeInWidth << " " << m_outWidth << " </header> " << std::endl;
    for (int in = 0; in < (1 << m_codeInWidth); ++in) {
      int out = lookup(in);
      stream << in << " " << out << std::endl;
    }
    read(stream);
  }
  m_initialized = true;
}

unsigned CSCLUTReader::checkedInput(unsigned in, unsigned maxWidth) const {
  unsigned maxIn = (1 << maxWidth) - 1;
  return (in < maxIn ? in : maxIn);
}

int CSCLUTReader::read(std::istream& stream) {
  data_.clear();

  int readHeaderCode = readHeader(stream);
  if (readHeaderCode != SUCCESS)
    return readHeaderCode;

  std::vector<std::pair<unsigned int, float> > entries;
  unsigned int maxAddress = addressMask_;
  std::string line;

  while (std::getline(stream, line)) {
    line.erase(std::find(line.begin(), line.end(), '#'), line.end());  //ignore comments
    std::istringstream lineStream(line);
    std::pair<unsigned int, float> entry;
    while (lineStream >> entry.first >> entry.second) {
      entry.first &= addressMask_;
      // entry.second &= dataMask_;
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

void CSCLUTReader::write(std::ostream& stream) const {
  stream << "#<header> V1 " << nrBitsAddress_ << " " << nrBitsData_ << " </header> " << std::endl;
  for (unsigned int address = 0; address < data_.size(); address++) {
    stream << (address & addressMask_) << " " << data(address) << std::endl;
  }
}

unsigned int CSCLUTReader::maxSize() const {
  return addressMask_ == std::numeric_limits<unsigned int>::max() ? addressMask_ : addressMask_ + 1;
}

int CSCLUTReader::readHeader(std::istream& stream) {
  int startPos = stream.tellg();  //we are going to reset to this position before we exit
  std::string line;
  while (std::getline(stream, line)) {
    if (line.find("#<header>") == 0) {  //line
      std::istringstream lineStream(line);

      std::string version;      //currently not doing anything with this
      std::string headerField;  //currently not doing anything with this
      if (lineStream >> headerField >> version >> nrBitsAddress_ >> nrBitsData_) {
        addressMask_ = nrBitsAddress_ != 32 ? (0x1 << nrBitsAddress_) - 1 : ~0x0;
        dataMask_ = (0x1 << nrBitsData_) - 1;
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
