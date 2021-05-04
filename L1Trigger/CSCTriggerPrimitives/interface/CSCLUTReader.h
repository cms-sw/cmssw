#ifndef L1Trigger_CSCTriggerPrimitives_CSCLUTReader
#define L1Trigger_CSCTriggerPrimitives_CSCLUTReader

#include <fstream>
#include <sstream>
#include <bitset>
#include <iostream>
#include <vector>
#include <limits>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CSCLUTReader {
public:
  enum ReadCodes {
    SUCCESS = 0,
    NO_ENTRIES = 1,
    DUP_ENTRIES = 2,
    MISS_ENTRIES = 3,
    MAX_ADDRESS_OUTOFRANGE = 4,
    NO_HEADER = 5
  };

  /* CSCLUTReader(); */
  explicit CSCLUTReader(const std::string&);
  ~CSCLUTReader() {}

  float lookup(int code) const;
  float lookupPacked(const int input) const;

  // populates the map.
  void initialize();

  unsigned checkedInput(unsigned in, unsigned maxWidth) const;

  // I/O functions
  void save(std::ofstream& output);
  int load(const std::string& inFileName);

  float data(unsigned int address) const;
  int read(std::istream& stream);
  void write(std::ostream& stream) const;

  unsigned int nrBitsAddress() const { return nrBitsAddress_; }
  unsigned int nrBitsData() const { return nrBitsData_; }
  //following the convention of vector::size()
  unsigned int maxSize() const;
  bool empty() const { return data_.empty(); }

private:
  int readHeader(std::istream&);

  unsigned int nrBitsAddress_;  //technically redundant with addressMask
  unsigned int nrBitsData_;     //technically redundant with dataMask
  unsigned int addressMask_;
  unsigned int dataMask_;

  std::vector<float> data_;

  int m_codeInWidth;
  unsigned m_outWidth;
  bool m_initialized;
};

#endif
