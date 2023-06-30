#ifndef L1GCTLUTFROMFILE_H_
#define L1GCTLUTFROMFILE_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctLut.h"

#include <vector>
#include <fstream>
#include <cassert>

template <int NAddressBits, int NDataBits>
class L1GctLutFromFile : public L1GctLut<NAddressBits, NDataBits> {
public:
  static L1GctLutFromFile<NAddressBits, NDataBits>* setupLut(const std::string filename);
  virtual ~L1GctLutFromFile();

  void readFromFile(const std::string filename);

protected:
  L1GctLutFromFile();

  virtual uint16_t value(const uint16_t lutAddress) const;

private:
  std::vector<uint16_t> m_lutContents;
};

template <int NAddressBits, int NDataBits>
L1GctLutFromFile<NAddressBits, NDataBits>* L1GctLutFromFile<NAddressBits, NDataBits>::setupLut(
    const std::string filename) {
  L1GctLutFromFile<NAddressBits, NDataBits>* newLut = new L1GctLutFromFile<NAddressBits, NDataBits>();
  newLut->readFromFile(filename);
  return newLut;
}

template <int NAddressBits, int NDataBits>
L1GctLutFromFile<NAddressBits, NDataBits>::L1GctLutFromFile()
    : L1GctLut<NAddressBits, NDataBits>(), m_lutContents(1 << NAddressBits) {}

template <int NAddressBits, int NDataBits>
L1GctLutFromFile<NAddressBits, NDataBits>::~L1GctLutFromFile() {}

template <int NAddressBits, int NDataBits>
void L1GctLutFromFile<NAddressBits, NDataBits>::readFromFile(const std::string filename) {
  static const unsigned maxAddress = L1GctLut<NAddressBits, NDataBits>::MAX_ADDRESS_BITMASK;
  static const unsigned rowLength = 16;

  std::ifstream inFile;
  std::string strFromFile;
  inFile.open(filename.c_str(), std::ios::in);

  // Read input values in hex
  inFile >> std::hex;

  // Read and discard the first lines of the file, looking for
  // a line entirely composed of '-' and '+' characters
  while (std::getline(inFile, strFromFile).good()) {
    if (strFromFile.length() > 0 && strFromFile.find_first_not_of("-+") == std::string::npos)
      break;
  }

  // Now read the lut data
  unsigned a = 0;
  while (a <= maxAddress) {
    unsigned val;
    inFile >> val;
    assert(val == a);
    inFile.ignore();
    for (unsigned c = 0; c < rowLength && a <= maxAddress; c++) {
      inFile >> val;
      m_lutContents.at(a++) = static_cast<uint16_t>(val);
    }
  }
  // All values read
  if (inFile.get() == '\n') {
    inFile.get();
  }
  L1GctLut<NAddressBits, NDataBits>::m_setupOk = (inFile.eof() && (m_lutContents.size() == a));
  inFile.close();
}

template <int NAddressBits, int NDataBits>
uint16_t L1GctLutFromFile<NAddressBits, NDataBits>::value(const uint16_t lutAddress) const {
  unsigned Address = static_cast<unsigned>(lutAddress & L1GctLut<NAddressBits, NDataBits>::MAX_ADDRESS_BITMASK);
  assert(Address < m_lutContents.size());
  return m_lutContents.at(Address);
}

#endif /*L1GCTLUTFROMFILE_H_*/
