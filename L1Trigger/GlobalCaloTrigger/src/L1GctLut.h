#ifndef L1GCTLUT_H_
#define L1GCTLUT_H_

#include <iomanip>
#include <sstream>
#include <cstdint>

/*!
 * \author Greg Heath
 * \date Feb 2007
 */

/*! \class L1GctLut
 * \brief Base class for LookUp Tables
 * 
*/

template <int NAddressBits, int NDataBits>
class L1GctLut {
public:
  static const uint16_t MAX_ADDRESS_BITMASK;
  static const uint16_t MAX_DATA_BITMASK;

  virtual ~L1GctLut();

  /// Overload << operator
  friend std::ostream& operator<<(std::ostream& os, const L1GctLut<NAddressBits, NDataBits>& lut) {
    //----------------------------------------------------------------------------------------
    // Define the code here for the friend template function to get around
    // compiler/linker problems when instantiating the template class.
    // See http://www.parashift.com/c++-faq-lite/templates.html#faq-35.16
    static const int maxAddress = L1GctLut<NAddressBits, NDataBits>::MAX_ADDRESS_BITMASK;
    static const int width = L1GctLut<NAddressBits, NDataBits>::printWidth;

    os << lut.printHeader();

    for (int a = 0; a <= maxAddress; a += width) {
      os << lut.printLine(a);
    }
    return os;
    // End of friend function definition
    //----------------------------------------------------------------------------------------
  }

  /// Access the look-up table contents for a given Address
  uint16_t lutValue(const uint16_t lutAddress) const;

  /// Access the look-up table contents for a given Address
  uint16_t operator[](const uint16_t lutAddress) const { return lutValue(lutAddress); }

  /// Equality check between look-up tables
  template <int KAddressBits, int KDataBits>
  int operator==(const L1GctLut<KAddressBits, KDataBits>& rhsLut) const {
    return equalityCheck(rhsLut);
  }

  /// Inequality check between look-up tables
  template <int KAddressBits, int KDataBits>
  int operator!=(const L1GctLut<KAddressBits, KDataBits>& rhsLut) const {
    return !equalityCheck(rhsLut);
  }

  bool setupOk() { return m_setupOk; }

  /// control output messages
  void setVerbose() { m_verbose = true; }
  void setTerse() { m_verbose = false; }

protected:
  L1GctLut();

  virtual uint16_t value(const uint16_t lutAddress) const = 0;

  template <int KAddressBits, int KDataBits>
  bool equalityCheck(const L1GctLut<KAddressBits, KDataBits>& c) const;

  bool m_setupOk;
  bool m_verbose;

private:
  // For use by the friend function to print the lut contents
  static const int printWidth;
  std::string printHeader() const;
  std::string printLine(const int add) const;
};

template <int NAddressBits, int NDataBits>
const uint16_t L1GctLut<NAddressBits, NDataBits>::MAX_ADDRESS_BITMASK = (1 << NAddressBits) - 1;
template <int NAddressBits, int NDataBits>
const uint16_t L1GctLut<NAddressBits, NDataBits>::MAX_DATA_BITMASK = (1 << NDataBits) - 1;

template <int NAddressBits, int NDataBits>
const int L1GctLut<NAddressBits, NDataBits>::printWidth = 16;

template <int NAddressBits, int NDataBits>
L1GctLut<NAddressBits, NDataBits>::L1GctLut() : m_setupOk(false) {}

template <int NAddressBits, int NDataBits>
L1GctLut<NAddressBits, NDataBits>::~L1GctLut() {}

template <int NAddressBits, int NDataBits>
uint16_t L1GctLut<NAddressBits, NDataBits>::lutValue(const uint16_t lutAddress) const {
  if (!m_setupOk)
    return (uint16_t)0;
  uint16_t address = (lutAddress & MAX_ADDRESS_BITMASK);
  uint16_t data = (value(address) & MAX_DATA_BITMASK);
  return data;
}

template <int NAddressBits, int NDataBits>
template <int KAddressBits, int KDataBits>
bool L1GctLut<NAddressBits, NDataBits>::equalityCheck(const L1GctLut<KAddressBits, KDataBits>& rhsLut) const {
  if (KAddressBits == NAddressBits && KDataBits == NDataBits) {
    bool match = true;
    for (uint16_t address = 0; address <= MAX_ADDRESS_BITMASK; address++) {
      if (this->lutValue(address) != rhsLut.lutValue(address)) {
        match = false;
        break;
      }
    }
    return match;
  } else {
    return false;
  }
}

template <int NAddressBits, int NDataBits>
std::string L1GctLut<NAddressBits, NDataBits>::printHeader() const {
  std::stringstream ss;
  ss << std::hex << std::showbase;
  ss << std::setw(8) << "|";
  for (int a = 0; ((a < printWidth) && (a <= MAX_ADDRESS_BITMASK)); ++a) {
    ss << std::setw(7) << a;
  }
  ss << std::endl;
  ss << std::setfill('-') << std::setw(8) << "+";
  for (int a = 0; ((a < printWidth) && (a <= MAX_ADDRESS_BITMASK)); ++a) {
    ss << std::setw(7) << "-";
  }
  ss << std::endl;

  return ss.str();
}

template <int NAddressBits, int NDataBits>
std::string L1GctLut<NAddressBits, NDataBits>::printLine(const int add) const {
  std::stringstream ss;
  ss << std::hex << std::showbase;
  int a = add;
  ss << std::setw(7) << a << "|";
  for (int c = 0; ((c < printWidth) && (a <= MAX_ADDRESS_BITMASK)); ++c) {
    uint16_t address = static_cast<uint16_t>(a++);
    ss << std::setw(7) << lutValue(address);
  }
  ss << std::endl;

  return ss.str();
}

#endif /*L1GCTLUT_H_*/
