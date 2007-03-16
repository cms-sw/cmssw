#ifndef L1GCTLUT_H_
#define L1GCTLUT_H_

//#include "FWCore/Framework/interface/ESHandle.h"

#include <boost/cstdint.hpp> //for uint16_t

#include <map>
#include <iostream>
#include <iomanip>

/*!
 * \author Greg Heath
 * \date Feb 2007
 */

/*! \class L1GctLut
 * \brief Base class for LookUp Tables
 * 
*/

template <int NAddressBits, int NDataBits>
class L1GctLut;
template <int NAddressBits, int NDataBits>
std::ostream& operator << (std::ostream& os, L1GctLut<NAddressBits, NDataBits>& lut);

template <int NAddressBits, int NDataBits>
class L1GctLut
{
public:
  static const uint16_t MAX_ADDRESS_BITMASK;
  static const uint16_t MAX_DATA_BITMASK;
  
  virtual ~L1GctLut();

  /// Overload << operator
  friend std::ostream& operator << <> (std::ostream& os, const L1GctLut<NAddressBits, NDataBits>& lut);

  uint16_t lutValue (const uint16_t lutAddress) const;

  uint16_t operator[] (const uint16_t lutAddress) const { return lutValue(lutAddress); } 

protected:
  
  L1GctLut();

  virtual uint16_t value (const uint16_t lutAddress) const=0;
  bool m_setupOk;

};

template <int NAddressBits, int NDataBits>
const uint16_t L1GctLut<NAddressBits, NDataBits>::MAX_ADDRESS_BITMASK = (1 << NAddressBits) - 1;
template <int NAddressBits, int NDataBits>
const uint16_t L1GctLut<NAddressBits, NDataBits>::MAX_DATA_BITMASK = (1 << NDataBits) - 1;

template <int NAddressBits, int NDataBits>
L1GctLut<NAddressBits, NDataBits>::L1GctLut() : m_setupOk(false) {}

template <int NAddressBits, int NDataBits>
L1GctLut<NAddressBits, NDataBits>::~L1GctLut() {}

template <int NAddressBits, int NDataBits>
uint16_t L1GctLut<NAddressBits, NDataBits>::lutValue(const uint16_t lutAddress) const
{
  assert (m_setupOk);
  uint16_t address=(lutAddress & MAX_ADDRESS_BITMASK);
  uint16_t data=(value(address) & MAX_DATA_BITMASK);
  return data;
}

template <int NAddressBits, int NDataBits>
std::ostream& operator << (std::ostream& os, const L1GctLut<NAddressBits, NDataBits>& lut)
{
  static const int maxAddress=L1GctLut<NAddressBits, NDataBits>::MAX_ADDRESS_BITMASK;
  static const int width=16;
  bool allZeros=true;
  os << "      |";
  for (int a=0; ((a<width) && (a<=maxAddress)); ++a) {
    os << std::setw(6) << std::hex << a;
  }
  os << std::endl;
  os << "------+";
  for (int a=0; ((a<width) && (a<=maxAddress)); ++a) {
    os << "------";
  }
  os << std::endl;
  for (int a=0; a<=maxAddress; ) {
    bool rowOfZeros=true;
    for (int c=0; ((c<width) && ((a+c)<=maxAddress)); ++c) {
      uint16_t address = static_cast<uint16_t>(a+c);
      rowOfZeros &= (lut.lutValue(address)==0);
    }
    if (!rowOfZeros) {
      allZeros=false;
      os << std::setw(6) << std::hex << a << "|";
      for (int c=0; ((c<width) && (a<=maxAddress)); ++c) {
        uint16_t address = static_cast<uint16_t>(a++);
        os << std::setw(6) << std::hex << lut.lutValue(address);
      }
      os << std::endl;
    } else { a += width; }
  }
  if (allZeros)
    { os << "      |   =====  All LUT contents are zero  ===== " << std::endl; }
  return os;
}

#endif /*L1GCTLUT_H_*/
