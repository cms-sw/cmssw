#ifndef L1GCTETTOTAL_H
#define L1GCTETTOTAL_H

#include <ostream>
#include <cstdint>

/*! \file L1GctEtTotal.h
 * \Header file for the GCT energy sum output
 * 
 * \author: Jim Brooke
 *
 */

/// \class L1GctEtTotal
/// \brief Persistable copy of total Et measured at Level-1

class L1GctEtTotal {
public:
  enum numberOfBits {
    kEtTotalNBits = 12,
    kEtTotalOFlowBit = 1 << kEtTotalNBits,
    kEtTotalMaxValue = kEtTotalOFlowBit - 1,
    kRawCtorMask = kEtTotalOFlowBit | kEtTotalMaxValue
  };

  L1GctEtTotal();
  L1GctEtTotal(uint16_t rawData);
  L1GctEtTotal(uint16_t rawData, int16_t bx);
  L1GctEtTotal(unsigned et, bool oflow);
  L1GctEtTotal(unsigned et, bool oflow, int16_t bx);
  virtual ~L1GctEtTotal();

  /// name method
  std::string name() const { return "EtTotal"; }

  /// empty method (= false; total Et is always calculated)
  bool empty() const { return false; }

  /// get the data
  uint16_t raw() const { return m_data; }

  /// get the Et
  unsigned et() const { return m_data & kEtTotalMaxValue; }

  /// get the overflow
  bool overFlow() const { return (m_data & kEtTotalOFlowBit) != 0; }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator
  int operator==(const L1GctEtTotal& e) const { return m_data == e.raw(); }

  /// inequality operator
  int operator!=(const L1GctEtTotal& e) const { return m_data != e.raw(); }

private:
  uint16_t m_data;
  int16_t m_bx;
};

/// Pretty-print operator for L1GctEtTotal
std::ostream& operator<<(std::ostream& s, const L1GctEtTotal& c);

#endif
