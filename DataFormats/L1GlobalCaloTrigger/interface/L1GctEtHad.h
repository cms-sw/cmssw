#ifndef L1GCTETHAD_H
#define L1GCTETHAD_H

#include <ostream>
#include <cstdint>

/*! \file L1GctEtHad.h
 * \Header file for the GCT energy sum output
 * 
 * \author: Jim Brooke
 *
 */

/// \class L1GctEtHad
/// \brief Persistable copy of total Ht measured at Level-1

class L1GctEtHad {
public:
  enum numberOfBits {
    kEtHadNBits = 12,
    kEtHadOFlowBit = 1 << kEtHadNBits,
    kEtHadMaxValue = kEtHadOFlowBit - 1,
    kRawCtorMask = kEtHadOFlowBit | kEtHadMaxValue
  };

  L1GctEtHad();
  L1GctEtHad(uint16_t rawData);
  L1GctEtHad(uint16_t rawData, int16_t bx);
  L1GctEtHad(unsigned et, bool oflow);
  L1GctEtHad(unsigned et, bool oflow, int16_t bx);
  virtual ~L1GctEtHad();

  /// name method
  std::string name() const { return "EtHad"; }

  /// empty method (= false; hadronic Et is always calculated)
  bool empty() const { return false; }

  /// get the data
  uint16_t raw() const { return m_data; }

  /// get the Et
  unsigned et() const { return m_data & kEtHadMaxValue; }

  /// get the overflow
  bool overFlow() const { return (m_data & kEtHadOFlowBit) != 0; }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator
  int operator==(const L1GctEtHad& e) const { return m_data == e.raw(); }

  /// inequality operator
  int operator!=(const L1GctEtHad& e) const { return m_data != e.raw(); }

private:
  uint16_t m_data;
  int16_t m_bx;
};

/// Pretty-print operator for L1GctEtHad
std::ostream& operator<<(std::ostream& s, const L1GctEtHad& c);

#endif
