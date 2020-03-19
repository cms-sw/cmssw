#ifndef L1GCTETMISS_H
#define L1GCTETMISS_H

#include <ostream>
#include <cstdint>

/*! \file L1GctEtMiss.h
 * \Header file for the GCT energy sum output
 * 
 * \author: Jim Brooke
 *
 */

/// \class L1GctEtMiss
/// \brief Persistable copy of missing Et measured at Level-1

class L1GctEtMiss {
public:
  /*! To match the RAW format, EtMiss is on bits 11:0, Overflow flag on bit 12,
   *  and Phi is on bits 22:16.  All other bits will be be zero. */
  enum numberOfBits {
    kEtMissNBits = 12,
    kEtMissOFlowBit = 1 << kEtMissNBits,
    kEtMissMaxValue = kEtMissOFlowBit - 1,
    kEtMissPhiShift = 16,
    kEtMissPhiNBits = 7,
    kETMissPhiMask = (1 << kEtMissPhiNBits) - 1,
    kEtMissPhiNBins = 72,
    kRawCtorMask = (kETMissPhiMask << kEtMissPhiShift) | kEtMissOFlowBit | kEtMissMaxValue
  };

  L1GctEtMiss();

  /// For use with raw data from the unpacker.
  L1GctEtMiss(uint32_t rawData);

  /// For use with raw data from the unpacker.
  L1GctEtMiss(uint32_t rawData, int16_t bx);

  L1GctEtMiss(unsigned et, unsigned phi, bool oflow);

  L1GctEtMiss(unsigned et, unsigned phi, bool oflow, int16_t bx);

  virtual ~L1GctEtMiss();

  /// name method
  std::string name() const { return "EtMiss"; }

  /// empty method (= false; missing Et is always calculated)
  bool empty() const { return false; }

  /// get the data
  uint32_t raw() const { return m_data; }

  /// get the magnitude
  unsigned et() const { return m_data & kEtMissMaxValue; }

  /// get the overflow
  bool overFlow() const { return (m_data & kEtMissOFlowBit) != 0; }

  /// get the Et
  unsigned phi() const { return (m_data >> kEtMissPhiShift) & kETMissPhiMask; }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator
  int operator==(const L1GctEtMiss& e) const { return m_data == e.raw(); }

  /// inequality operator
  int operator!=(const L1GctEtMiss& e) const { return m_data != e.raw(); }

private:
  uint32_t m_data;
  int16_t m_bx;
};

/// Pretty-print operator for L1GctEtMiss
std::ostream& operator<<(std::ostream& s, const L1GctEtMiss& c);

#endif
