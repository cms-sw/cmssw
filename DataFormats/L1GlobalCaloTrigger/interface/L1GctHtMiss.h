#ifndef L1GCTHTMISS_H
#define L1GCTHTMISS_H

#include <ostream>
#include <cstdint>

/*! \file L1GctHtMiss.h
 * 
 * \author: Jim Brooke
 *
 */

/// \class L1GctHtMiss
/// \brief Persistable copy of missing Et measured at Level-1

class L1GctHtMiss {
 public:

  /*! To match the RAW format:
   *  HtMissPhi is on bits 4:0,
   *  HtMissMagnitude is on bits 11:5,
   *  Overflow flag on bit 12,
   *  All other bits will be be zero. */
  enum numberOfBits {
    kHtMissPhiNBits  = 5,
    kHtMissMagNBits  = 7,
    kHtMissPhiMask   = (1 << kHtMissPhiNBits) - 1,
    kHtMissMagMask   = (1 << kHtMissMagNBits) - 1,
    kHtMissPhiShift  = 0,
    kHtMissMagShift  = kHtMissPhiNBits,
    kHtMissOFlowBit  = (1 << (kHtMissPhiNBits + kHtMissMagNBits)),
    kHtMissPhiNBins  = 18,
    kRawCtorMask     = kHtMissOFlowBit | (kHtMissMagMask << kHtMissMagShift) | (kHtMissPhiMask << kHtMissPhiShift)
  };

  L1GctHtMiss();
  
  /// For use with raw data from the unpacker.
  L1GctHtMiss(uint32_t rawData);
  
  /// For use with raw data from the unpacker.
  L1GctHtMiss(uint32_t rawData, int16_t bx);
  
  L1GctHtMiss(unsigned et, unsigned phi, bool oflow);

  L1GctHtMiss(unsigned et, unsigned phi, bool oflow, int16_t bx);

  virtual ~L1GctHtMiss();

  /// name method
  std::string name() const { return "HtMiss"; }

  /// empty method (= false; missing Et is always calculated)
  bool empty() const { return false; }

  /// get the data
  uint32_t raw() const { return m_data; }

  /// get the magnitude
  unsigned et() const { return (m_data >> kHtMissMagShift) & kHtMissMagMask; }

  /// get the overflow
  bool overFlow() const { return (m_data & kHtMissOFlowBit)!=0; }

  /// get the Et
  unsigned phi() const { return (m_data >> kHtMissPhiShift) & kHtMissPhiMask; }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator
  int operator==(const L1GctHtMiss& e) const { return m_data==e.raw(); }

  /// inequality operator
  int operator!=(const L1GctHtMiss& e) const { return !(*this == e); }

 private:

  uint32_t m_data;
  int16_t m_bx;

};

/// Pretty-print operator for L1GctHtMiss
std::ostream& operator<<(std::ostream& s, const L1GctHtMiss& c);


#endif
