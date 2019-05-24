#ifndef L1GCTINTERNETSUM_H
#define L1GCTINTERNETSUM_H

#include <ostream>
#include <string>
#include <cstdint>

/// \class L1GctInternEtSum
/// \brief L1 GCT internal energy sum
/// \author Jim Brooke
/// \date June 2008
///

class L1GctInternEtSum {
public:
  /// et sum type - not clear this is required
  enum L1GctInternEtSumType {
    null,
    jet_tot_et,      // from jet_tot_et_and_ht in leaf output
    jet_tot_ht,      // from jet_tot_et_and_ht in leaf output
    jet_miss_et,     // leaf output
    total_et_or_ht,  // conc input, wheel input and output
    miss_etx_or_ety  // conc input, wheel input and output
  };

  enum numberOfBits {
    kTotEtOrHtNBits = 12,
    kJetMissEtNBits = 17,
    kMissExOrEyNBits = 20,
    kTotEtOrHtOFlowBit = 1 << kTotEtOrHtNBits,
    kJetMissEtOFlowBit = 1 << kJetMissEtNBits,
    kMissExOrEyOFlowBit = 1 << kMissExOrEyNBits,
    kTotEtOrHtMaxValue = kTotEtOrHtOFlowBit - 1,
    kJetMissEtMaxValue = kJetMissEtOFlowBit - 1,
    kMissExOrEyMaxValue = kMissExOrEyOFlowBit - 1,
    kTotEtOrHtRawCtorMask = kTotEtOrHtOFlowBit | kTotEtOrHtMaxValue,
    kJetMissEtRawCtorMask = kJetMissEtOFlowBit | kJetMissEtMaxValue,
    kMissExOrEyRawCtorMask = kMissExOrEyOFlowBit | kMissExOrEyMaxValue
  };

  /// default constructor (for vector initialisation etc.)
  L1GctInternEtSum();

  /// construct from individual quantities
  L1GctInternEtSum(uint16_t capBlock, uint16_t capIndex, int16_t bx, uint32_t et, uint8_t oflow);

  /// destructor
  ~L1GctInternEtSum();

  // named ctors
  static L1GctInternEtSum fromJetTotEt(uint16_t capBlock, uint16_t capIndex, int16_t bx, uint32_t data);

  static L1GctInternEtSum fromJetTotHt(uint16_t capBlock, uint16_t capIndex, int16_t bx, uint32_t data);

  static L1GctInternEtSum fromJetMissEt(uint16_t capBlock, uint16_t capIndex, int16_t bx, uint32_t data);

  static L1GctInternEtSum fromTotalEtOrHt(uint16_t capBlock, uint16_t capIndex, int16_t bx, uint32_t data);

  static L1GctInternEtSum fromMissEtxOrEty(uint16_t capBlock, uint16_t capIndex, int16_t bx, uint32_t data);

  static L1GctInternEtSum fromEmulatorJetTotEt(unsigned totEt, bool overFlow, int16_t bx);

  static L1GctInternEtSum fromEmulatorJetTotHt(unsigned totHt, bool overFlow, int16_t bx);

  static L1GctInternEtSum fromEmulatorJetMissEt(int missEtxOrEty, bool overFlow, int16_t bx);

  static L1GctInternEtSum fromEmulatorTotalEtOrHt(unsigned totEtOrHt, bool overFlow, int16_t bx);

  static L1GctInternEtSum fromEmulatorMissEtxOrEty(int missEtxOrEty, bool overFlow, int16_t bx);

  /// metadata

  /// 'type' of object - not required?
  L1GctInternEtSum::L1GctInternEtSumType type() const { return type_; }

  /// get capture block
  uint16_t capBlock() const { return capBlock_; }

  /// get index within capture block
  uint16_t capIndex() const { return capIndex_; }

  /// get BX number
  int16_t bx() const { return bx_; }

  /// is the sum non-zero
  bool empty() const { return (data_ == 0); }

  /// get the actual bits

  /// get the raw data
  uint32_t raw() const { return data_; }

  /// get value
  uint32_t value() const { return data_ & 0x7fffffff; }

  /// get et
  uint32_t et() const { return value(); }

  /// get count
  uint32_t count() const { return value(); }

  /// get oflow
  uint8_t oflow() const { return (data_ >> 31) & 0x1; }

  // setters

  /// set cap block
  void setCapBlock(uint16_t capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(uint16_t bx) { bx_ = bx; }

  /// set type
  void setType(L1GctInternEtSumType type) { type_ = type; }

  /// set value
  void setValue(uint32_t val);

  /// set Et sum
  void setEt(uint32_t et);

  /// set count
  void setCount(uint32_t count);

  /// set overflow bit
  void setOflow(uint8_t oflow);

  /// operators

  /// equality operator
  bool operator==(const L1GctInternEtSum& c) const;

  /// inequality operator
  bool operator!=(const L1GctInternEtSum& c) const { return !(*this == c); }

private:
  // type of data
  L1GctInternEtSumType type_;

  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  // the captured data
  uint32_t data_;
};

std::ostream& operator<<(std::ostream& s, const L1GctInternEtSum& c);

#endif
