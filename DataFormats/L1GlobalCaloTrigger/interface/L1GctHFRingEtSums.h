#ifndef L1GCTHFRINGETSUMS_H
#define L1GCTHFRINGETSUMS_H

#include <ostream>
#include <string>
#include <cstdint>

/// \class L1GctHFRingEtSums
/// \brief L1 GCT HF ring Et sums
/// \author Jim Brooke
/// \date August 2008
///
/// Will store four Et sums of 3 bits each
///

class L1GctHFRingEtSums {
public:
  static const unsigned N_SUMS = 4;

public:
  /// default constructor (for vector initialisation etc.)
  L1GctHFRingEtSums();

  /// destructor
  ~L1GctHFRingEtSums();

  /// named ctor for unpacker
  /// note that this expects a 32 bit word that also contains
  /// the HF bit counts, which are ignored
  static L1GctHFRingEtSums fromConcRingSums(const uint16_t capBlock,
                                            const uint16_t capIndex,
                                            const int16_t bx,
                                            const uint32_t data);

  /// named ctor for GCT emulator
  static L1GctHFRingEtSums fromGctEmulator(const int16_t bx,
                                           const uint16_t etSumPosEtaRing1,
                                           const uint16_t etSumNegEtaRing1,
                                           const uint16_t etSumPosEtaRing2,
                                           const uint16_t etSumNegEtaRing2);

  // optional named ctor for GT if required
  // arguments to be defined
  // static L1GctHfRingEtSums fromGtPsb()

  // get number of ring sums
  static unsigned nSums() { return N_SUMS; }

  // getters

  /// get GCT unpacker capture block
  uint16_t capBlock() const { return capBlock_; }

  /// get index within GCT unpacker capture block
  uint16_t capIndex() const { return capIndex_; }

  /// get BX number
  int16_t bx() const { return bx_; }

  /// is the sum non-zero
  bool empty() const { return (data_ == 0); }

  /// get the raw data
  uint16_t raw() const { return data_; }

  /// get an Et sum
  ///  index : sum
  ///    0   :  Ring 1 Positive Rapidity HF Et sum
  ///    1   :  Ring 1 Negative Rapidity HF Et sum
  ///    2   :  Ring 2 Positive Rapidity HF Et sum
  ///    3   :  Ring 2 Negative Rapidity HF Et sum
  uint16_t etSum(unsigned const i) const;

  // setters

  /// set cap block
  void setCapBlock(uint16_t capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(uint16_t bx) { bx_ = bx; }

  /// set a sum
  void setEtSum(unsigned i, uint16_t et);

  /// set the raw data
  void setData(uint32_t data) { data_ = data; }

  /// operators

  /// equality operator
  bool operator==(const L1GctHFRingEtSums& c) const;

  /// inequality operator
  bool operator!=(const L1GctHFRingEtSums& c) const { return !(*this == c); }

private:
  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  // the captured data
  uint16_t data_;
};

std::ostream& operator<<(std::ostream& s, const L1GctHFRingEtSums& cand);

#endif
