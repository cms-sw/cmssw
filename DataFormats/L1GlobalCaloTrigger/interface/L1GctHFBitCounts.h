#ifndef L1GCTHFBITCOUNTS_H
#define L1GCTHFBITCOUNTS_H

#include <ostream>
#include <string>
#include <cstdint>

/// \class L1GctHFBitCounts
/// \brief L1 GCT HF ring Et sums
/// \author Jim Brooke
/// \date August 2008
///
/// Will store four Et sums of 3 bits each
///

class L1GctHFBitCounts {
public:
  static const unsigned N_SUMS = 4;

public:
  /// default constructor (for vector initialisation etc.)
  L1GctHFBitCounts();

  /// destructor
  ~L1GctHFBitCounts();

  /// named ctor for unpacker
  /// note this expects a 32 bit word that also contains the
  /// HF ring Et sums, which are ignored
  static L1GctHFBitCounts fromConcHFBitCounts(const uint16_t capBlock,
                                              const uint16_t capIndex,
                                              const int16_t bx,
                                              const uint32_t data);

  /// named ctor for GCT emulator
  static L1GctHFBitCounts fromGctEmulator(const int16_t bx,
                                          const uint16_t bitCountPosEtaRing1,
                                          const uint16_t bitCountNegEtaRing1,
                                          const uint16_t bitCountPosEtaRing2,
                                          const uint16_t bitCountNegEtaRing2);

  // optional named ctor for GT if required
  // arguments to be defined
  // static L1GctHFBitCounts fromGtPsb()

  // getters

  // get number of ring sums
  static unsigned nCounts() { return N_SUMS; }

  /// get GCT unpacker capture block
  uint16_t capBlock() const { return capBlock_; }

  /// get index within GCT unpacker capture block
  uint16_t capIndex() const { return capIndex_; }

  /// get BX number
  int16_t bx() const { return bx_; }

  /// is the sum non-zero
  bool empty() const { return (data_ == 0); }

  /// the raw data
  uint16_t raw() const { return data_; }

  /// get a bit count
  ///  index : sum
  ///    0   :  Ring 1 Positive Rapidity HF bit count
  ///    1   :  Ring 1 Negative Rapidity HF bit count
  ///    2   :  Ring 2 Positive Rapidity HF bit count
  ///    3   :  Ring 2 Negative Rapidity HF bit count
  uint16_t bitCount(unsigned const i) const;

  // setters

  /// set cap block
  void setCapBlock(uint16_t capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(int16_t bx) { bx_ = bx; }

  /// set a sum
  void setBitCount(unsigned i, uint16_t c);

  /// set the raw data
  void setData(uint32_t data) { data_ = data; }

  /// operators

  /// equality operator
  bool operator==(const L1GctHFBitCounts& c) const;

  /// inequality operator
  bool operator!=(const L1GctHFBitCounts& c) const { return !(*this == c); }

private:
  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  // the captured data
  uint16_t data_;
};

std::ostream& operator<<(std::ostream& s, const L1GctHFBitCounts& cand);

#endif
