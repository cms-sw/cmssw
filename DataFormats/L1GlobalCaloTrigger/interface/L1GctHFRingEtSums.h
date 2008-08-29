#ifndef L1GCTINTERNHFRINGETSUMS_H
#define L1GCTINTERNHFRINGETSUMS_H

#include <ostream>
#include <string>


/// \class L1GctInternHFRingEtSums
/// \brief L1 GCT HF ring Et sums
/// \author Jim Brooke
/// \date August 2008
///
/// Will store four Et sums of 3 bits each
/// 


class L1GctInternHFRingEtSums {

 public:

  /// default constructor (for vector initialisation etc.)
  L1GctInternHFRingEtSums();

  /// destructor
  ~L1GctInternHFRingEtSums();

  // named ctor for unpacker
  static L1GctInternHFRingEtSums fromConcRingSums(const uint16_t capBlock,
						  const uint16_t capIndex,
						  const uint8_t bx,
						  const uint16_t data);

  // named ctor for GCT emulator
  static L1GctInternHFRingEtSums fromGctEmulator(const uint8_t bx,
						 const uint16_t etSumPosEtaRing1,
						 const uint16_t etSumPosEtaRing2,
						 const uint16_t etSumNegEtaRing1,
						 const uint16_t etSumNegEtaRing2);
  
  // optional named ctor for GT if required
  // arguments to be defined
  // static L1GctInternHfRingEtSums fromGtPsb()
  
  
  // metadata
  
  /// get GCT unpacker capture block
  uint16_t capBlock() const { return capBlock_; }

  /// get index within GCT unpacker capture block
  uint16_t capIndex() const { return capIndex_; }

  /// get BX number
  int16_t bx() const { return bx_; }

  /// is the sum non-zero
  bool empty() const { return (data_ == 0); }


  // the actual data

  /// get the raw data
  uint16_t raw() const { return data_; }
  
  /// get an Et sum
  ///  index : sum
  ///    0   :  Ring 1 Positive Rapidity HF Et sum
  ///    1   :  Ring 1 Negative Rapidity HF Et sum
  ///    2   :  Ring 2 Positive Rapidity HF Et sum
  ///    3   :  Ring 2 Negative Rapidity HF Et sum
  uint16_t etSum(unsigned const i);


  /// operators

  /// equality operator
  bool operator==(const L1GctHFRingEtSums& c) const;
  
  /// inequality operator
  bool operator!=(const L1GctHFRingEtSums& c) const { return !(*this == c); }
  
  // private methods
 private:
  
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
