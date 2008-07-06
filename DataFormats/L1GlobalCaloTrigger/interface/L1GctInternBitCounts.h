#ifndef L1GCTINTERNBITCOUNTS_H
#define L1GCTINTERNBITCOUNTS_H

#include <ostream>
#include <string>


/// \class L1GctInternBitCounts
/// \brief L1 GCT internal ring sum
/// \author Jim Brooke
/// \date June 2008
///


class L1GctInternBitCounts {

 public:

  /// et sum type - not clear this is required
  enum L1GctInternBitCountsType { null }

  /// default constructor (for vector initialisation etc.)
  L1GctInternBitCounts();

  /// construct from individual quantities
  L1GctInternBitCounts(uint16_t capBlock,
		      uint16_t capIndex,
		      int16_t bx
		      );

  /// destructor
  ~L1GctInternBitCounts();


  /// metadata

  /// 'type' of object - not required?
  L1GctInternBitCounts::L1GctInternBitCountsType type() const { return type_; }

  /// get capture block
  uint16_t capBlock() const { return capBlock_; }

  /// get index within capture block
  uint16_t capIndex() const { return capIndex_; }

  /// get BX number
  int16_t bx() const { return bx_; }

  /// is the sum non-zero
  bool empty() const { return (data_ != 0); }


  /// get the actual bits

  /// get the raw data
  uint32_t raw() const { return data_; }
  

  /// operators

  /// equality operator
  bool operator==(const L1GctInternBitCounts& c) const;
  
  /// inequality operator
  bool operator!=(const L1GctInternBitCounts& c) const { return !(*this == c); }
  
  // private methods
 private:
  
  /// set cap block
  void setCapBlock(uint16_t capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(uint16_t bx) { bx_ = bx; }

  /// set type
  void setType(L1GctInternBitCountsType type) { type_ = type; }


  // private data
 private:

  // type of data
  L1GctInternBitCountsType type_;

  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  // the captured data
  uint32_t data_;

 };

std::ostream& operator<<(std::ostream& s, const L1GctInternBitCounts& cand);

#endif
