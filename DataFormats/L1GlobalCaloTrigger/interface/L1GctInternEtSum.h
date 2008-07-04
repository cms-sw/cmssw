#ifndef L1GCTINTERNETSUM_H
#define L1GCTINTERNETSUM_H

#include <ostream>
#include <string>


/// \class L1GctInternEtSum
/// \brief L1 GCT internal energy sum
/// \author Jim Brooke
/// \date June 2008
///


class L1GctInternEtSum {

 public:

  /// et sum type - not clear this is required
  enum L1GctInternEtSumType { null };

  /// default constructor (for vector initialisation etc.)
  L1GctInternEtSum();

  /// construct from individual quantities
  L1GctInternEtSum(uint16_t capBlock,
		   uint16_t capIndex,
		   int16_t bx,
		   uint32_t et,
		   uint8_t oflow);

  /// destructor
  ~L1GctInternEtSum();


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
  bool empty() const { return (data_ != 0); }


  /// get the actual bits

  /// get the raw data
  uint32_t raw() const { return data_; }
  
  /// get et
  uint32_t et() const { return data_ & 0x1ffff; }

  /// get oflow
  uint8_t oflow() const { return (data_>>16) & 0x1; }


  /// operators

  /// equality operator
  bool operator==(const L1GctInternEtSum& c) const;
  
  /// inequality operator
  bool operator!=(const L1GctInternEtSum& c) const { return !(*this == c); }
  
  // private methods
 private:
  
  /// set cap block
  void setCapBlock(uint16_t capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(uint16_t bx) { bx_ = bx; }

  /// set type
  void setType(L1GctInternEtSumType type) { type_ = type; }

  /// set Et sum
  void setEt(uint32_t et);

  /// set overflow bit
  void setOflow(uint8_t oflow);

  // private data
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

std::ostream& operator<<(std::ostream& s, const L1GctInternEtSum& cand);

#endif
