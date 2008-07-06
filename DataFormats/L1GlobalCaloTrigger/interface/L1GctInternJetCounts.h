#ifndef L1GCTINTERNJETCOUNTS_H
#define L1GCTINTERNJETCOUNTS_H

#include <ostream>
#include <string>


/// \class L1GctInternJetCounts
/// \brief L1 GCT internal energy sum
/// \author Jim Brooke
/// \date June 2008
///


class L1GctInternJetCounts {

 public:

  /// et sum type - not clear this is required
  enum L1GctInternJetCountsType { null }

  /// default constructor (for vector initialisation etc.)
  L1GctInternJetCounts();

  /// construct from individual quantities
  L1GctInternJetCounts(uint16_t capBlock,
		       uint16_t capIndex,
		       int16_t bx,
		       );

  /// destructor
  ~L1GctInternJetCounts();


  /// metadata

  /// 'type' of object - not required?
  L1GctInternJetCounts::L1GctInternJetCountsType type() const { return type_; }

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
  bool operator==(const L1GctInternJetCounts& c) const;
  
  /// inequality operator
  bool operator!=(const L1GctInternJetCounts& c) const { return !(*this == c); }
  
  // private methods
 private:
  
  /// set cap block
  void setCapBlock(uint16_t capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(uint16_t bx) { bx_ = bx; }

  /// set type
  void setType(L1GctInternJetCountsType type) { type_ = type; }


  // private data
 private:

  // type of data
  L1GctInternJetCountsType type_;

  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  // the captured data
  uint32_t data_;

 };

std::ostream& operator<<(std::ostream& s, const L1GctInternJetCounts& cand);

#endif
