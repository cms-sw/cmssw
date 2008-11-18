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
  enum L1GctInternEtSumType { null,
			      wheel_hf_ring_et_sum,
			      wheel_hf_ring_bit_count,  // aka wheel_hf_ring_jets_above_threshold
			      jet_tot_et,  // from jet_tot_et_and_ht
			      jet_miss_et,
			      total_et  // from total_et_or_ht
  };

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

  // named ctors
  /// use this for wheel_hf_et_sum
  static L1GctInternEtSum fromWheelHfRingSum(uint16_t capBlock,
					     uint16_t capIndex,
					     int16_t bx,
					     uint16_t data);
  
  static L1GctInternEtSum fromWheelHfBitCount(uint16_t capBlock,
					      uint16_t capIndex,
					      int16_t bx,
					      uint16_t data);
  

  static L1GctInternEtSum fromJetTotEt(uint16_t capBlock,
				       uint16_t capIndex,
				       int16_t bx,
				       uint16_t data);
  

  static L1GctInternEtSum fromJetMissEt(uint16_t capBlock,
					uint16_t capIndex,
					int16_t bx,
					uint32_t data);
  

  static L1GctInternEtSum fromTotalEt(uint16_t capBlock,
				      uint16_t capIndex,
				      int16_t bx,
				      uint32_t data);

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
  uint8_t oflow() const { return (data_>>31) & 0x1; }


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
