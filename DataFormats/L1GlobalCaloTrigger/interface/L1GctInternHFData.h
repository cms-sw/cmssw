#ifndef L1GCTINTERNHFDATA_H
#define L1GCTINTERNHFDATA_H

#include <ostream>
#include <string>
#include <cstdint>

/// \class L1GctInternHFData
/// \brief L1 GCT internal ring sums and/or bit counts
/// \author Jim Brooke
/// \date June 2008
///
/// Will store 4 sums/counts of up to 8 bits each
/// 


class L1GctInternHFData {

 public:

  /// et sum type - not clear this is required
  enum L1GctInternHFDataType { null,
			       conc_hf_ring_et_sums,
			       conc_hf_bit_counts,
                               wheel_hf_ring_et_sums,
                               wheel_hf_bit_counts
  };
  
  /// default constructor (for vector initialisation etc.)
  L1GctInternHFData();

  /// destructor
  ~L1GctInternHFData();

  static L1GctInternHFData fromConcRingSums(const uint16_t capBlock,
					    const uint16_t capIndex,
					    const int16_t bx,
					    const uint32_t data);
  
  static L1GctInternHFData fromConcBitCounts(const uint16_t capBlock,
					     const uint16_t capIndex,
					     const int16_t bx,
					     const uint32_t data);
  
  static L1GctInternHFData fromWheelRingSums(const uint16_t capBlock,
                                             const uint16_t capIndex,
                                             const int16_t bx,
                                             const uint32_t data);
  
  static L1GctInternHFData fromWheelBitCounts(const uint16_t capBlock,
                                              const uint16_t capIndex,
                                              const int16_t bx,
                                              const uint32_t data);

  /// metadata

  /// 'type' of object 
  L1GctInternHFData::L1GctInternHFDataType type() const { return type_; }

  /// get capture block
  uint16_t capBlock() const { return capBlock_; }

  /// get index within capture block
  uint16_t capIndex() const { return capIndex_; }

  /// get BX number
  int16_t bx() const { return bx_; }

  /// is the sum non-zero
  bool empty() const { return (data_ == 0); }


  /// get the actual data

  /// is this ring sums or bit counts?
  bool isRingSums() const { return (type_ == conc_hf_ring_et_sums || type_ == wheel_hf_ring_et_sums); }

  /// get the raw data
  uint32_t raw() const { return data_; }
  
  /// get value
  uint16_t value(unsigned i) const;

  /// get the et sums
  uint16_t et(unsigned i) const;

  /// get the counts
  uint16_t count(unsigned i) const;


  // setters  
  
  /// set cap block
  void setCapBlock(uint16_t const capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t const capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(int16_t const bx) { bx_ = bx; }

  /// set type
  void setType(L1GctInternHFDataType type) { type_ = type; }

  /// set value
  void setValue(unsigned const i, uint16_t const val);

  /// set the sum
  void setEt(unsigned const i, uint16_t const et);

  /// set the count
  void setCount(unsigned const i, uint16_t const count);

  void setData(uint32_t const data) { data_ = data; }
  

  /// operators

  /// equality operator
  bool operator==(const L1GctInternHFData& c) const;
  
  /// inequality operator
  bool operator!=(const L1GctInternHFData& c) const { return !(*this == c); }


 private:

  // type of data
  L1GctInternHFDataType type_;

  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  // the captured data
  uint32_t data_;

 };

std::ostream& operator<<(std::ostream& s, const L1GctInternHFData& cand);

#endif
