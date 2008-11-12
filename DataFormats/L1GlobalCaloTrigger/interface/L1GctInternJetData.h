#ifndef L1GCTINTERNJETDATA_H
#define L1GCTINTERNJETDATA_H

#include <string>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

/// \class L1GctInternJetData
/// \brief L1 GCT internal jet candidate
/// \author Jim Brooke
/// \date June 2006
///


class L1GctInternJetData {

 public:

  enum L1GctInternJetType { null, emulator, jet_cluster, jet_cluster_minimal, jet_precluster,gct_trig_object };

public:

  /// default constructor (for vector initialisation etc.)
  L1GctInternJetData();

  /// construct from individual quantities
  L1GctInternJetData(L1CaloRegionDetId rgn,
		     uint16_t capBlock,
		     uint16_t capIndex,
		     int16_t bx,
		     uint8_t sgnEta,
		     uint8_t oflow,
		     uint16_t et,
		     uint8_t eta,
		     uint8_t phi,
		     uint8_t tauVeto,
		     uint8_t rank);

  // 'named' constructors to avoid confusion

  // emulator calibrated jet ctor
  static L1GctInternJetData fromEmulator(L1CaloRegionDetId rgn,
					 int16_t bx,
					 uint16_t et, 
					 bool overFlow, 
					 bool tauVeto,
					 uint8_t eta,
					 uint8_t phi,
					 uint16_t rank);
				       
  /// construct from "jet_cluster"
  static L1GctInternJetData fromJetCluster(L1CaloRegionDetId rgn,
					   uint16_t capBlock,
					   uint16_t capIndex,
					   int16_t bx,
					   uint32_t data);

  /// construct from "jet_precluster"
  static L1GctInternJetData fromJetPreCluster(L1CaloRegionDetId rgn,
					      uint16_t capBlock,
					      uint16_t capIndex,
					      int16_t bx,
					      uint32_t data);

  /// construct from "jet_cluster_minimal"
  static L1GctInternJetData fromJetClusterMinimal(L1CaloRegionDetId rgn,
						  uint16_t capBlock,
						  uint16_t capIndex,
						  int16_t bx,
						  uint32_t data);
					 
  /// construct from "gct_trig_object"
  static L1GctInternJetData fromGctTrigObject(L1CaloRegionDetId rgn,
					      uint16_t capBlock,
					      uint16_t capIndex,
					      int16_t bx,
					      uint32_t data);


  /// destructor (virtual to prevent compiler warnings)
  virtual ~L1GctInternJetData();


  // getters

  /// 'type' of object
  L1GctInternJetData::L1GctInternJetType type() const { return type_; }

  /// region associated with the candidate
  L1CaloRegionDetId regionId() const { return regionId_; }

  /// was an object really found?
  bool empty() const { return (data_ == 0); }

  /// get capture block
  uint16_t capBlock() const { return capBlock_; }

  /// get index within capture block
  uint16_t capIndex() const { return capIndex_; }

  /// get BX number
  int16_t bx() const { return bx_; }

  /// get the raw data
  uint32_t raw() const { return data_; }
  
  /// get rank bits
  uint16_t rank() const { return data_ & 0x3f; }
  
  /// get tau veto
  uint16_t tauVeto() const { return (data_>>6) & 0x1; }

  /// get phi
  uint16_t phi() const { return (data_>>7) & 0x1f; }

  /// get eta
  uint16_t eta() const { return (data_>>12) & 0xf; }
  
  /// get et
  uint16_t et() const { return (data_>>16) & 0xfff; }

  /// get oflow
  uint16_t oflow() const { return (data_>>28) & 0x1; }

  /// get sign of eta
  uint16_t sgnEta() const { return (data_>>29) & 0x1; }


  // setters

  /// set region
  void setRegionId(L1CaloRegionDetId rgn) { regionId_ = rgn; }

  /// set cap block
  void setCapBlock(uint16_t capBlock) { capBlock_ = capBlock; }

  /// set cap index
  void setCapIndex(uint16_t capIndex) { capIndex_ = capIndex; }

  /// set bx
  void setBx(uint16_t bx) { bx_ = bx; }

  /// set type
  void setType(L1GctInternJetType type) { type_ = type; }

  /// set data
  void setRawData(uint32_t data) { data_ = data; }

  /// construct data word from  components
  void setData(uint8_t sgnEta,
	       uint8_t oflow,
	       uint16_t et,
	       uint8_t eta,
	       uint8_t phi,
	       uint8_t tauVeto,
	       uint8_t rank);


  // operators

  /// equality operator
  bool operator==(const L1GctInternJetData& c) const;
  
  /// inequality operator
  bool operator!=(const L1GctInternJetData& c) const { return !(*this == c); }
  

 private:

  // location in calorimeter (optionally set by unpacker)
  L1CaloRegionDetId regionId_;

  // source of the data
  uint16_t capBlock_;
  uint16_t capIndex_;
  int16_t bx_;

  // type of data
  L1GctInternJetType type_;

  // the captured data
  uint32_t data_;

 };

std::ostream& operator<<(std::ostream& s, const L1GctInternJetData& cand);

#endif
