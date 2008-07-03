#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternJetData.h"

L1GctInternJetData::L1GctInternJetData() {

}

/// construct from individual quantities
L1GctInternJetData::L1GctInternJetData(L1CaloRegionDetId rgn,
				       uint16_t capBlock,
				       uint16_t capIndex,
				       int16_t bx,
				       uint8_t sgnEta,
				       uint8_t oflow,
				       uint16_t et,
				       uint8_t eta,
				       uint8_t phi,
				       uint8_t tauVeto,
				       uint8_t rank) :
  regionId_(rgn),
  capBlock_(capBlock),
  capIndex_(capIndex),
  bx_(bx),
  type_(null),
  data_(0)
{
  setData(sgnEta, oflow, et, eta, phi, tauVeto, rank);
}

// 'named' constructors to avoid confusion
/// construct from "jet_cluster"
L1GctInternJetData L1GctInternJetData::fromJetCluster(L1CaloRegionDetId rgn,
						      uint16_t capBlock,
						      uint16_t capIndex,
						      int16_t bx,
						      uint32_t data) {
  L1GctInternJetData d;

  d.setRegionId(rgn);
  d.setCapBlock(capBlock);
  d.setCapIndex(capIndex);
  d.setBx(bx);
  d.setType(jet_cluster);
  d.setRawData(data&0x1fffffff);

  return d;
}

/// construct from "jet_precluster"
L1GctInternJetData L1GctInternJetData::fromJetPreCluster(L1CaloRegionDetId rgn,
							 uint16_t capBlock,
							 uint16_t capIndex,
							 int16_t bx,
							 uint32_t data) {
  L1GctInternJetData d;

  d.setRegionId(rgn);
  d.setCapBlock(capBlock);
  d.setCapIndex(capIndex);
  d.setBx(bx);
  d.setType(jet_precluster);
  d.setData( 0, (data>>10)&0x1, data&0x3ff, (data>>12)&0xf, 0, (data>>11)&0x1, 0 );

  return d;
}

/// construct from "jet_cluster_minimal"
L1GctInternJetData L1GctInternJetData::fromJetClusterMinimal(L1CaloRegionDetId rgn,
							     uint16_t capBlock,
							     uint16_t capIndex,
							     int16_t bx,
							     uint32_t data) {
  L1GctInternJetData d;

  d.setRegionId(rgn);
  d.setCapBlock(capBlock);
  d.setCapIndex(capIndex);
  d.setBx(bx);
  d.setType(jet_cluster_minimal);
  d.setData( 0, (data>>10)&0x1, 0, (data>>6)&0xf, (data>>12)&0x1, (data>>1)&0x1, data&0x3f );

  return d;
}

/// construct from "gct_trig_object"
L1GctInternJetData L1GctInternJetData::fromGctTrigObject(L1CaloRegionDetId rgn,
							 uint16_t capBlock,
							 uint16_t capIndex,
							 int16_t bx,
							 uint32_t data) {
  L1GctInternJetData d;

  d.setRegionId(rgn);
  d.setCapBlock(capBlock);
  d.setCapIndex(capIndex);
  d.setBx(bx);
  d.setType(gct_trig_object);
  d.setData( (data>>10)&0x1, 0, 0, (data>>6)&0xf, (data>>11)&0x1f, 0, data&0x3f );

  return d;
}

/// destructor (virtual to prevent compiler warnings)
L1GctInternJetData::~L1GctInternJetData() {

}


/// construct
void L1GctInternJetData::setData(uint8_t sgnEta,
				 uint8_t oflow,
				 uint16_t et,
				 uint8_t eta,
				 uint8_t phi,
				 uint8_t tauVeto,
				 uint8_t rank) {
  data_ = rank & 0x3f;
  data_ += (tauVeto & 0x1) << 6;
  data_ += (phi & 0x1f) << 7;
  data_ += (eta & 0xf) << 12;
  data_ += (et & 0xfff) << 16;
  data_ += (oflow & 0x1) << 28;
  data_ += (sgnEta & 0x1) << 29;
}
 

/// operators

/// equality operator
bool L1GctInternJetData::operator==(const L1GctInternJetData& c) const {
  return (type_ == c.type() && data_ == c.raw() && rgn_ == c.regionId() && bx_ == c.bx());
}

/// inequality operator
bool L1GctInternJetData::operator!=(const L1GctInternJetData& c) const { 
  return !(*this == c); 
}

/// pretty print
std::ostream& operator<<(std::ostream& s, const L1GctInternJetData& c) {
  s << "L1GctInternJetData :";
  if (c.empty()) { 
    s << " empty"; 
    return s; 
  }
  else {
    s << " oflow=" << oflow_;
    s << " et=" << et_;
    s << " eta=" << eta_;
    s << " phi=" << phi_;
    s << " tauVeto=" << tauVeto_;
    s << " rank=" << rank_;
  }
  return s; 
}
