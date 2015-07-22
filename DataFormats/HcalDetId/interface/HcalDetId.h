#ifndef DATAFORMATS_HCALDETID_HCALDETID_H
#define DATAFORMATS_HCALDETID_HCALDETID_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"


/** \class HcalDetId
 *  Cell identifier class for the HCAL subdetectors, precision readout cells only
 *
 *  \author J. Mans - Minnesota
 *
 *  Rev.1.11: A.Kubik,R.Ofierzynski: add the hashed_index
 */
class HcalDetId : public DetId {
public:
  static const int kHcalPhiMask1       = 0x7F;
  static const int kHcalPhiMask2       = 0x3FF;
  static const int kHcalEtaOffset1     = 7;
  static const int kHcalEtaOffset2     = 10;
  static const int kHcalEtaMask1       = 0x3F;
  static const int kHcalEtaMask2       = 0x1FF;
  static const int kHcalZsideMask1     = 0x2000;
  static const int kHcalZsideMask2     = 0x80000;
  static const int kHcalDepthOffset1   = 14;
  static const int kHcalDepthOffset2   = 20;
  static const int kHcalDepthMask1     = 0x1F;
  static const int kHcalDepthMask2     = 0xF;
  static const int kHcalDepthSet1      = 0x1C000;
  static const int kHcalDepthSet2      = 0xF00000;
  static const int kHcalIdFormat2      = 0x1000000;
  /** Create a null cellid*/
  HcalDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HcalDetId(uint32_t rawid);
  /** Constructor from subdetector, signed tower ieta,iphi,and depth */
  HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth);
  /** Constructor from a generic cell id */
  HcalDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalDetId& operator=(const DetId& id);

  /// get the subdetector
  HcalSubdetector subdet() const { return (HcalSubdetector)(subdetId()); }
  /// get the z-side of the cell (1/-1)
  int zside() const { return (id_&kHcalZsideMask1)?(1):(-1); }
  /// get the absolute value of the cell ieta
  int ietaAbs() const { return (id_>>kHcalEtaOffset1)&kHcalEtaMask1; }
  /// get the cell ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the cell iphi
  int iphi() const { return id_&kHcalPhiMask1; }
  /// get the tower depth
  int depth() const { return (id_>>kHcalDepthOffset1)&kHcalDepthMask1; }
  /// get the tower depth
  uint32_t maskDepth() const { return (id_ | kHcalDepthSet1); }
  /// get the smallest crystal_ieta of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_ieta_low() const { return ((ieta()-zside())*5)+zside(); }
  /// get the largest crystal_ieta of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_ieta_high() const { return ((ieta()-zside())*5)+5*zside(); }
  /// get the smallest crystal_iphi of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_iphi_low() const; 
  /// get the largest crystal_iphi of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_iphi_high() const;

  static const HcalDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HcalDetId& id);

#endif
