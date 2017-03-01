#ifndef DATAFORMATS_HCALDETID_HCALDETID_H
#define DATAFORMATS_HCALDETID_HCALDETID_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"


/** \class HcalDetId
 *  Cell identifier class for the HCAL subdetectors, precision readout cells only
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
  static const int kHcalIdMask         = 0xFE000000;

public:
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
  /** Comparison operator */
  bool operator==(DetId id) const;
  bool operator!=(DetId id) const;
  bool operator<(DetId id) const;

  /// get the subdetector
  HcalSubdetector subdet() const { return (HcalSubdetector)(subdetId()); }
  bool oldFormat() const { return ((id_&kHcalIdFormat2)==0)?(true):(false); }
  /// get the z-side of the cell (1/-1)
  int zside() const;
  /// get the absolute value of the cell ieta
  int ietaAbs() const;
  /// get the cell ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the cell iphi
  int iphi() const;
  /// get the tower depth
  int depth() const;
  /// get full depth information for HF
  int hfdepth() const;
  /// get the tower depth
  uint32_t maskDepth() const;
  /// change format
  uint32_t otherForm() const;
  void changeForm();
  uint32_t newForm() const;
  static uint32_t newForm(const uint32_t&);
  /// base detId for HF dual channels
  bool sameBaseDetId(const DetId&) const;
  HcalDetId baseDetId() const;
  /// second PMT anode detId for HF dual channels
  HcalDetId secondAnodeId() const;

  /// get the smallest crystal_ieta of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_ieta_low() const { return ((ieta()-zside())*5)+zside(); }
  /// get the largest crystal_ieta of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_ieta_high() const { return ((ieta()-zside())*5)+5*zside(); }
  /// get the smallest crystal_iphi of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_iphi_low() const; 
  /// get the largest crystal_iphi of the crystal in front of this tower (HB and HE tower 17 only)
  int crystal_iphi_high() const;

  static const HcalDetId Undefined;

private:

  void newFromOld(const uint32_t&);
  static void unpackId(const uint32_t&, int&, int&, int&, int&);
};

std::ostream& operator<<(std::ostream&,const HcalDetId& id);

#endif
