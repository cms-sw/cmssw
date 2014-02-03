#ifndef DATAFORMATS_HCALDETID_HCALDETID_H
#define DATAFORMATS_HCALDETID_HCALDETID_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"


/** \class HcalDetId
 *  Cell identifier class for the HCAL subdetectors, precision readout cells only
 *
 *  $Date: 2012/11/12 20:52:53 $
 *  $Revision: 1.21 $
 *  \author J. Mans - Minnesota
 *
 *  Rev.1.11: A.Kubik,R.Ofierzynski: add the hashed_index
 */
class HcalDetId : public DetId {
public:
  /** Create a null cellid*/
  HcalDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HcalDetId(uint32_t rawid);
  /** Constructor from subdetector, signed tower ieta,iphi,and depth */
  HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth, bool oldFormat = false);
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
  bool oldFormat() const { return ((id_&0x1000000)==0)?(true):(false); }
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
  /// reverse format
  uint32_t otherForm() const;
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
