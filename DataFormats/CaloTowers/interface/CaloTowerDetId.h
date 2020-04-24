#ifndef DATAFORMATS_CALOTOWERS_CALOTOWERDETID_H
#define DATAFORMATS_CALOTOWERS_CALOTOWERDETID_H 1

#include "DataFormats/DetId/interface/DetId.h"

/** \class CaloTowerDetId
 *   
 * CaloTowerDetId uses DetId::Det of Calo and subdetId() of 1.
 *
 * \author J. Mans - Minnesota
 */
class CaloTowerDetId : public DetId {
public:
  /** Create a null cellid*/
  CaloTowerDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  explicit CaloTowerDetId(uint32_t rawid);
  /** Constructor from signed tower ieta and iphi  */
  CaloTowerDetId(int tower_ieta, int tower_iphi);
  /** Constructor from a generic cell id */
  CaloTowerDetId(const DetId& id);
  /** Assignment from a generic cell id */
  CaloTowerDetId& operator=(const DetId& id);

  /// get the z-side of the tower (1/-1)
  int zside() const { return (id_&0x2000)?(1):(-1); }
  /// get the absolute value of the tower ieta
  int ietaAbs() const { return (id_>>7)&0x3f; }
  /// get the tower ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the tower iphi
  int iphi() const;

  static const int SubdetId = 1;

};

std::ostream& operator<<(std::ostream&, const CaloTowerDetId& id);

#endif
