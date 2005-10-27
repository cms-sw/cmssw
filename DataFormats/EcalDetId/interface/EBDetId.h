#ifndef ECALDETID_EBDETID_H
#define ECALDETID_EBDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

/** \class EBDetId
 *  Crystal identifier class for the ECAL barrel
 *
 *
 *  $Id: EBDetId.h,v 1.4 2005/10/12 13:38:21 meridian Exp $
 */


class EBDetId : public DetId {
 public:
  /** Constructor of a null id */
  EBDetId();
  /** Constructor from a raw value */
  EBDetId(uint32_t rawid);
  /** Constructor from crystal ieta and iphi */
  EBDetId(int crystal_ieta, int crystal_iphi);
  /** Constructor from a generic cell id */
  EBDetId(const DetId& id);
  /** Assignment operator from cell id */
  EBDetId& operator=(const DetId& id);

  /// get the subdetector
  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }

  /// get the z-side of the crystal (1/-1)
  int zside() const { return (id_&0x10000)?(1):(-1); }
  /// get the absolute value of the crystal ieta
  int ietaAbs() const { return (id_>>9)&0x7F; }
  /// get the crystal ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the crystal iphi
  int iphi() const { return id_&0x1FF; }
  /// get the HCAL/trigger ieta of this crystal
  int tower_ieta() const { return ((ieta()-1)/5+1)*zside(); }
  /// get the HCAL/trigger iphi of this crystal
  int tower_iphi() const { return ((iphi()-1)/5)+1; }
  /// get the HCAL/trigger iphi of this crystal
  EcalTrigTowerDetId tower() const { return EcalTrigTowerDetId(tower_ieta(),tower_iphi()); }
  /// get the ECAL/SM id
  int ism() const;
  /// get ECAL/crystal number inside SM
  int ic() const;
  /// get a compact index for arrays
  int hashedIndex() const;

  /// range constants
  static const int MIN_IETA = 1;
  static const int MIN_IPHI = 1;
  static const int MAX_IETA = 85;
  static const int MAX_IPHI = 360;
  static const int kChannelsPerCard = 5;
  static const int kTowersInPhi = 4;
};

std::ostream& operator<<(std::ostream& s,const EBDetId& id);


#endif
