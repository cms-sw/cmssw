#ifndef L1CALOREGIONDETID_H
#define L1CALOREGIONDETID_H

#include "DataFormats/DetId/interface/DetId.h"

/** \class L1CaloRegionDetId
 *  Cell identifier class for L1 Calo Trigger Regions (4x4 trigger tower sums)
 *
 *  $Date: $
 *  $Revision: $
 *  \author Jim Brooke 
*/

/**
 * Stores eta value in bits 2-0, eta sign in bit 3, phi value in bits 8-4
 *
 *
 */

class L1CaloRegionDetId : public DetId {

 public:

  /// create null id
  L1CaloRegionDetId();
  
  /// create id from raw data (0=invalid code?)
  L1CaloRegionDetId(uint32_t rawid);

  /// create id from global eta, phi indices
  L1CaloRegionDetId(int ieta, int iphi);

  /// create id from RCT crate, RCT card, RCT region (within card)
  L1CaloRegionDetId(unsigned icrate, unsigned icard, unsigned irgn);

  /// which half of the detector?
  int zside() const { return (id_&0x8)?(1):(-1); }

  /// absolute eta value
  int ietaAbs() const { return id_ & 0x7; }

  /// global eta index (-10 - +10)
  int ieta() const { return zside()*ietaAbs(); }

  /// global phi index (0-18)
  int iphi() const { return (id_>>4) & 0x1f; }

  /// return central or forward type
  bool isForward() const { return (ietaAbs()>6); }

  /// return RCT crate number (0-17)
  unsigned rctCrate() const { return 0; }

  /// return RCT card number (0-6)
  unsigned rctCard() const { return 0; }

  /// return RCT region index (0-21 ???)
  unsigned rctRegion() const { return 0; }

  /// return local RCT eta index (0-10)
  unsigned rctEta() const { return ietaAbs() ; }

  /// return local RCT phi index (0-1)
  unsigned rctPhi() const { return iphi() % 2; }

};

#endif
