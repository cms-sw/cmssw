#ifndef DATAFORMATS_CALOTOWERS_CALOTOWERDETID_H
#define DATAFORMATS_CALOTOWERS_CALOTOWERDETID_H 1

#include "DataFormats/HcalDetId/interface/HcalCompositeDetId.h"

namespace cms {

  /** \class CaloTowerDetId
      
  $Date: $
  $Revision: $
   \author J. Mans - Minnesota
  */
  class CaloTowerDetId : public HcalCompositeDetId {
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
  };

  std::ostream& operator<<(std::ostream&, const CaloTowerDetId& id);
}

#endif
