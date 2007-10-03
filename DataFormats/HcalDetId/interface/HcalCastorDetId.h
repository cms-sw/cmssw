#ifndef HcalCastorDetId_h_included
#define HcalCastorDetId_h_included 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

/** \class HcalCastorDetId
  *  
  *  Contents of the HcalCastorDetId :
  *     [9]   Z position (true for positive)
  *     [8:7] Section (EM/HAD)
  *     [6:4] sector (depth)
  *	[3:0] module
  * $Date: 2007/09/20 15:39:08 $
  * $Revision: 1.1 $
  * \author P. Katsas - UoA
  */
class HcalCastorDetId : public DetId {
public:
  enum Section { Unknown=0, EM=1, HAD=2 };
  // 1 => CaloTower, 2 => ZDC, 3 => Castor
  static const int SubdetectorId = 3;

  /** Create a null cellid*/
  HcalCastorDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HcalCastorDetId(uint32_t rawid);
  /** Constructor from section, z-side, sector and module */
  HcalCastorDetId(Section section, bool true_for_positive_eta, int sector, int module);
  /** Constructor from a generic cell id */
  HcalCastorDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalCastorDetId& operator=(const DetId& id);

  /// get the z-side of the cell (1/-1)
  int zside() const { return (id_&0x40)?(1):(-1); }
  /// get the section
  Section section() const { return (Section)((id_>>7)&0x3); }
  /// get the module (1-2 for EM, 1-12 for HAD)
  int module() const { return id_&0xF; }
  /// get the sector (1-16)
  int sector() const { return (id_>>6)&0x3; }
  // get the individual cell id
//  int channel() const;
};

std::ostream& operator<<(std::ostream&,const HcalCastorDetId& id);


#endif // HcalCastorDetId_h_included

