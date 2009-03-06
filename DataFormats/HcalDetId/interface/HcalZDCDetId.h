#ifndef HcalZDCDetId_h_included
#define HcalZDCDetId_h_included 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

/** \class HcalZDCDetId
  *  
  *  Contents of the HcalZDCDetId :
  *     [6]   Z position (true for positive)
  *     [5:4] Section (EM/HAD/Lumi)
  *     [3:0] Channel (depth)
  *
  * $Date: 2006/06/16 16:45:04 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalZDCDetId : public DetId {
public:
  enum Section { Unknown=0, EM=1, HAD=2, LUM=3 };
  // 1 => CaloTower, 3 => Castor
  static const int SubdetectorId = 2;

  /** Create a null cellid*/
  HcalZDCDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HcalZDCDetId(uint32_t rawid);
  /** Constructor from section, eta sign, and depth/channel */
  HcalZDCDetId(Section section, bool true_for_positive_eta, int depth);
  /** Constructor from a generic cell id */
  HcalZDCDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalZDCDetId& operator=(const DetId& id);

  /// get the z-side of the cell (1/-1)
  int zside() const { return (id_&0x40)?(1):(-1); }
  /// get the section
  Section section() const { return (Section)((id_>>4)&0x3); }
  /// get the depth
  int depth() const { return id_&0xF; }
  /// get the channel (equivalent to depth)
  int channel() const { return id_&0xF; }
};

std::ostream& operator<<(std::ostream&,const HcalZDCDetId& id);


#endif // HcalZDCDetId_h_included
