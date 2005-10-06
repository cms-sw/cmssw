#ifndef ECALDETID_ESDETID_H
#define ECALDETID_ESDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

/** \class ESDetId

   Det id for a preshower (endcap) strip
    
   $Id: ESDetId.h,v 1.3 2005/07/27 19:41:12 mansj Exp $
*/

class ESDetId : public DetId {
 public:
  /** Constructor of a null id */
  ESDetId();
  /** Constructor from a raw value */
  ESDetId(uint32_t rawid);  
  /// constructor from strip, ix, iy, plane, and iz
  ESDetId(int strip, int ixs, int iys, int plane, int iz);
  /** constructor from a generic DetId */
  ESDetId(const DetId& id);
  /** assignment from a generic DetId */
  ESDetId& operator=(const DetId& id);

  /// get the subdetector
  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }
  /** get the zside */
  int zside() const { return (id_&0x40000)?(1):(-1); }
  /** get the plane */
  int plane() const { return (id_>>17)&0x1; }
  /** get the sensor ix */
  int six() const { return (id_>>5)&0x3F; }
  /** get the sensor iy */
  int siy() const { return (id_>>11)&0x3F; }
  /** get the strip */
  int strip() const { return id_&0x1F; }
  /// get a compact index for arrays [TODO: NEEDS WORK]
  int hashedIndex() const;

  static const int IX_MIN=1;
  static const int IY_MIN=1;
  static const int IX_MAX=40;
  static const int IY_MAX=40;
  static const int ISTRIP_MIN=0;
  static const int ISTRIP_MAX=31;

};

std::ostream& operator<<(std::ostream&,const ESDetId& id);


#endif
