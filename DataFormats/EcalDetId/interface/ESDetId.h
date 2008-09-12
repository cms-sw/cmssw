#ifndef ECALDETID_ESDETID_H
#define ECALDETID_ESDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

/** \class ESDetId

   Det id for a preshower (endcap) strip
    
   $Id: ESDetId.h,v 1.5 2006/02/27 18:58:30 meridian Exp $
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
  int zside() const { return (id_&0x80000)?(1):(-1); }
  /** get the plane */
  int plane() const { return ((id_>>18)&0x1)+1; }
  /** get the sensor ix */
  int six() const { return (id_>>6)&0x3F; }
  /** get the sensor iy */
  int siy() const { return (id_>>12)&0x3F; }
  /** get the strip */
  int strip() const { return (id_&0x3F); }
  /// get a compact index for arrays [TODO: NEEDS WORK]
  int hashedIndex() const;

  /// check if a valid index combination
  static bool validDetId(int istrip, int ixs, int iys, int iplane, int iz) ;

  static const int IX_MIN=1;
  static const int IY_MIN=1;
  static const int IX_MAX=40;
  static const int IY_MAX=40;
  static const int ISTRIP_MIN=1;
  static const int ISTRIP_MAX=32;

};

std::ostream& operator<<(std::ostream&,const ESDetId& id);


#endif
