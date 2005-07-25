#ifndef ECALDETID_EEDETID_H
#define ECALDETID_EEDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EEDetId
 *  Crystal/cell identifier class for the ECAL endcap
 *
 *
 *  $Id: $
 */

namespace cms
{

  class EEDetId : public DetId {
  public:
    /** Constructor of a null id */
    EEDetId();
    /** Constructor from a raw value */
    EEDetId(uint32_t rawid);
    /** Constructor from crystal ix,iy,iz (iz=+1/-1) */
    EEDetId(int ix, int iy, int iz);  
    /** Constructor from a generic cell id */
    EEDetId(const DetId& id);
    /// assignment operator
    EEDetId& operator=(const DetId& id);

    /// get the subdetector
    EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }
    /// get the z-side of the crystal (1/-1)
    int zside() const { return (id_&0x4000)?(1):(-1); }
    /// get the crystal ix
    int ix() const { return (id_>>7)&0x7F; }
    /// get the crystal iy
    int iy() const { return id_&0x7F; }  
    /// get a compact index for arrays
    int hashedIndex() const;

    static const int IX_MIN=1;
    static const int IY_MIN=1;
    static const int IX_MAX=100;
    static const int IY_MAX=100;
  };

}

std::ostream& operator<<(std::ostream& s,const cms::EEDetId& id);

#endif
