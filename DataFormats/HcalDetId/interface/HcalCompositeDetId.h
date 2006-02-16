#ifndef DATAFORMATS_HCALDETID_HCALCOMPOSITEDETID_H
#define DATAFORMATS_HCALDETID_HCALCOMPOSITEDETID_H 1

#include <ostream>
#include <boost/cstdint.hpp>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/** \class HcalCompositeDetId
    
Detector id class which serves as base class for composite det ids which use the 
standard eta/phi grid of HCAL but may have additional data.

Packing:

[31:28] Global det == HCAL
[27:25] HCAL det == Composite
[24:21] Composite type
[20:14] Composite extra data (7 bits)
[13]    Ieta sign bit
[12:7]  Ieta (absolute)
[6:0]   Iphi

$Date: 2005/10/06 00:38:43 $
$Revision: 1.3 $
\author J. Mans - Minnesota
*/
class HcalCompositeDetId : public DetId {
public:
  /** Constructor from a generic cell id */
  HcalCompositeDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalCompositeDetId& operator=(const DetId& id);
  
  enum CompositeType { HFCombinedHitType=1 };
  
  /// get the z-side of the tower (1/-1)
  int zside() const { return (id_&0x2000)?(1):(-1); }
  /// get the absolute value of the tower ieta
  int ietaAbs() const { return (id_>>7)&0x3f; }
  /// get the tower ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the tower iphi
  int iphi() const { return id_&0x7F; }
  
  /// get the composite type
  CompositeType getCompositeType() const { return CompositeType((id_>>21)&0xF); }
  /// get the composite data field (7 bits)
  int getCompositeData() const { return ((id_>>14)&0x7F); }

  static const HcalCompositeDetId Undefined;

protected:
  /** Constructor of a null id */
  HcalCompositeDetId();
  /** Constructor from a raw value */
  explicit HcalCompositeDetId(uint32_t rawid);  
  /** \brief Constructor from signed ieta, iphi plus composite type and composite data */
  HcalCompositeDetId(CompositeType composite_type, int composite_data, int ieta, int iphi);
};

#endif
