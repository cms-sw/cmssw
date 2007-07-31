#ifndef DATAFORMATS_HCALDETID_HCALTRIGTOWERDETID_H
#define DATAFORMATS_HCALDETID_HCALTRIGTOWERDETID_H 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/** \class HcalTrigTowerDetId
    
Cell id for an Calo Trigger tower

   $Date: 2006/02/16 17:01:57 $
   $Revision: 1.7 $
   \author J. Mans - Minnesota
*/
class HcalTrigTowerDetId : public DetId {
public:
   /** Constructor of a null id */
  HcalTrigTowerDetId();
  /** Constructor from a raw value */
  HcalTrigTowerDetId(uint32_t rawid);  
  /** \brief Constructor from signed ieta, iphi
  */
  HcalTrigTowerDetId(int ieta, int iphi);

  /** Constructor from a generic cell id */
  HcalTrigTowerDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalTrigTowerDetId& operator=(const DetId& id);

  /// get the subdetector
  HcalSubdetector subdet() const { return (HcalSubdetector)(subdetId()); }
  /// get the z-side of the tower (1/-1)
  int zside() const { return (id_&0x2000)?(1):(-1); }
  /// get the absolute value of the tower ieta
  int ietaAbs() const { return (id_>>7)&0x3f; }
  /// get the tower ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the tower iphi
  int iphi() const { return id_&0x7F; }

  static const HcalTrigTowerDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HcalTrigTowerDetId& id);


#endif
