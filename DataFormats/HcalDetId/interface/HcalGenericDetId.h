#ifndef HCALGENERICDETID_H
#define HCALGENERICDETID_H

/** \class HcalGenericDetId
    \author F.Ratnikov, UMd
   Generic HCAL detector ID suitable for all Hcal subdetectors
   $Id: HcalGenericDetId.h,v 1.7 2008/07/15 20:14:11 rofierzy Exp $
   
   R.Ofierzynski, 22.02.2008, added hashedId
*/

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HcalGenericDetId : public DetId {
 public:
  enum HcalGenericSubdetector {HcalGenEmpty=0, HcalGenBarrel=1, HcalGenEndcap=2, HcalGenOuter=3, HcalGenForward=4, 
			       HcalGenTriggerTower=5, HcalGenZDC=8, HcalGenCalibration=9, HcalGenCastor=10, HcalGenUnknown=99};
  HcalGenericDetId () : DetId () {}
  HcalGenericDetId (uint32_t rawid) : DetId (rawid) {}
  HcalGenericDetId (const DetId& id) : DetId (id) {}
  HcalOtherSubdetector otherSubdet () const;
  HcalGenericSubdetector genericSubdet () const;
  bool isHcalDetId () const;
  bool isHcalCalibDetId () const;
  bool isHcalTrigTowerDetId () const;
  bool isHcalZDCDetId () const;
  bool isHcalCastorDetId () const;

  // delivers hashed id for this cell
  // hashed id's are determined separately for each subdetector !
  int hashedId(bool h2mode_ = false) const;

  enum hashlimits{
    HBhalf = 1296,
    HEhalf = 1296,
    HEhalfh2mode = 4032,
    HOhalf = 1080,
    HFhalf = 864,
    HThalf = 2088,
    ZDChalf = 11,
    CASTORhalf = 224,
    CALIBhalf = 693
  };
};

std::ostream& operator<<(std::ostream&,const HcalGenericDetId& id);


#endif
