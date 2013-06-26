#ifndef HcalOtherDetId_h_included
#define HcalOtherDetId_h_included 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/** \class HcalOtherDetId
    
Detector id class which serves as base class for other HCAL-related 
det ids  (cascaded from top bit).

Packing:

[31:28] Global det == HCAL
[27:25] HCAL subdet == Other
[24:20] Other subdet id
[19:0]  Available for use

$Date: 2007/07/31 15:20:09 $
$Revision: 1.2 $
\author J. Mans - Minnesota
*/
class HcalOtherDetId : public DetId {
public:
  /** Constructor from a generic cell id */
  HcalOtherDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalOtherDetId& operator=(const DetId& id);
  
  /// get the category
  HcalOtherSubdetector subdet() const { return HcalOtherSubdetector((id_>>20)&0x1F); }

protected:
  /** Constructor of a null id */
  HcalOtherDetId();
  /** Constructor from a raw value */
  explicit HcalOtherDetId(uint32_t rawid);  
  /** \brief Constructor from signed ieta, iphi plus composite type and composite data */
  HcalOtherDetId(HcalOtherSubdetector subdet);
};

#endif // HcalOtherDetId_h_included
