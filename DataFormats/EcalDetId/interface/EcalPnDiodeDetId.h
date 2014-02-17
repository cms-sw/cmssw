#ifndef ECALDETID_ECALPNDIODEDETID_H
#define ECALDETID_ECALPNDIODEDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EcalPnDiodeDetId
    
   DetId for an Calo Trigger tower
   Packing:

   [31:28] Global det == ECAL
   [27:25] ECAL det == EcalLaserPnDiode
   [24:12] Not Used
   [10]    SubDetectorId: EB (1) ,EE (2)
   [9:4]   DCCId (relative to SubDetector. In barrel it is the SupermoduleId from 1-36)
   [3:0]   PnId (In barrel from 1-10 according CMS IN-2005/021)

   $Id: EcalPnDiodeDetId.h,v 1.4 2007/10/24 20:18:36 ferriff Exp $
*/


class EcalPnDiodeDetId : public DetId {
 public:
  /** Constructor of a null id */
  EcalPnDiodeDetId();
  /** Constructor from a raw value */
  EcalPnDiodeDetId(uint32_t rawid);  
  /** \brief Constructor from signed EcalSubDetectorId, DCCId, PnId
   */
  EcalPnDiodeDetId(int EcalSubDetectorId, int DCCId, int PnId);
  /** Constructor from a generic cell id */
  EcalPnDiodeDetId(const DetId& id);
  /** Assignment from a generic cell id */
  EcalPnDiodeDetId& operator=(const DetId& id);

  static const int MAX_DCCID = 54;
  static const int MIN_DCCID = 1;
  static const int MAX_PNID = 15; 
  static const int MIN_PNID = 1;

  /// get EcalSubDetectorId
  int iEcalSubDetectorId() const { return (id_ & 0x800 ) ? (EcalEndcap):(EcalBarrel); }
  /// get the DCCId
  int iDCCId() const { return (id_>>4) & 0x7F; }
  /// get the PnId
  int iPnId() const { return id_&0xF; }
  /// get a compact index for arrays [TODO: NEEDS WORK]
  int hashedIndex() const;

};

std::ostream& operator<<(std::ostream&,const EcalPnDiodeDetId& id);

#endif
