/****************************************************************************
 * Author: Seyed Mohsen Etesami
 *  September 2016 
 ****************************************************************************/

#ifndef DataFormats_CTPPSDetId_CTPPSDiamondDetId
#define DataFormats_CTPPSDetId_CTPPSDiamondDetId

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>
#include <string>

/**
 *\brief Detector ID class for CTPPS Timing Diamond detectors.
 * Bits [19:31] : Assigend in CTPPSDetId Calss
 * Bits [17:18] : 2 bits for diamond plane 0,1,2,3 
 * Bits [12:16] : 5 bits for Diamond  det numbers 1,2,3,..16
 * Bits [0:11]  : unspecified yet
 **/

class CTPPSDiamondDetId : public CTPPSDetId
{  
 public:
  /// Construct from a raw id
  explicit CTPPSDiamondDetId(uint32_t id);

 CTPPSDiamondDetId(const CTPPSDetId &id) : CTPPSDetId(id)
  {
  }
  
  /// Construct from hierarchy indeces.
  CTPPSDiamondDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot=0, uint32_t Plane=0, uint32_t Det=0);

  static const uint32_t startPlaneBit = 17, maskPlane = 0x3, maxPlane = 3, lowMaskPlane = 0x1FFFF;
  static const uint32_t startDetBit = 12, maskDet = 0x1F, maxDet = 12, lowMaskDet = 0xFFF;


  /// returns true if the raw ID is a PPS-timing one
  static bool Check(unsigned int raw)
  {
    return (((raw >>DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
	    ((raw >> DetId::kSubdetOffset) & 0x7) == sdTimingDiamond);
  }    
  //-------------------- getting and setting methods --------------------
     
  uint32_t plane() const
  {
    return ((id_>>startPlaneBit) & maskPlane);
  }

  void setPlane(uint32_t det)
  {
    id_ &= ~(maskPlane << startPlaneBit);
    id_ |= ((det & maskPlane) << startPlaneBit);
  }

  uint32_t det() const
  {
    return ((id_>>startDetBit) & maskDet);
  }

  void setDet(uint32_t det)
  {
    id_ &= ~(maskDet << startDetBit);
    id_ |= ((det & maskDet) << startDetBit);
  }

  //-------------------- id getters for higher-level objects --------------------

  CTPPSDiamondDetId getPlaneId() const
  {
    return CTPPSDiamondDetId( rawId() & (~lowMaskPlane) );
  }


   
    /// returns official name of a plane characterized by ''id''; if ''full'' is true, name of RP is prefixed
    std::string planeName(NameFlag flag = nFull);
  
    /// returns official name of a chip characterized by ''id''; if ''full'' is true, name of plane is prefixed
    std::string channelName(NameFlag flag = nFull);
    
    





};

std::ostream& operator<<(std::ostream& os, const CTPPSDiamondDetId& id);

#endif 
