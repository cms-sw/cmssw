#ifndef DataFormats_CTPPSPixelDetId_h
#define DataFormats_CTPPSPixelDetId_h

/*
  Author: F.Ferro INFN Genova
  October 2016

*/

#include <DataFormats/CTPPSDetId/interface/CTPPSDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iosfwd>
#include <iostream>


class CTPPSPixelDetId :public CTPPSDetId {
  
public:

  /// Construct from a packed id.
  explicit CTPPSPixelDetId(uint32_t id);
  
CTPPSPixelDetId(const CTPPSDetId &id) : CTPPSDetId(id)
  {
  }
  /// Construct from fully qualified identifier.
  CTPPSPixelDetId(uint32_t Arm, uint32_t Station, uint32_t RP=0, uint32_t Plane=0);

  /// Bit 24 = Arm: 0=z>0 1=z<0
  /// Bits [22:23] Station (0 = 210 or ex 147) 
  /// Bits [19:21] RP
  /// Bits [16:18] Si Plane

  static const uint32_t startPlaneBit, maskPlane, maxPlane;

  static bool check(unsigned int raw)
  {
    return (((raw >>DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
	    ((raw >> DetId::kSubdetOffset) & 0x7) == sdTrackingPixel);
  }    

  uint32_t plane() const{
    return int((id_>>startPlaneBit) & maskPlane);
  }

  void set(uint32_t a, uint32_t b, uint32_t c,uint32_t d ){
    this->init(a,b,c,d);
  }
 
  void setPlane(uint32_t pl)
  {
    id_ &= ~(maskPlane << startPlaneBit);
    id_ |= ((pl & maskPlane) << startPlaneBit);
  }


private:
  void init(uint32_t Arm, uint32_t Station,uint32_t RP, uint32_t Plane); 


}; // CTPPSPixelDetId

std::ostream& operator<<( std::ostream& os, const CTPPSPixelDetId& id );

#endif


