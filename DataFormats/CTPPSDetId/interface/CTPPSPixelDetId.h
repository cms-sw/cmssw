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
      
  // CTPPSPixelDetId();

  /// Construct from a packed id.
  explicit CTPPSPixelDetId(uint32_t id);
  
CTPPSPixelDetId(const CTPPSDetId &id) : CTPPSDetId(id)
  {
  }
  /// Construct from fully qualified identifier.
  CTPPSPixelDetId(unsigned int Arm, 
		    unsigned int Station,
		    unsigned int RP,
		    unsigned int Plane);
    


 /// Bit 24 = Arm: 0=z>0 1=z<0
  /// Bits [22:23] Station (0 = 210 or ex 147) 
  /// Bits [19:21] RP
  /// Bits [16:18] Si Plane
/// 

 
  static bool check(unsigned int raw)
  {
    return (((raw >>DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
	    ((raw >> DetId::kSubdetOffset) & 0x7) == sdTrackingPixel);
  }    


 int Arm() const{
    return int((id_>>startArmBit) & 0X1);
  }
  inline int Station() const
    {
      return int((id_>>startStationBit) & 0x3);
    }
  int RP() const{
    return int((id_>>startRPBit) & 0X7);
  }

  int Plane() const{
    return int((id_>>startPlaneBit) & 0X7);
  }

  void set(unsigned int a, unsigned int b, unsigned int c,unsigned int d ){
//    unsigned int d=0;
    this->init(a,b,c,d);
  }


  static const int startArmBit = 24;
  static const int startStationBit = 22;
  static const int startRPBit = 19;
  static const int startPlaneBit = 16;
 
 private:
  void init(unsigned int Arm, unsigned int Station,unsigned int RP, unsigned int Plane); 


}; // CTPPSPixelDetId

std::ostream& operator<<( std::ostream& os, const CTPPSPixelDetId& id );

#endif


