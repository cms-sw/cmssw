
/*
Author: F.Ferro INFN Genova
October 2016

 */

#include <DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h>
#include <FWCore/Utilities/interface/Exception.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// VeryForward =7, Tracker = 4

CTPPSPixelDetId::CTPPSPixelDetId(uint32_t id):CTPPSDetId(id) {
  //  std::cout<<" constructor of the CTPPSPixelDetId" <<std::endl;
 if (! check(id))
    {
      throw cms::Exception("InvalidDetId") << "CTPPSPixelDetId ctor:"
					   << " det: " << det()
					   << " subdet: " << subdetId()
					   << " is not a valid CTPPS Tracker id";  
    }
}



CTPPSPixelDetId::CTPPSPixelDetId(unsigned int Arm, unsigned int Station, unsigned int RP, unsigned int Plane):	      
  CTPPSDetId(sdTrackingPixel,Arm,Station,RP)
{
//  unsigned int d=0;
  this->init(Arm,Station,RP,Plane);
}


void
CTPPSPixelDetId::init(unsigned int Arm, unsigned int Station, unsigned int RP, unsigned int Plane)
{
  if ( 
      (Arm != 0 && Arm !=1) || Station != 0 ||
      Plane > 5 ||
      RP > 3 || RP < 2
      ) {
    throw cms::Exception("InvalidDetId") << "CTPPSPixelDetId ctor:" 
					 << " Invalid parameterss: " 
					 << " Arm "<<Arm
					 << " RP "<<RP
					 << " Plane "<<Plane
					 << std::endl;
  }

  uint32_t ok=0xfe000000;
  id_ &= ok;
/*
  id_ |=  Plane   << 20    | 
    RP    << 23    |
    Arm  << 24 ;
*/
  id_ |= ((Arm&0x1) << startArmBit);
  id_ |= ((Station&0x3) << startStationBit);
  id_ |= ((RP&0x7) << startRPBit);
  id_ |= ((Plane&0x7) << startPlaneBit);
//  std::cout  << id_ << " " << Arm << " " << RP << " "<< Plane << std::endl; 

}

std::ostream& operator<<( std::ostream& os, const CTPPSPixelDetId& id ){
  os <<  " Arm "<<id.Arm()
     << " RP "<<id.RP()
     << " Plane "<<id.Plane();

  return os;
}


