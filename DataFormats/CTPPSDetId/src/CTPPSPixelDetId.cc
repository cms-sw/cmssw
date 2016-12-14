/*
  Author: F.Ferro INFN Genova
  October 2016
*/

#include <DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

// VeryForward =7, Tracker = 4

const uint32_t CTPPSPixelDetId::startPlaneBit = 16, CTPPSPixelDetId::maskPlane = 0x7, CTPPSPixelDetId::maxPlane = 5; 

CTPPSPixelDetId::CTPPSPixelDetId(uint32_t id):CTPPSDetId(id) {

  if (! check(id))
    {
      throw cms::Exception("InvalidDetId") << "CTPPSPixelDetId ctor:"
					   << " det: " << det()
					   << " subdet: " << subdetId()
					   << " is not a valid CTPPS Pixel id";  
    }
}



CTPPSPixelDetId::CTPPSPixelDetId(unsigned int Arm, unsigned int Station, unsigned int RP, unsigned int Plane):	      
  CTPPSDetId(sdTrackingPixel,Arm,Station,RP)
{
  this->init(Arm,Station,RP,Plane);
}

void CTPPSPixelDetId::init(unsigned int Arm, unsigned int Station, unsigned int RP, unsigned int Plane)
{
  if ( 
      Arm > maxArm || Station > maxStation || RP > maxRP || Plane > maxPlane
       ) {
    throw cms::Exception("InvalidDetId") << "CTPPSPixelDetId ctor:" 
					 << " Invalid parameterss: " 
					 << " Arm "<<Arm
					 << " Station "<<Station
					 << " RP "<<RP
					 << " Plane "<<Plane
					 << std::endl;
  }

  uint32_t ok=0xfe000000;
  id_ &= ok;

  id_ |= ((Arm & maskArm) << startArmBit);
  id_ |= ((Station & maskStation) << startStationBit);
  id_ |= ((RP & maskRP) << startRPBit);
  id_ |= ((Plane & maskPlane) << startPlaneBit);

}

std::ostream& operator<<( std::ostream& os, const CTPPSPixelDetId& id ){
  os <<  " Arm "<<id.arm()
     << " Station " << id.station()
     << " RP "<<id.rp()
     << " Plane "<<id.plane();

  return os;
}


