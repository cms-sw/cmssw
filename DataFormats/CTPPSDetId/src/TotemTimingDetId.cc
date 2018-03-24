/****************************************************************************
 * Author: Nicola Minafra
 *  March 2018
 ****************************************************************************/


#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

const uint32_t TotemTimingDetId::startPlaneBit = 17, TotemTimingDetId::maskPlane = 0x3, TotemTimingDetId::maxPlane = 3, TotemTimingDetId::lowMaskPlane = 0x1FFFF;
const uint32_t TotemTimingDetId::startDetBit = 12, TotemTimingDetId::maskChannel = 0x1F, TotemTimingDetId::maxChannel = 31, TotemTimingDetId::lowMaskChannel = 0xFFF;

//----------------------------------------------------------------------------------------------------

TotemTimingDetId::TotemTimingDetId(uint32_t id) : CTPPSDetId(id)
{
  if (! check(id))
    {
      throw cms::Exception("InvalidDetId") << "TotemTimingDetId ctor:"
					   << " channel: " << channel()
					   << " subdet: " << subdetId()
					   << " is not a valid Totem Timing id";  
    }
}

//----------------------------------------------------------------------------------------------------

TotemTimingDetId::TotemTimingDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot, uint32_t Plane, uint32_t Channel) :       
  CTPPSDetId(sdTimingFastSilicon, Arm, Station, RomanPot)
{
  if (Arm > maxArm || Station > maxStation || RomanPot > maxRP || Plane > maxPlane || Channel > maxChannel)
    {
      throw cms::Exception("InvalidDetId") << "TotemTimingDetId ctor:" 
					   << " Invalid parameters:" 
					   << " arm=" << Arm
					   << " station=" << Station
					   << " rp=" << RomanPot
					   << " plane=" << Plane
					   << " detector=" << Channel
					   << std::endl;
    }

  uint32_t ok=0xfe000000;
  id_ &= ok;

  id_ |= ((Arm & maskArm) << startArmBit);
  id_ |= ((Station & maskStation) << startStationBit);
  id_ |= ((RomanPot & maskRP) << startRPBit);
  id_ |= ((Plane & maskPlane) << startPlaneBit);
  id_ |= ((Channel & maskChannel) << startDetBit);
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const TotemTimingDetId& id)
{
  os << "arm=" << id.arm()
     << " station=" << id.station()
     << " rp=" << id.rp()
     << " plane=" << id.plane()
     << " Detector=" << id.channel();

  return os;
}
