/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *	Hubert Niewiadomski
 *	Jan Kašpar (jan.kaspar@gmail.com) 
 *
 ****************************************************************************/


#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

const uint32_t TotemRPDetId::startPlaneBit = 15, TotemRPDetId::maskPlane = 0xF, TotemRPDetId::maxPlane = 9, TotemRPDetId::lowMaskPlane = 0x7FFF;
const uint32_t TotemRPDetId::startChipBit = 13, TotemRPDetId::maskChip = 0x3, TotemRPDetId::maxChip = 3, TotemRPDetId::lowMaskChip = 0x1FFF;

const string TotemRPDetId::planeNames[] = { "01", "02", "03", "04", "05", "06", "07", "08", "09", "10" };
const string TotemRPDetId::chipNames[] = { "1", "2", "3", "4" };

//----------------------------------------------------------------------------------------------------

TotemRPDetId::TotemRPDetId(uint32_t id) : CTPPSDetId(id)
{
  bool inputOK = (det() == DetId::VeryForward && subdetId() == sdTrackingStrip);

  if (!inputOK)
  {
    throw cms::Exception("InvalidDetId") << "TotemRPDetId ctor:"
      << " det: " << det()
      << " subdet: " << subdetId()
      << " is not a valid TotemRP id.";  
  }
}

//----------------------------------------------------------------------------------------------------

TotemRPDetId::TotemRPDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot, uint32_t Plane, uint32_t Chip) :       
  CTPPSDetId(sdTrackingStrip, Arm, Station, RomanPot)
{
  if (Arm > maxArm || Station > maxStation || RomanPot > maxRP || Plane > maxPlane || Chip > maxChip)
  {
      throw cms::Exception("InvalidDetId") << "TotemRPDetId ctor:" 
             << " Invalid parameters:" 
             << " arm=" << Arm
             << " station=" << Station
             << " rp=" << RomanPot
             << " plane=" << Plane
             << " chip=" << Chip
             << std::endl;
  }

  uint32_t ok=0xfe000000;
  id_ &= ok;

  id_ |= ((Arm & maskArm) << startArmBit);
  id_ |= ((Station & maskStation) << startStationBit);
  id_ |= ((RomanPot & maskRP) << startRPBit);
  id_ |= ((Plane & maskPlane) << startPlaneBit);
  id_ |= ((Chip & maskChip) << startChipBit);
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const TotemRPDetId& id)
{
  os << "arm=" << id.arm()
     << " station=" << id.station()
     << " rp=" << id.rp()
     << " plane=" << id.plane()
     << " chip=" << id.chip();

  return os;
}
