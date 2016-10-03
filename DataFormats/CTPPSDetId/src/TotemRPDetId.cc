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

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::planeName(NameFlag flag) const
{
  string name;

  switch (flag)
  {
    case nShort: name = ""; break;
    case nFull: name = rpName(flag) + "_"; break;
    case nPath: name = rpName(flag) + "/plane "; break;
  }

  uint32_t id = plane();
  char buf[10];
  sprintf(buf, "%02u", id + 1);

  return name + buf;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::chipName(NameFlag flag) const
{
  string name;

  switch (flag)
  {
    case nShort: name = ""; break;
    case nFull: name = planeName(flag) + "_"; break;
    case nPath: name = planeName(flag) + "/chip "; break;
  }

  uint32_t id = chip();
  char buf[10];
  sprintf(buf, "%u", (id % 10) + 1);

  return name + buf;
}
