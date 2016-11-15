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

TotemRPDetId::TotemRPDetId(uint32_t id) : DetId(id)
{
  if (! check(id))
  {
    throw cms::Exception("InvalidDetId") << "TotemRPDetId ctor:"
      << " det: " << det()
      << " subdet: " << subdetId()
      << " is not a valid Totem RP id";  
  }
}

//----------------------------------------------------------------------------------------------------

TotemRPDetId::TotemRPDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot, uint32_t Detector, uint32_t Chip) :       
  DetId(DetId::VeryForward, totem_rp_subdet_id)
{
  if (Arm > maxArm || Station > maxStation || RomanPot > maxRP || Detector > maxDet || Chip > maxChip)
  {
      throw cms::Exception("InvalidDetId") << "TotemRPDetId ctor:" 
             << " Invalid parameters: " 
             << " Arm "<<Arm
             << " Station "<<Station
             << " RomanPot "<<RomanPot
             << " Detector "<<Detector
             << std::endl;
  }

  uint32_t ok=0xfe000000;
  id_ &= ok;

  id_ |= ((Arm & maskArm) << startArmBit);
  id_ |= ((Station & maskStation) << startStationBit);
  id_ |= ((RomanPot & maskRP) << startRPBit);
  id_ |= ((Detector & maskDet) << startDetBit);
  id_ |= ((Chip & maskChip) << startChipBit);
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const TotemRPDetId& id)
{
  os << " Arm " << id.arm()
     << " Station " << id.station()
     << " RomanPot " << id.romanPot()
     << " Detector " << id.detector()
     << " Chip " << id.chip();

  return os;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::systemName(NameFlag flag)
{
  string name;
  if (flag == nFull) name = "rp";
  if (flag == nPath) name = "RP";

  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::armName(uint32_t id, NameFlag flag)
{
  string name;
  if (flag == nFull) name = systemName(flag) + "_";
  if (flag == nPath) name = systemName(flag) + "/sector ";

  if (id == 0) name += "45";
  if (id == 1) name += "56";

  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::stationName(uint32_t id, NameFlag flag)
{
  string name;
  if (flag == nFull) name = armName(id / 10, flag) + "_";
  if (flag == nPath) name = armName(id / 10, flag) + "/station ";

  if ((id % 10) == 0) name += "210";
  if ((id % 10) == 2) name += "220";

  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::rpName(uint32_t id, NameFlag flag)
{
  string name; 
  if (flag == nFull) name = stationName(id / 10, flag) + "_";
  if (flag == nPath) name = stationName(id / 10, flag) + "/";

  if ((id % 10) == 0) name += "nr_tp";
  if ((id % 10) == 1) name += "nr_bt";
  if ((id % 10) == 2) name += "nr_hr";
  if ((id % 10) == 3) name += "fr_hr";
  if ((id % 10) == 4) name += "fr_tp";
  if ((id % 10) == 5) name += "fr_bt";
  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::planeName(uint32_t id, NameFlag flag)
{
  string name;
  if (flag == nFull) name = rpName(id / 10, flag) + "_";
  if (flag == nPath) name = rpName(id / 10, flag) + "/plane ";

  char buf[10];
  sprintf(buf, "%02u", (id % 10) + 1);

  return name + buf;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::chipName(uint32_t id, NameFlag flag)
{
  string name;
  if (flag == nFull) name = planeName(id / 10, flag) + "_";
  if (flag == nPath) name = planeName(id / 10, flag) + "/chip ";

  char buf[10];
  sprintf(buf, "%u", (id % 10) + 1);

  return name + buf;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::stripName(uint32_t id, unsigned char strip, NameFlag flag)
{
  string name;
  if (flag == nFull) name = chipName(id, flag) + "_";
  if (flag == nPath) name = chipName(id, flag) + "/strip";

  char buf[10];
  sprintf(buf, "%u", strip);

  return name + buf;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::officialName(ElementLevel level, uint32_t id, NameFlag flag, unsigned char strip)
{
  switch (level)
  {
    case lSystem: return systemName(flag);
    case lArm: return armName(id, flag);
    case lStation: return stationName(id, flag);
    case lRP: return rpName(id, flag);
    case lPlane: return planeName(id, flag);
    case lChip: return chipName(id, flag);
    case lStrip: return stripName(id, flag);
    default: return "";
  }
}
