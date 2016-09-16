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
  bool inputOK = ((id >> DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
    ((id >> DetId::kSubdetOffset) & 0x7) == totem_rp_subdet_id;

  if (!inputOK)
  {
    throw cms::Exception("InvalidDetId") << "TotemRPDetId ctor:"
      << " det: " << det()
      << " subdet: " << subdetId()
      << " is not a valid Totem RP id";  
  }
}

//----------------------------------------------------------------------------------------------------

TotemRPDetId::TotemRPDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot, uint32_t Plane, uint32_t Chip) :       
  DetId(DetId::VeryForward, totem_rp_subdet_id)
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

string TotemRPDetId::subDetectorName(NameFlag flag) const
{
  string name;
  if (flag == nFull) name = "ctpps_tr_strip";
  if (flag == nPath) name = "CTPPS/TrackingStrip";

  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::armName(NameFlag flag) const
{
  string name;
  if (flag == nFull) name = subDetectorName(flag) + "_";
  if (flag == nPath) name = subDetectorName(flag) + "/sector ";

  uint32_t id = arm();
  if (id == 0) name += "45";
  if (id == 1) name += "56";

  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::stationName(NameFlag flag) const
{
  string name;
  if (flag == nFull) name = armName(flag) + "_";
  if (flag == nPath) name = armName(flag) + "/station ";

  uint32_t id = station();
  if (id == 0) name += "210";
  if (id == 1) name += "220cyl";
  if (id == 2) name += "220";

  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::rpName(NameFlag flag) const
{
  string name; 
  if (flag == nFull) name = stationName(flag) + "_";
  if (flag == nPath) name = stationName(flag) + "/";

  uint32_t id = rp();
  if (id == 0) name += "nr_tp";
  if (id == 1) name += "nr_bt";
  if (id == 2) name += "nr_hr";
  if (id == 3) name += "fr_hr";
  if (id == 4) name += "fr_tp";
  if (id == 5) name += "fr_bt";

  return name;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::planeName(NameFlag flag) const
{
  string name;
  if (flag == nFull) name = rpName(flag) + "_";
  if (flag == nPath) name = rpName(flag) + "/plane ";

  uint32_t id = plane();
  char buf[10];
  sprintf(buf, "%02u", id + 1);

  return name + buf;
}

//----------------------------------------------------------------------------------------------------

string TotemRPDetId::chipName(NameFlag flag) const
{
  string name;
  if (flag == nFull) name = planeName(flag) + "_";
  if (flag == nPath) name = planeName(flag) + "/chip ";

  uint32_t id = chip();
  char buf[10];
  sprintf(buf, "%u", (id % 10) + 1);

  return name + buf;
}
