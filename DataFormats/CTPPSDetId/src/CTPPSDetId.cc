/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *	Hubert Niewiadomski
 *	Jan Kašpar (jan.kaspar@gmail.com) 
 *
 ****************************************************************************/


#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

CTPPSDetId::CTPPSDetId(uint32_t id) : DetId(id)
{
  bool inputOK = (det() == DetId::VeryForward);

  if (!inputOK)
  {
    throw cms::Exception("InvalidDetId") << "CTPPSDetId ctor:"
      << " det: " << det()
      << " subdet: " << subdetId()
      << " is not a valid CTPPS id.";  
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSDetId::CTPPSDetId(uint32_t SubDet, uint32_t Arm, uint32_t Station, uint32_t RomanPot) :       
  DetId(DetId::VeryForward, SubDet)
{
  if (SubDet != sdTrackingStrip && SubDet != sdTrackingPixel && SubDet != sdTimingDiamond && SubDet != sdTimingFastSilicon)
  {
    throw cms::Exception("InvalidDetId") << "CTPPSDetId ctor: invalid sub-detector " << SubDet << ".";
  }

  if (Arm > maxArm || Station > maxStation || RomanPot > maxRP)
  {
    throw cms::Exception("InvalidDetId") << "CTPPSDetId ctor:" 
           << " Invalid parameters:" 
           << " arm=" << Arm
           << " station=" << Station
           << " rp=" << RomanPot
           << std::endl;
  }

  uint32_t ok=0xfe000000;
  id_ &= ok;

  id_ |= ((Arm & maskArm) << startArmBit);
  id_ |= ((Station & maskStation) << startStationBit);
  id_ |= ((RomanPot & maskRP) << startRPBit);
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const CTPPSDetId& id)
{
  os
    << "subDet=" << id.subdetId()
    << " arm=" << id.arm()
    << " station=" << id.station()
    << " rp=" << id.rp();

  return os;
}

//----------------------------------------------------------------------------------------------------

string CTPPSDetId::subDetectorName(NameFlag flag) const
{
  string name;

  if (flag == nFull)
  {
    if (subdetId() == sdTrackingStrip) name = "ctpps_tr_strip";
    if (subdetId() == sdTrackingPixel) name = "ctpps_tr_pixel";
    if (subdetId() == sdTimingDiamond) name = "ctpps_ti_diamond";
    if (subdetId() == sdTimingFastSilicon) name = "ctpps_ti_fastsilicon";
  }

  if (flag == nPath)
  {
    if (subdetId() == sdTrackingStrip) name = "CTPPS/TrackingStrip";
    if (subdetId() == sdTrackingPixel) name = "CTPPS/TrackingPixel";
    if (subdetId() == sdTimingDiamond) name = "CTPPS/TimingDiamond";
    if (subdetId() == sdTimingFastSilicon) name = "CTPPS/TimingFastSilicon";
  }

  return name;
}

//----------------------------------------------------------------------------------------------------

string CTPPSDetId::armName(NameFlag flag) const
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

string CTPPSDetId::stationName(NameFlag flag) const
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

string CTPPSDetId::rpName(NameFlag flag) const
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
