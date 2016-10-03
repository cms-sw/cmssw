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
    switch (subdetId())
    {
      case sdTrackingStrip: name = "ctpps_tr_strip"; break;
      case sdTrackingPixel: name = "ctpps_tr_pixel"; break;
      case sdTimingDiamond: name = "ctpps_ti_diamond"; break;
      case sdTimingFastSilicon: name = "ctpps_ti_fastsilicon"; break;
    }
  }

  if (flag == nPath)
  {
    switch (subdetId())
    {
      case sdTrackingStrip: name = "CTPPS/TrackingStrip"; break;
      case sdTrackingPixel: name = "CTPPS/TrackingPixel"; break;
      case sdTimingDiamond: name = "CTPPS/TimingDiamond"; break;
      case sdTimingFastSilicon: name = "CTPPS/TimingFastSilicon"; break;
    }
  }

  return name;
}

//----------------------------------------------------------------------------------------------------

string CTPPSDetId::armName(NameFlag flag) const
{
  string name;

  switch (flag)
  {
    case nShort: name = ""; break;
    case nFull: name = subDetectorName(flag) + "_"; break;
    case nPath: name = subDetectorName(flag) + "/sector "; break;
  }

  switch (arm())
  {
    case 0: name += "45"; break;
    case 1: name += "56"; break;
  }

  return name;
}

//----------------------------------------------------------------------------------------------------

string CTPPSDetId::stationName(NameFlag flag) const
{
  string name;

  switch (flag)
  {
    case nShort: name = ""; break;
    case nFull: name = armName(flag) + "_"; break;
    case nPath: name = armName(flag) + "/station "; break;
  }

  switch (station())
  {
    case 0: name += "210"; break;
    case 1: name += "220cyl"; break;
    case 2: name += "220"; break;
  }

  return name;
}

//----------------------------------------------------------------------------------------------------

string CTPPSDetId::rpName(NameFlag flag) const
{
  string name;

  switch (flag)
  {
    case nShort: name = ""; break;
    case nFull: name = stationName(flag) + "_"; break;
    case nPath: name = stationName(flag) + "/"; break;
  }

  uint32_t id = rp();
  switch (id)
  {
    case 0: name += "nr_tp"; break;
    case 1: name += "nr_bt"; break;
    case 2: name += "nr_hr"; break;
    case 3: name += "fr_hr"; break;
    case 4: name += "fr_tp"; break;
    case 5: name += "fr_bt"; break;
    case 6: name += "cyl_hr"; break;
  }

  return name;
}
