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

const uint32_t CTPPSDetId::startArmBit = 24, CTPPSDetId::maskArm = 0x1, CTPPSDetId::maxArm = 1,
               CTPPSDetId::lowMaskArm = 0xFFFFFF;
const uint32_t CTPPSDetId::startStationBit = 22, CTPPSDetId::maskStation = 0x3, CTPPSDetId::maxStation = 2,
               CTPPSDetId::lowMaskStation = 0x3FFFFF;
const uint32_t CTPPSDetId::startRPBit = 19, CTPPSDetId::maskRP = 0x7, CTPPSDetId::maxRP = 6,
               CTPPSDetId::lowMaskRP = 0x7FFFF;

const string CTPPSDetId::subDetectorNames[] = {
    "", "", "", "ctpps_tr_strip", "ctpps_tr_pixel", "ctpps_ti_diamond", "ctpps_ti_fastsilicon", "totem_t2"};
const string CTPPSDetId::subDetectorPaths[] = {"",
                                               "",
                                               "",
                                               "CTPPS/TrackingStrip",
                                               "CTPPS/TrackingPixel",
                                               "CTPPS/TimingDiamond",
                                               "CTPPS/TimingFastSilicon",
                                               "TotemT2"};
const string CTPPSDetId::armNames[] = {"45", "56"};
const string CTPPSDetId::stationNames[] = {"210", "220cyl", "220"};
const string CTPPSDetId::rpNames[] = {"nr_tp", "nr_bt", "nr_hr", "fr_hr", "fr_tp", "fr_bt", "cyl_hr"};

//----------------------------------------------------------------------------------------------------

CTPPSDetId::CTPPSDetId(uint32_t id) : DetId(id) {
  bool inputOK = (det() == DetId::VeryForward);

  if (!inputOK) {
    throw cms::Exception("InvalidDetId") << "CTPPSDetId ctor:"
                                         << " det: " << det() << " subdet: " << subdetId()
                                         << " is not a valid CTPPS id.";
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSDetId::CTPPSDetId(uint32_t SubDet, uint32_t Arm, uint32_t Station, uint32_t RomanPot)
    : DetId(DetId::VeryForward, SubDet) {
  if (SubDet != sdTrackingStrip && SubDet != sdTrackingPixel && SubDet != sdTimingDiamond &&
      SubDet != sdTimingFastSilicon && SubDet != sdTotemT2) {
    throw cms::Exception("InvalidDetId") << "CTPPSDetId ctor: invalid sub-detector " << SubDet << ".";
  }

  if (Arm > maxArm || Station > maxStation || RomanPot > maxRP) {
    throw cms::Exception("InvalidDetId") << "CTPPSDetId ctor:"
                                         << " Invalid parameters:"
                                         << " arm=" << Arm << " station=" << Station << " rp=" << RomanPot << std::endl;
  }

  uint32_t ok = 0xfe000000;
  id_ &= ok;

  id_ |= ((Arm & maskArm) << startArmBit);
  id_ |= ((Station & maskStation) << startStationBit);
  id_ |= ((RomanPot & maskRP) << startRPBit);
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const CTPPSDetId& id) {
  os << "subDet=" << id.subdetId() << " arm=" << id.arm() << " station=" << id.station() << " rp=" << id.rp();

  return os;
}
