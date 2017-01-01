/****************************************************************************
 * Author: Seyed Mohsen Etesami 
 * Spetember 2016
 ****************************************************************************/


#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

const uint32_t CTPPSDiamondDetId::startPlaneBit = 17, CTPPSDiamondDetId::maskPlane = 0x3, CTPPSDiamondDetId::maxPlane = 3, CTPPSDiamondDetId::lowMaskPlane = 0x1FFFF;
const uint32_t CTPPSDiamondDetId::startDetBit = 12, CTPPSDiamondDetId::maskChannel = 0x1F, CTPPSDiamondDetId::maxChannel = 11, CTPPSDiamondDetId::lowMaskChannel = 0xFFF;

const string CTPPSDiamondDetId::planeNames[] = { "0", "1", "2", "3" };
const string CTPPSDiamondDetId::channelNames[] = { "0", "1", "2", "3", "4", "05", "06", "07", "08", "09", "10", "11" };

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDetId::CTPPSDiamondDetId(uint32_t id) : CTPPSDetId(id)
{
  if (! check(id))
    {
      throw cms::Exception("InvalidDetId") << "CTPPSDiamondDetId ctor:"
					   << " channel: " << channel()
					   << " subdet: " << subdetId()
					   << " is not a valid CTPPS Timing Diamond id";  
    }
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDetId::CTPPSDiamondDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot, uint32_t Plane, uint32_t Channel) :       
  CTPPSDetId(sdTimingDiamond, Arm, Station, RomanPot)
{
  if (Arm > maxArm || Station > maxStation || RomanPot > maxRP || Plane > maxPlane || Channel > maxChannel)
    {
      throw cms::Exception("InvalidDetId") << "CTPPSDiamondDetId ctor:" 
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

std::ostream& operator << (std::ostream& os, const CTPPSDiamondDetId& id)
{
  os << "arm=" << id.arm()
     << " station=" << id.station()
     << " rp=" << id.rp()
     << " plane=" << id.plane()
     << " Detector=" << id.channel();

  return os;
}
