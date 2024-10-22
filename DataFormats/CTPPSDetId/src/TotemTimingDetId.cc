/****************************************************************************
 * Author: Nicola Minafra
 *  March 2018
 ****************************************************************************/

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

//----------------------------------------------------------------------------------------------------

TotemTimingDetId::TotemTimingDetId(uint32_t id) : CTPPSDetId(id) {
  if (!check(id)) {
    throw cms::Exception("InvalidDetId") << "TotemTimingDetId ctor:"
                                         << " channel: " << channel() << " subdet: " << subdetId()
                                         << " is not a valid Totem Timing id";
  }
}

//----------------------------------------------------------------------------------------------------

TotemTimingDetId::TotemTimingDetId(
    uint32_t arm, uint32_t station, uint32_t romanPot, uint32_t plane, uint32_t channel, uint32_t subdet)
    : CTPPSDetId(subdet, arm, station, romanPot) {
  if (arm > maxArm || station > maxStation || romanPot > maxRP || plane > maxPlane || channel > maxChannel) {
    throw cms::Exception("InvalidDetId") << "TotemTimingDetId ctor:"
                                         << " Invalid parameters:"
                                         << " arm=" << arm << " station=" << station << " rp=" << romanPot
                                         << " plane=" << plane << " detector=" << channel << std::endl;
  }

  uint32_t ok = 0xfe000000;
  id_ &= ok;

  id_ |= ((arm & maskArm) << startArmBit);
  id_ |= ((station & maskStation) << startStationBit);
  id_ |= ((romanPot & maskRP) << startRPBit);
  id_ |= ((plane & maskPlane) << startPlaneBit);
  id_ |= ((channel & maskChannel) << startDetBit);
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const TotemTimingDetId& id) {
  return os << "arm=" << id.arm() << " station=" << id.station() << " rp=" << id.rp() << " plane=" << id.plane()
            << " Detector=" << id.channel();
}
