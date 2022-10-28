/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *
 ****************************************************************************/

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

//----------------------------------------------------------------------------------------------------

TotemT2DetId::TotemT2DetId(uint32_t id) : CTPPSDetId(id) {
  if (!check(id))
    throw cms::Exception("InvalidDetId") << "TotemT2DetId ctor:"
                                         << " channel: " << channel() << " subdet: " << subdetId()
                                         << " is not a valid Totem nT2 id";
}

//----------------------------------------------------------------------------------------------------

TotemT2DetId::TotemT2DetId(uint32_t arm, uint32_t plane, uint32_t channel) : CTPPSDetId(sdTotemT2, arm, 0, 0) {
  if (arm > maxArm || plane > maxPlane || channel > maxChannel)
    throw cms::Exception("InvalidDetId") << "TotemT2DetId ctor:"
                                         << " Invalid parameters:"
                                         << " arm=" << arm << " plane=" << plane << " detector=" << channel;

  uint32_t ok = 0xfe000000;
  id_ &= ok;

  id_ |= ((arm & maskArm) << startArmBit);
  id_ |= ((plane & maskPlane) << startPlaneBit);
  id_ |= ((channel & maskChannel) << startChannelBit);
}

//----------------------------------------------------------------------------------------------------

void TotemT2DetId::planeName(std::string& name, NameFlag flag) const {
  switch (flag) {
    case nShort:
      name = "";
      break;
    case nFull:
      armName(name, flag);
      name += "_";
      break;
    case nPath:
      armName(name, flag);
      name += "/plane ";
      break;
  }
  name += std::to_string(plane());
}

//----------------------------------------------------------------------------------------------------

void TotemT2DetId::channelName(std::string& name, NameFlag flag) const {
  switch (flag) {
    case nShort:
      name = "";
      break;
    case nFull:
      planeName(name, flag);
      name += "_";
      break;
    case nPath:
      planeName(name, flag);
      name += "/channel ";
      break;
  }
  name += std::to_string(channel());
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const TotemT2DetId& id) {
  return os << "arm=" << id.arm() << " plane=" << id.plane() << " channel=" << id.channel();
}
