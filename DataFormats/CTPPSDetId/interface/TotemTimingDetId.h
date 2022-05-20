/****************************************************************************
 * Author: Nicola Minafra
 *  March 2018
 ****************************************************************************/

#ifndef DataFormats_CTPPSDetId_TotemTimingDetId
#define DataFormats_CTPPSDetId_TotemTimingDetId

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>
#include <string>

/**
 *\brief Detector ID class for CTPPS Totem Timing detectors.
 * Bits [19:31] : Assigend in CTPPSDetId Calss
 * Bits [17:18] : 2 bits for UFSD plane 0,1,2,3
 * Bits [12:16] : 5 bits for UFSD channel numbers 1,2,3,..16
 * Bits [0:11]  : unspecified yet
 *
 * This class is very similar to CTPPSDiamondDetId; however the detector is completely separated, therefore it is useful to keep them separate and independent.
 **/

class TotemTimingDetId : public CTPPSDetId {
public:
  enum { ID_NOT_SET = 28 };

  /// Construct from a raw id
  explicit TotemTimingDetId(uint32_t id);
  TotemTimingDetId(const CTPPSDetId& id) : CTPPSDetId(id) {}

  /// Construct from hierarchy indices.
  TotemTimingDetId(uint32_t arm,
                   uint32_t station,
                   uint32_t romanPot = 0,
                   uint32_t plane = 0,
                   uint32_t channel = 0,
                   uint32_t subdet = sdTimingFastSilicon);

  static constexpr uint32_t startPlaneBit = 17, maskPlane = 0x3, maxPlane = 3, lowMaskPlane = 0x1FFFF;
  static constexpr uint32_t startDetBit = 12, maskChannel = 0x1F, maxChannel = 31, lowMaskChannel = 0xFFF;

  /// returns true if the raw ID is a PPS-timing one
  static bool check(unsigned int raw) {
    return (((raw >> DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
            (((raw >> DetId::kSubdetOffset) & 0x7) == sdTimingFastSilicon ||
             ((raw >> DetId::kSubdetOffset) & 0x7) == sdTimingDiamond));
  }
  //-------------------- getting and setting methods --------------------

  uint32_t plane() const { return ((id_ >> startPlaneBit) & maskPlane); }

  void setPlane(uint32_t channel) {
    id_ &= ~(maskPlane << startPlaneBit);
    id_ |= ((channel & maskPlane) << startPlaneBit);
  }

  uint32_t channel() const { return ((id_ >> startDetBit) & maskChannel); }

  void setChannel(uint32_t channel) {
    id_ &= ~(maskChannel << startDetBit);
    id_ |= ((channel & maskChannel) << startDetBit);
  }

  //-------------------- id getters for higher-level objects --------------------

  TotemTimingDetId planeId() const { return TotemTimingDetId(rawId() & (~lowMaskPlane)); }

  //-------------------- name methods --------------------

  inline void planeName(std::string& name, NameFlag flag = nFull) const {
    switch (flag) {
      case nShort:
        name = "";
        break;
      case nFull:
        rpName(name, flag);
        name += "_";
        break;
      case nPath:
        rpName(name, flag);
        name += "/plane ";
        break;
    }
    name += std::to_string(plane());
  }

  inline void channelName(std::string& name, NameFlag flag = nFull) const {
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
};

std::ostream& operator<<(std::ostream& os, const TotemTimingDetId& id);

#endif
