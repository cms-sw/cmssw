/****************************************************************************
 * Author: Seyed Mohsen Etesami
 *  September 2016
 ****************************************************************************/

#ifndef DataFormats_CTPPSDetId_CTPPSDiamondDetId
#define DataFormats_CTPPSDetId_CTPPSDiamondDetId

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>
#include <string>

/**
 *\brief Detector ID class for CTPPS Timing Diamond detectors.
 * Bits [19:31] : Assigend in CTPPSDetId Calss
 * Bits [17:18] : 2 bits for diamond plane 0,1,2,3
 * Bits [12:16] : 5 bits for Diamond  channel numbers 1,2,3,..16
 * Bits [0:11]  : unspecified yet
 **/

class CTPPSDiamondDetId : public CTPPSDetId {
public:
  /// Construct from a raw id
  explicit CTPPSDiamondDetId(uint32_t id);

  CTPPSDiamondDetId(const CTPPSDetId& id) : CTPPSDetId(id) {}

  /// Construct from hierarchy indeces.
  CTPPSDiamondDetId(uint32_t Arm, uint32_t Station, uint32_t RomanPot = 0, uint32_t Plane = 0, uint32_t Channel = 0);

  static const uint32_t startPlaneBit, maskPlane, maxPlane, lowMaskPlane;
  static const uint32_t startDetBit, maskChannel, maxChannel, lowMaskChannel;

  /// returns true if the raw ID is a PPS-timing one
  static bool check(unsigned int raw) {
    return (((raw >> DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
            ((raw >> DetId::kSubdetOffset) & 0x7) == sdTimingDiamond);
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

  CTPPSDiamondDetId planeId() const { return CTPPSDiamondDetId(rawId() & (~lowMaskPlane)); }

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

std::ostream& operator<<(std::ostream& os, const CTPPSDiamondDetId& id);

#endif
