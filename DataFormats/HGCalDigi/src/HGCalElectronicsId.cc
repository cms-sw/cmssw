#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

//
HGCalElectronicsId::HGCalElectronicsId(
    bool zside, uint16_t fedid, uint8_t captureblock, uint8_t econdidx, uint8_t econderx, uint8_t halfrocch) {
  value_ = ((zside & kZsideMask) << kZsideShift) | ((fedid & kLocalFEDIDMask) << kLocalFEDIDShift) |
           ((captureblock & kCaptureBlockMask) << kCaptureBlockShift) | ((econdidx & kECONDIdxMask) << kECONDIdxShift) |
           ((econderx & kECONDeRxMask) << kECONDeRxShift) | ((halfrocch & kHalfROCChannelMask) << kHalfROCChannelShift);
}

//
uint16_t HGCalElectronicsId::localFEDId() const { return (value_ >> kLocalFEDIDShift) & kLocalFEDIDMask; }

//
bool HGCalElectronicsId::zSide() const { return (value_ >> kZsideShift) & kZsideMask; }

//
bool HGCalElectronicsId::isCM() const {
  uint8_t halfrocch = halfrocChannel();
  return (halfrocch == 37) || (halfrocch == 38);
}

//
uint8_t HGCalElectronicsId::captureBlock() const { return (value_ >> kCaptureBlockShift) & kCaptureBlockMask; }

//
uint8_t HGCalElectronicsId::econdIdx() const { return (value_ >> kECONDIdxShift) & kECONDIdxMask; }

//
uint8_t HGCalElectronicsId::econdeRx() const { return (value_ >> kECONDeRxShift) & kECONDeRxMask; }

//
uint8_t HGCalElectronicsId::halfrocChannel() const { return (value_ >> kHalfROCChannelShift) & kHalfROCChannelMask; }

//
uint8_t HGCalElectronicsId::cmWord() const { return halfrocChannel() - 37; }

//
uint8_t HGCalElectronicsId::rocChannel() const {
  if (isCM())
    return cmWord() + 2 * (econdeRx() % 2);
  return halfrocChannel() + 37 * (econdeRx() % 2);
}
