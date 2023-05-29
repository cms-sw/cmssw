#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

//
HGCalElectronicsId::HGCalElectronicsId(
    uint16_t fedid, uint8_t captureblock, uint8_t econdidx, uint8_t econderx, uint8_t halfrocch) {
  value_ = ((fedid & kFEDIDMask) << kFEDIDShift) | ((captureblock & kCaptureBlockMask) << kCaptureBlockShift) |
           ((econdidx & kECONDIdxMask) << kECONDIdxShift) | ((econderx & kECONDeRxMask) << kECONDeRxShift) |
           ((halfrocch & kHalfROCChannelMask) << kHalfROCChannelShift);
}

//
uint16_t HGCalElectronicsId::fedId() const { return (value_ >> kFEDIDShift) & kFEDIDMask; }

//
uint8_t HGCalElectronicsId::captureBlock() const { return (value_ >> kCaptureBlockShift) & kCaptureBlockMask; }

//
uint8_t HGCalElectronicsId::econdIdx() const { return (value_ >> kECONDIdxShift) & kECONDIdxMask; }

//
uint8_t HGCalElectronicsId::econdeRx() const { return (value_ >> kECONDeRxShift) & kECONDeRxMask; }

//
uint8_t HGCalElectronicsId::halfrocChannel() const { return (value_ >> kHalfROCChannelShift) & kHalfROCChannelMask; }
