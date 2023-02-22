#ifndef DataFormats_HGCalDigis_HGCalElectronicsId_h
#define DataFormats_HGCalDigis_HGCalElectronicsId_h

#include <iostream>
#include <ostream>
#include <cstdint>

/**
   @class HGCalElectronicsId
   @short wrapper for a 32b data word identifying a readout channel in the raw data
   The format is the following:
   Reserved: b'[28,31]
   FED ID: b'[18,27]
   Capture Block ID: b'[14,17]
   ECON-D idx: b'[10,13]
   ECON-D eRx: b'[6,9]
   1/2 ROC channel number: b'[0-5]
 */

class HGCalElectronicsId {
public:
  enum HGCalElectronicsIdMask {
    kFEDIDMask = 0x3ff,
    kCaptureBlockMask = 0xf,
    kECONDIdxMask = 0xf,
    kECONDeRxMask = 0xf,
    kHalfROCChannelMask = 0x3f
  };
  enum HGCalElectronicsIdShift {
    kFEDIDShift = 18,
    kCaptureBlockShift = 14,
    kECONDIdxShift = 10,
    kECONDeRxShift = 6,
    kHalfROCChannelShift = 0
  };

  /**
     @short CTOR
  */
  HGCalElectronicsId() : value_(0) {}
  explicit HGCalElectronicsId(
      uint16_t fedid, uint8_t captureblock, uint8_t econdidx, uint8_t econderx, uint8_t halfrocch);
  explicit HGCalElectronicsId(uint32_t value) : value_(value) {}
  HGCalElectronicsId(const HGCalElectronicsId& o) : value_(o.value_) {}

  /**
     @short getters
  */
  uint32_t operator()() const { return value_; }
  bool operator<(const HGCalElectronicsId& oth) const { return value_ < oth.value_; }
  uint32_t raw() const { return value_; }
  uint16_t fedId() const;
  uint8_t captureBlock() const;
  uint8_t econdIdx() const;
  uint8_t econdeRx() const;
  uint8_t halfrocChannel() const;

  void print(std::ostream& out = std::cout) {
    out << "Raw=0x" << std::hex << raw() << std::dec << std::endl
        << "\tFED-ID: " << (uint32_t)fedId() << " Capture Block: " << (uint32_t)captureBlock()
        << " ECON-D idx: " << (uint32_t)econdIdx() << " eRx: " << (uint32_t)econdeRx()
        << " 1/2 ROC ch.: " << (uint32_t)halfrocChannel() << std::endl;
  }

private:
  // a 32-bit word
  uint32_t value_;
};

#endif
