#ifndef DataFormats_HGCalDigis_HGCalElectronicsId_h
#define DataFormats_HGCalDigis_HGCalElectronicsId_h

#include <iostream>
#include <ostream>
#include <cstdint>

/**
   @class HGCalElectronicsId
   @short wrapper for a 32b data word identifying a readout channel in the raw data
   The format is the following:
   Reserved: b'[29,31]
   z side: b'[28]
   Local FED ID: b'[18,27]
   Capture Block ID: b'[14,17]
   ECON-D idx: b'[10,13]
   ECON-D eRx: b'[6,9]
   1/2 ROC channel number: b'[0-5]
 */

class HGCalElectronicsId {
public:
  enum HGCalElectronicsIdMask {
    kZsideMask = 0x1,
    kLocalFEDIDMask = 0x3ff,
    kCaptureBlockMask = 0xf,
    kECONDIdxMask = 0xf,
    kECONDeRxMask = 0xf,
    kHalfROCChannelMask = 0x3f
  };
  enum HGCalElectronicsIdShift {
    kZsideShift = 28,
    kLocalFEDIDShift = 18,
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
      bool zside, uint16_t localfedid, uint8_t captureblock, uint8_t econdidx, uint8_t econderx, uint8_t halfrocch);
  explicit HGCalElectronicsId(uint32_t value) : value_(value) {}

  /**
     @short getters
  */

  uint32_t operator()() const { return value_; }
  bool operator<(const HGCalElectronicsId& oth) const { return value_ < oth.value_; }
  bool operator==(const HGCalElectronicsId& oth) const { return value_ == oth.value_; }
  uint32_t raw() const { return value_; }
  bool zSide() const;
  uint16_t localFEDId() const;
  uint8_t captureBlock() const;
  uint8_t econdIdx() const;
  uint8_t econdeRx() const;
  uint8_t halfrocChannel() const;
  uint8_t rocChannel() const;
  uint8_t cmWord() const;
  bool isCM() const;
  void print(std::ostream& out = std::cout) const {
    out << "Raw=0x" << std::hex << raw() << std::dec << std::endl
        << "\tLocal FED-ID: " << (uint32_t)localFEDId() << " Capture Block: " << (uint32_t)captureBlock()
        << " ECON-D idx: " << (uint32_t)econdIdx() << " eRx: " << (uint32_t)econdeRx()
        << " 1/2 ROC ch.: " << (uint32_t)halfrocChannel() << " isCM=" << isCM() << " zSide=" << zSide() << std::endl;
  }

private:
  // a 32-bit word
  uint32_t value_;
};

#endif
