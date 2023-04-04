#ifndef DataFormats_HGCalDigis_HGCROCChannelDataFrame_h
#define DataFormats_HGCalDigis_HGCROCChannelDataFrame_h

#include <iostream>
#include <ostream>
#include <cstdint>

/**
   @class HGCROCChannelDataFrame
   @short wrapper for a 32b data word from a single channel and its detid
   The format is always the same: |1b|1b|10b|10b|10b|
   The filling depends on the operation mode (normal or characterization)
   and on the value of the Tc (TOT-complete) and Tp (TOT-in-progress) flags
   See EDMS CMS-CE-ES-0004 for details
 */

template <class D>
class HGCROCChannelDataFrame {
public:
  //although not used directly below, it is used to sort the collection
  typedef D key_type;

  enum HGCROCChannelDataFrameMask { kFlagMask = 0x1, kPacketMask = 0x3ff };

  enum HGCROCChannelDataFrameShift {
    kFlag2Shift = 31,
    kFlag1Shift = 30,
    kPacket3Shift = 20,
    kPacket2Shift = 10,
    kPacket1Shift = 0
  };

  /**
     @short CTOR
  */
  HGCROCChannelDataFrame() : id_(0), value_(0) {}
  HGCROCChannelDataFrame(const D& id) : id_(id), value_(0) {}
  HGCROCChannelDataFrame(uint32_t value) : id_(0), value_(value) {}
  HGCROCChannelDataFrame(const D& id, uint32_t value) : id_(id), value_(value) {}
  HGCROCChannelDataFrame(const HGCROCChannelDataFrame& o) : id_(o.id_), value_(o.value_) {}

  /**
     @short det id
   */
  const D& id() const { return id_; }

  /**
     @short fills the 32b word
     characterization mode : tc|tp|adc|tot|toa
     normal mode: tc|tp|adcm1|*|toa with *=tot if tc==True else adc
   */
  void fill(bool charMode, bool tc, bool tp, uint16_t adcm1, uint16_t adc, uint16_t tot, uint16_t toa) {
    uint16_t word3(charMode ? adc : adcm1);
    uint16_t word2((charMode || tc) ? compressToT(tot) : adc);
    fillRaw(tc, tp, word3, word2, toa);
  }

  /**
     @short setters
  */
  void fillFlag2(bool flag) { fillPacket(flag, kFlagMask, kFlag2Shift); }
  void fillFlag1(bool flag) { fillPacket(flag, kFlagMask, kFlag1Shift); }
  void fillPacket3(int word) { fillPacket(word, kPacketMask, kPacket3Shift); }
  void fillPacket2(int word) { fillPacket(word, kPacketMask, kPacket2Shift); }
  void fillPacket1(int word) { fillPacket(word, kPacketMask, kPacket1Shift); }
  void fillRaw(bool flag2, bool flag1, uint16_t word3, uint16_t word2, uint16_t word1) {
    fillFlag2(flag2);
    fillFlag1(flag1);
    fillPacket3(word3);
    fillPacket2(word2);
    fillPacket1(word1);
  }

  /**
     @short the 12-bit TOT is compressed to a 10bit word truncating the first two bits
     when the value is above 0x1ff=2^8-1. The MSB is set to 1 in case truncation occurs.
   */
  uint16_t compressToT(uint16_t totraw) {
    if (totraw > 0x1ff)
      return (0x200 | (totraw >> 3));
    return (totraw & 0x1ff);
  }

  /**
     @short the 10-bit TOT word is decompressed back to 12 bit word
     In case truncation occurred the word is shifted by 2 bit
   */
  uint16_t decompressToT(uint16_t totraw) const {
    uint16_t totout(totraw & 0x1ff);
    if (totraw & 0x200) {
      totout = ((totraw & 0x1ff) << 3);
      totout += (1 << 2);
    }
    return totout;
  }

  /**
     @short getters
  */
  uint32_t operator()() const { return value_; }
  uint32_t raw() const { return value_; }
  bool tc() const { return flag2(); }
  bool tp() const { return flag1(); }
  uint16_t tctp() const { return (tc() << 1) | tp(); }
  uint16_t adc(bool charMode = false) const { return charMode ? packet3() : (tc() ? 0 : packet2()); }
  uint16_t adcm1(bool charMode = false) const { return charMode ? 0 : packet3(); }
  uint16_t tot(bool charMode = false) const {
    uint16_t tot12b(decompressToT(packet2()));
    return charMode || tc() ? tot12b : 0;
  }
  uint16_t rawtot(bool charMode = false) const { return charMode || tc() ? packet2() : 0; }
  uint16_t toa() const { return packet1(); }
  bool flag2() const { return readPacket(kFlagMask, kFlag2Shift); }
  bool flag1() const { return readPacket(kFlagMask, kFlag1Shift); }
  uint16_t packet3() const { return readPacket(kPacketMask, kPacket3Shift); }
  uint16_t packet2() const { return readPacket(kPacketMask, kPacket2Shift); }
  uint16_t packet1() const { return readPacket(kPacketMask, kPacket1Shift); }

  void print(std::ostream& out = std::cout) const {
    out << "Raw=0x" << std::hex << raw() << std::dec << std::endl
        << "\tf2: " << flag2() << " f1: " << flag1() << " p3: " << packet3() << " p2: " << packet2()
        << " p1: " << packet1() << std::endl
        << "\ttc: " << tc() << " tp: " << tp() << " adcm1: " << adcm1() << " (" << adcm1(false) << ") "
        << " adc: " << adc() << " (" << adc(false) << ") "
        << " tot: " << tot() << " (" << tot(false) << ") "
        << " toa: " << toa() << std::endl;
  }

private:
  /**
     @short wrapper to reset words at a given position
  */
  void fillPacket(uint16_t word, HGCROCChannelDataFrameMask mask, HGCROCChannelDataFrameShift shift) {
    // mask and shift bits
    const uint32_t masked_word = (word & mask) << shift;

    //clear to 0 bits which will be set by word
    value_ &= ~(mask << shift);

    //now set bits
    value_ |= (masked_word);
  }

  /**
     @short wrapper to get packet at a given position
  */
  uint32_t readPacket(HGCROCChannelDataFrameMask mask, HGCROCChannelDataFrameShift shift) const {
    return ((value_ >> shift) & mask);
  }

  //det-id for this dataframe
  D id_;

  // a 32-bit word
  uint32_t value_;
};

#endif
