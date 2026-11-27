#ifndef EventFilter_MTDRawToDigi_interface_BTLRawDataFormat_h
#define EventFilter_MTDRawToDigi_interface_BTLRawDataFormat_h

// Format of the BTL channel payload in the raw data: one 128-bit word per TOFHIR channel.
// Shared by the packer (BTLDigiToRaw), the alpaka unpacker (BTLRawToDigi) and the test tools,
// so that the bit layout is defined in a single place.

#include <cstddef>
#include <cstdint>

#include <alpaka/core/Common.hpp>

namespace btlraw {

  // One channel word: bits 0-63 in lo, bits 64-127 in hi (16 bytes in the raw data, lo first)
  struct ChannelWord {
    uint64_t lo = 0;
    uint64_t hi = 0;
  };
  static_assert(sizeof(ChannelWord) == 16, "a BTL channel word must be 128 bit");

  // A field of the channel word: first bit (0 = LSB of lo) and number of bits
  struct Field {
    int pos;
    int width;
  };

  // FIXME: provisional bit layout, to be updated once the BTL payload format is frozen
  // bits 82-127: header (BC0count, status, S-link, HS-link, e-link, channel, BC count); bits 0-81: TOFHIR data
  inline constexpr Field kBC0count{118, 10};
  inline constexpr Field kStatus{117, 1};
  inline constexpr Field kSLink{110, 7};  // likely to be removed from the channel payload
  inline constexpr Field kHSLink{104, 6};
  inline constexpr Field kELink{99, 5};
  inline constexpr Field kChId{94, 5};
  inline constexpr Field kBCcount{82, 12};
  inline constexpr Field kT1Coarse{67, 15};
  inline constexpr Field kT2Coarse{57, 10};
  inline constexpr Field kEOICoarse{47, 10};
  inline constexpr Field kCharge{37, 10};
  inline constexpr Field kT1Fine{27, 10};
  inline constexpr Field kT2Fine{17, 10};
  inline constexpr Field kIdleTime{7, 10};
  inline constexpr Field kPrevTrigF{3, 4};
  inline constexpr Field kTACId{0, 3};

  inline constexpr Field kAllFields[] = {kBC0count,
                                         kStatus,
                                         kSLink,
                                         kHSLink,
                                         kELink,
                                         kChId,
                                         kBCcount,
                                         kT1Coarse,
                                         kT2Coarse,
                                         kEOICoarse,
                                         kCharge,
                                         kT1Fine,
                                         kT2Fine,
                                         kIdleTime,
                                         kPrevTrigF,
                                         kTACId};
  inline constexpr std::size_t kNumberOfFields = sizeof(kAllFields) / sizeof(Field);

  // mask with the lowest 'width' bits set, width in [1, 64]
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint64_t lowMask(int width) {
    return (width == 64) ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);
  }

  // Read a field (returned in the low bits), with position and width passed as integers:
  // used in device code, where passing a Field object makes nvcc read it from global memory
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint64_t extractBits(const ChannelWord& w, int pos, int width) {
    const uint64_t mask = lowMask(width);
    if (pos >= 64)  // entirely in hi
      return (w.hi >> (pos - 64)) & mask;
    if (pos + width <= 64)  // entirely in lo
      return (w.lo >> pos) & mask;
    // across the 64-bit boundary (0 < pos < 64)
    return ((w.lo >> pos) | (w.hi << (64 - pos))) & mask;
  }

  // Read a field (returned in the low bits)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint64_t extractBits(const ChannelWord& w, Field f) {
    return extractBits(w, f.pos, f.width);
  }

  // Write a field: only the lowest f.width bits of value are used, the other bits of the word are unchanged
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr void insertBits(ChannelWord& w, Field f, uint64_t value) {
    value &= lowMask(f.width);
    if (f.pos >= 64) {  // entirely in hi
      const uint64_t mask = lowMask(f.width) << (f.pos - 64);
      w.hi = (w.hi & ~mask) | (value << (f.pos - 64));
      return;
    }
    const uint64_t maskLo = lowMask(f.width) << f.pos;  // the bits above 63 are dropped by the shift
    w.lo = (w.lo & ~maskLo) | (value << f.pos);
    if (f.pos + f.width > 64) {  // across the 64-bit boundary: the upper part goes in the low bits of hi
      const uint64_t maskHi = lowMask(f.pos + f.width - 64);
      w.hi = (w.hi & ~maskHi) | (value >> (64 - f.pos));
    }
  }

  // ---- compile-time checks of the format ----

  // every field has width in [1, 64] and lies in the 128 bits, and no two fields overlap
  constexpr bool isValidFormat() {
    for (std::size_t i = 0; i < kNumberOfFields; ++i) {
      const Field a = kAllFields[i];
      if (a.width < 1 || a.width > 64 || a.pos < 0 || a.pos + a.width > 128)
        return false;
      for (std::size_t j = i + 1; j < kNumberOfFields; ++j) {
        const Field b = kAllFields[j];
        if (a.pos < b.pos + b.width && b.pos < a.pos + a.width)
          return false;
      }
    }
    return true;
  }
  static_assert(isValidFormat(), "invalid BTL channel payload format: field out of range or overlapping fields");

  // writing and reading back each field with all bits set gives the same value and leaves the other fields at zero
  constexpr bool isRoundTripOk() {
    for (std::size_t i = 0; i < kNumberOfFields; ++i) {
      ChannelWord w;
      insertBits(w, kAllFields[i], ~uint64_t(0));
      if (extractBits(w, kAllFields[i]) != lowMask(kAllFields[i].width))
        return false;
      for (std::size_t j = 0; j < kNumberOfFields; ++j) {
        if (j != i && extractBits(w, kAllFields[j]) != 0)
          return false;
      }
    }
    return true;
  }
  static_assert(isRoundTripOk(), "insertBits/extractBits round trip failed for the BTL channel payload format");

}  // namespace btlraw

#endif
