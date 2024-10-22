#ifndef DataFormats_HcalRecHit_CaloRecHitAuxSetter_h_
#define DataFormats_HcalRecHit_CaloRecHitAuxSetter_h_

#include <cstdint>

// Useful helpers for uint32_t fields
namespace CaloRecHitAuxSetter {
  constexpr inline void setField(uint32_t* u, const unsigned mask, const unsigned offset, const unsigned value) {
    *u &= ~(mask << offset);
    *u |= ((value & mask) << offset);
  }

  constexpr inline unsigned getField(const uint32_t u, const unsigned mask, const unsigned offset) {
    return (u >> offset) & mask;
  }

  constexpr inline void setBit(uint32_t* u, const unsigned bitnum, const bool b) {
    if (b) {
      *u |= (1U << bitnum);
    } else {
      *u &= ~(1U << bitnum);
    }
  }

  constexpr inline void orBit(uint32_t* u, const unsigned bitnum, const bool b) {
    if (b) {
      *u |= (1U << bitnum);
    }
  }

  constexpr inline void andBit(uint32_t* u, const unsigned bitnum, const bool b) {
    if (!b) {
      *u &= ~(1U << bitnum);
    }
  }

  constexpr inline bool getBit(const uint32_t u, const unsigned bitnum) { return u & (1U << bitnum); }
}  // namespace CaloRecHitAuxSetter

#endif  // DataFormats_HcalRecHit_CaloRecHitAuxSetter_h_
