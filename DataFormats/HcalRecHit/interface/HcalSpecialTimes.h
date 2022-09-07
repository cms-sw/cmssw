#ifndef DataFormats_HcalRecHit_HcalSpecialTimes_h_
#define DataFormats_HcalRecHit_HcalSpecialTimes_h_

// This is an excerpt from QIE10/QIE11 TDC specifications (by T. Zimmerman):
//
// Special codes: Special code 62 is generated when the discriminator
// starts high. Special code 63 is generated when the TDC discriminator
// starts low and remains low (nothing happened). Special code 58 indicates
// "Invalid Code". This can be caused by an SEU in the TDC encoder logic.
// It can also happen in certain situations when the TDC is operated in
// "Last Mode", as discussed above. Code 59 is generated if the either
// of the Delay Locked Loops on the QIE10 (the Phase DLL or the TDC DLL)
// are not locked. Code 60 is generated when the Phase DLL is unlocked
// (but the TDC DLL is locked), and code 61 for the TDC DLL unlocked
// (but the Phase DLL is locked). If either of the DLLs on the chip are
// unlocked, this takes precedence over any TDC data that might be present,
// and the appropriate DLL no-lock condition is reported.
//
// (end of Zimmerman's special code explanation).
//
// In addition, during normal operation, codes above 49 are not supposed
// to happen. If, due to jitter or something, the upcrossing would happen
// at TDC=50, it would be reported as TDC=0 in the next BX.

namespace HcalSpecialTimes {
  // Special value for the rise time used in case the QIE10/11 pulse
  // is always below the discriminator
  constexpr float UNKNOWN_T_UNDERSHOOT = -100.f;

  // "Invalid Code" TDC value
  constexpr float UNKNOWN_T_INVALID_CODE = -105.f;

  // Special value for the rise time used in case the QIE10/11 pulse
  // is always above the discriminator
  constexpr float UNKNOWN_T_OVERSHOOT = -110.f;

  // Any of the codes indicating DLL failures
  constexpr float UNKNOWN_T_DLL_FAILURE = -115.f;

  // Special value for the time to use in case the TDC info is
  // not available or not meaningful (e.g., for QIE8)
  constexpr float UNKNOWN_T_NOTDC = -120.f;

  // Special value which indicates a possible bug in the dataframe
  constexpr float UNKNOWN_T_INVALID_RANGE = -125.f;

  // Special value for invalid codes 50-57. I don't know the
  // exact explanation why they occur, but they do. Their origin
  // is likely to be just a bit-flip.
  constexpr float UNKNOWN_T_50TO57 = -130.f;

  // Check if the given time represents one of the special values
  constexpr inline bool isSpecial(const float t) { return t <= UNKNOWN_T_UNDERSHOOT; }

  constexpr float DEFAULT_ccTIME = -999.f;

  constexpr inline float getTDCTime(const int tdc) {
    constexpr float tdc_to_ns = 0.5f;

    constexpr int six_bits_mask = 0x3f;
    constexpr int tdc_code_largestnormal = 49;
    constexpr int tdc_code_invalid = 58;
    constexpr int tdc_code_overshoot = 62;
    constexpr int tdc_code_undershoot = 63;

    float t = tdc_to_ns * tdc;
    if (tdc > six_bits_mask || tdc < 0)
      t = UNKNOWN_T_INVALID_RANGE;
    else if (tdc > tdc_code_largestnormal) {
      // The undershoot code happens by far more often
      // than any other special code. So check for it first.
      if (tdc == tdc_code_undershoot)
        t = UNKNOWN_T_UNDERSHOOT;
      else if (tdc == tdc_code_overshoot)
        t = UNKNOWN_T_OVERSHOOT;
      else if (tdc == tdc_code_invalid)
        t = UNKNOWN_T_INVALID_CODE;
      else if (tdc < tdc_code_invalid)
        t = UNKNOWN_T_50TO57;
      else
        t = UNKNOWN_T_DLL_FAILURE;
    }

    return t;
  }
}  // namespace HcalSpecialTimes

#endif  // DataFormats_HcalRecHit_HcalSpecialTimes_h_
