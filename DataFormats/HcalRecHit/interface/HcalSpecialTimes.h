#ifndef DataFormats_HcalRecHit_HcalSpecialTimes_h_
#define DataFormats_HcalRecHit_HcalSpecialTimes_h_

namespace HcalSpecialTimes
{
    // Special value for the rise time used in case the QIE10/11 pulse
    // is always below the discriminator
    constexpr float UNKNOWN_T_UNDERSHOOT = -100.f;

    // Special value for the rise time used in case the QIE10/11 pulse
    // is always above the discriminator
    constexpr float UNKNOWN_T_OVERSHOOT = -110.f;

    // Special value for the time to use in case the TDC info is
    // not available or not meaningful (e.g., for QIE8)
    constexpr float UNKNOWN_T_NOTDC = -120.f;

    // Check if the given time represents one of the special values
    inline bool isSpecial(const float t)
    {
        return t == UNKNOWN_T_UNDERSHOOT ||
               t == UNKNOWN_T_OVERSHOOT ||
               t == UNKNOWN_T_NOTDC;
    }
}

#endif // DataFormats_HcalRecHit_HcalSpecialTimes_h_
