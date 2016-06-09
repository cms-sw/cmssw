#ifndef RecoLocalCalo_HcalRecAlgos_HFRecHitAuxSetter_h_
#define RecoLocalCalo_HcalRecAlgos_HFRecHitAuxSetter_h_

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HFPreRecHit.h"

//
// Set rechit "auxiliary words" for the dual-anode HF reco
//
struct HFRecHitAuxSetter
{
    // We will store up to three 8-bit ADC values.
    // If the channel is read out, the "sample of
    // interest" ADC will be the third one. The
    // preceding ADCs may or may not be stored.
    static const unsigned MASK_ADC = 0xffffff;
    static const unsigned OFF_ADC = 0;

    // Actual number of ADC values stored.
    // Could be 0, 1, 2, or 3.
    static const unsigned MASK_NTS = 0x3;
    static const unsigned OFF_NTS = 24;

    // CAPID for the "sample of interest"
    static const unsigned MASK_CAPID = 0x3;
    static const unsigned OFF_CAPID = 26;

    // Anode status value. It is assumed that
    // the possible anode status values are
    // defined in the HFAnodeStatus.h header.
    static const unsigned MASK_STATUS = 0xf;
    static const unsigned OFF_STATUS = 28;

    // Main function for setting the aux words
    static void setAux(const HFPreRecHit& prehit,
                       const unsigned anodeStates[2],
                       HFRecHit* rechit);

    // Useful helpers for unsigned fields
    inline static void setField(unsigned* u, const unsigned mask,
                                const unsigned offset, const unsigned value)
        {*u &= ~(mask << offset); *u |= ((value & mask) << offset);}

    inline static unsigned getField(const unsigned u, const unsigned mask,
                                    const unsigned offset)
        {return (u >> offset) & mask;}

    inline static void setBit(unsigned* u, const unsigned bitnum, const bool b)
        {if (b) {*u |= (1U << bitnum);} else {*u &= ~(1U << bitnum);}}

    inline static bool getBit(const unsigned u, const unsigned bitnum)
        {return u & (1U << bitnum);}
};

#endif // RecoLocalCalo_HcalRecAlgos_HFRecHitAuxSetter_h_
