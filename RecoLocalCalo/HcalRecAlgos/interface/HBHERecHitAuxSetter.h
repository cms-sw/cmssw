#ifndef RecoLocalCalo_HcalRecAlgos_HBHERecHitAuxSetter_h_
#define RecoLocalCalo_HcalRecAlgos_HBHERecHitAuxSetter_h_

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"

//
// Set rechit "auxiliary words" for the Phase 1 HBHE reco.
//
// The standars CaloRecHit "aux" word contains ADC values 0-3.
//
// The "auxHBHE" word of HBHERecHit contains ADC values 4-7.
//
// The "auxPhase1" word of HBHERecHit contains ADC values 8-9
// and other info, as specified by the masks and offsets below.
//
struct HBHERecHitAuxSetter
{
    static const unsigned MASK_ADC = 0xffff;
    static const unsigned OFF_ADC = 0;

    // How many ADC values are actually stored
    static const unsigned MASK_NSAMPLES = 0xf;
    static const unsigned OFF_NSAMPLES = 16;

    // Which ADC corresponds to the sample of interest.
    static const unsigned MASK_SOI = 0xf;
    static const unsigned OFF_SOI = 20;

    // CAPID for the sample of interest.
    static const unsigned MASK_CAPID = 0x3;
    static const unsigned OFF_CAPID = 24;

    // Various status bits (pack bools from HBHEChannelInfo)
    static const unsigned OFF_TDC_TIME = 26;
    static const unsigned OFF_DROPPED = 27;
    static const unsigned OFF_LINK_ERR = 28;
    static const unsigned OFF_CAPID_ERR = 29;

    // Main function for setting the aux words.
    static void setAux(const HBHEChannelInfo& info,
                       HBHERecHit* rechit);
};

#endif // RecoLocalCalo_HcalRecAlgos_HBHERecHitAuxSetter_h_
