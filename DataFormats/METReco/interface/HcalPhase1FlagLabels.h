#ifndef DataFormats_METReco_HcalPhase1FlagLabels_h
#define DataFormats_METReco_HcalPhase1FlagLabels_h

#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

namespace HcalPhase1FlagLabels {
  enum HBHEStatusFlag {
    /*  0 */ HBHEHpdHitMultiplicity = HcalCaloFlagLabels::HBHEHpdHitMultiplicity,
    /* 11 */ HBHEIsolatedNoise = HcalCaloFlagLabels::HBHEIsolatedNoise,
    /* 12 */ HBHEFlatNoise = HcalCaloFlagLabels::HBHEFlatNoise,
    /* 13 */ HBHESpikeNoise = HcalCaloFlagLabels::HBHESpikeNoise,
    /* 15 */ HBHETS4TS5Noise = HcalCaloFlagLabels::HBHETS4TS5Noise,
    /* 27 */ HBHENegativeNoise = HcalCaloFlagLabels::HBHENegativeNoise,
    /* 29 */ HBHEPulseFitBit = HcalCaloFlagLabels::HBHEPulseFitBit,
    /* 30 */ HBHEOOTPU = HcalCaloFlagLabels::HBHEOOTPU
  };

  enum HFStatusFlag {
    /*  0 */ HFLongShort = HcalCaloFlagLabels::HFLongShort,
    /*  3 */ HFS8S1Ratio = HcalCaloFlagLabels::HFS8S1Ratio,
    /*  4 */ HFPET = HcalCaloFlagLabels::HFPET,
    /*  5 */ HFSignalAsymmetry = 5,
    /*  6 */ HFAnomalousHit = 6
  };

  enum CommonFlag {
    /* 20 */ TimingFromTDC = 20,
    /* 31 */ UserDefinedBit0 = HcalCaloFlagLabels::UserDefinedBit0
  };
}  // namespace HcalPhase1FlagLabels

#endif  // DataFormats_METReco_HcalPhase1FlagLabels_h
