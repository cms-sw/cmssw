#ifndef RecoLocalCalo_HcalRecAlgos_HFRecHitAuxSetter_h_
#define RecoLocalCalo_HcalRecAlgos_HFRecHitAuxSetter_h_

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HFPreRecHit.h"

//
// Set rechit "auxiliary words" for the dual-anode HF reco
//
struct HFRecHitAuxSetter {
  // We will store up to three 8-bit ADC values
  static const unsigned MASK_ADC = 0xffffff;
  static const unsigned OFF_ADC = 0;

  // Which byte is used by the sample of interest.
  // Normally 0, 1, or 2. Value of 3 means that
  // something went wrong, and SOI was not stored
  // in the HFPreRecHit.
  static const unsigned MASK_SOI = 0x3;
  static const unsigned OFF_SOI = 24;

  // CAPID for the sample of interest.
  // Will be correct only if the SOI value
  // is less than 3.
  static const unsigned MASK_CAPID = 0x3;
  static const unsigned OFF_CAPID = 26;

  // Anode status value. It is assumed that
  // the possible anode status values are
  // defined in the HFAnodeStatus.h header.
  static const unsigned MASK_STATUS = 0xf;
  static const unsigned OFF_STATUS = 28;

  // Main function for setting the aux words.
  //
  // "soiPhase" argument tells us the byte into
  // which the sample of interest should be placed
  // in the aux word. "soiPhase" should be 0, 1, or 2.
  // The number of presamples stored will normally
  // be equal "soiPhase" and the number of postsamples
  // (2 - soiPhase).
  //
  static void setAux(const HFPreRecHit& prehit, const unsigned anodeStates[2], unsigned soiPhase, HFRecHit* rechit);
};

#endif  // RecoLocalCalo_HcalRecAlgos_HFRecHitAuxSetter_h_
