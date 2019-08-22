#ifndef RecoLocalCalo_HcalRecAlgos_HFAnodeStatus_h_
#define RecoLocalCalo_HcalRecAlgos_HFAnodeStatus_h_

// Mutually exclusive status settings for the dual-anode HF readout
namespace HFAnodeStatus {
  enum {
    OK = 0,          // Good for rechit reconstruction
    NOT_DUAL,        // Single-anode readout in the mixed-readout scheme
    NOT_READ_OUT,    // Zero-suppressed (by hardware or software) or missing
    HARDWARE_ERROR,  // "OK" flag is not set by hardware
    FLAGGED_BAD,     // Flagged as bad channel by calibrations
    FAILED_TIMING,   // Failed timing selection cuts
    FAILED_OTHER,    // Rejected for some other reason
    N_POSSIBLE_STATES
  };
}

#endif  // RecoLocalCalo_HcalRecAlgos_HFAnodeStatus_h_
