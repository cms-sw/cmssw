#ifndef RecoLocalCalo_HcalRecAlgos_HFSimpleTimeCheck_h_
#define RecoLocalCalo_HcalRecAlgos_HFSimpleTimeCheck_h_

#include <utility>

#include "RecoLocalCalo/HcalRecAlgos/interface/AbsHFPhase1Algo.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFAnodeStatus.h"

class HFSimpleTimeCheck : public AbsHFPhase1Algo
{
public:
    // "tlimits" are the rise time limits for the anode pair.
    // The first element of the pair is the min rise time and the
    // second element is the max rise time. tlimits[0] is for the
    // first anode and tlimits[1] is for the second one.
    //
    // "energyWeights" is the lookup table for the energy weights
    // based on the multi-state decision about anode quality.
    // The first index of this array corresponds to the decision
    // about the status of the anodes, and the second index corresponds
    // to the anode number. Possible status values are given in the
    // HFAnodeStatus enum. Mapping of the first index to the possible
    // status values is as follows:
    //
    // Indices 0 to HFAnodeStatus::N_POSSIBLE_STATES-1 correspond to
    // the situations in which the first anode has the status "OK"
    // and the second anode has the status given by the index.
    // 
    // HFAnodeStatus::N_POSSIBLE_STATES to HFAnodeStatus::N_POSSIBLE_STATES-2
    // correspond to the situations in which the second anode has
    // the status "OK" and the first anode has the status given
    // by index - HFAnodeStatus::N_POSSIBLE_STATES + 1. This excludes
    // the state {OK, OK} already covered.
    //
    // "soiPhase" argument specifies the desired position of the
    // sample of interest ADC in the ADC bytes written out into the
    // aux words of the HFRecHit. For more detail, see comments
    // inside the HFRecHitAuxSetter.h header.
    //
    // "timeShift" value (in ns) will be added to all valid times
    // returned by QIE10 TDCs. This shift is used both for applying
    // the timing cuts and for rechit construction.
    //
    // "triseIfNoTDC" and "tfallIfNoTDC": the rechit rise and
    // fall times will be set to these values in case meaningful
    // TDC information is not available for any of the PMT anodes
    // (time shift is not added to these numbers).
    //
    // For monitoring purposes, "rejectAllFailures" can be set to
    // "false". In this case, for the energy reconstruction purposes,
    // all status values indicating that the anode is not passing
    // algorithm cuts will be mapped to "OK". However, HFRecHit
    // will still be made using proper status flags.
    //
    // If "alwaysCalculateChargeAsymmetry" is true, charge asymmetry
    // status bit will be set whenever the data is available for both
    // anodes. If "alwaysCalculateChargeAsymmetry" is false, the bit
    // will be set only if the status of both anodes is "OK" (or mapped
    // into "OK").
    //
    HFSimpleTimeCheck(const std::pair<float,float> tlimits[2],
                      const float energyWeights[2*HFAnodeStatus::N_POSSIBLE_STATES-1][2],
                      unsigned soiPhase, float timeShift,
                      float triseIfNoTDC, float tfallIfNoTDC,
                      float minChargeForUndershoot, float minChargeForOvershoot,
                      bool rejectAllFailures = true,
                      bool alwaysCalculateChargeAsymmetry = true);

    inline ~HFSimpleTimeCheck() override {}

    inline bool isConfigurable() const override {return false;}

    HFRecHit reconstruct(const HFPreRecHit& prehit,
                         const HcalCalibrations& calibs,
                         const bool flaggedBadInDB[2],
                         bool expectSingleAnodePMT) override;

    inline unsigned soiPhase() const {return soiPhase_;}
    inline float timeShift() const {return timeShift_;}
    inline float triseIfNoTDC() const {return triseIfNoTDC_;}
    inline float tfallIfNoTDC() const {return tfallIfNoTDC_;}
    inline float minChargeForUndershoot() const {return minChargeForUndershoot_;}
    inline float minChargeForOvershoot() const {return minChargeForOvershoot_;}
    inline bool rejectingAllFailures() const {return rejectAllFailures_;}
    inline bool alwaysCalculatingQAsym() const {return alwaysQAsym_;}

protected:
    virtual unsigned determineAnodeStatus(unsigned anodeNumber,
                                          const HFQIE10Info& anode,
                                          bool* isTimingReliable) const;
private:
    // Map possible status values into the first index of "energyWeights_"
    unsigned mapStatusIntoIndex(const unsigned states[2]) const;

    std::pair<float,float> tlimits_[2];
    float energyWeights_[2*HFAnodeStatus::N_POSSIBLE_STATES-1][2];
    unsigned soiPhase_;
    float timeShift_;
    float triseIfNoTDC_;
    float tfallIfNoTDC_;
    float minChargeForUndershoot_;
    float minChargeForOvershoot_;
    bool rejectAllFailures_;
    bool alwaysQAsym_;
};

#endif // RecoLocalCalo_HcalRecAlgos_HFSimpleTimeCheck_h_
