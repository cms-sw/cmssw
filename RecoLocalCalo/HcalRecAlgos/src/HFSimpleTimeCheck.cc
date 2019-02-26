#include <cstring>
#include <climits>

#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HFSimpleTimeCheck.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFRecHitAuxSetter.h"

// Phase 1 rechit status bit assignments
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"

namespace {  
    inline float build_rechit_time(const float weightedEnergySum,
                                   const float weightedSum,
                                   const float sum,
                                   const unsigned count,
                                   const float valueIfNothingWorks,
                                   bool* resultComesFromTDC)
    {
        if (weightedEnergySum > 0.f)
        {
            *resultComesFromTDC = true;
            return weightedSum/weightedEnergySum;
        }
        else if (count)
        {
            *resultComesFromTDC = true;
            return sum/count;
        }
        else
        {
            *resultComesFromTDC = false;
            return valueIfNothingWorks;
        }
    }
}


HFSimpleTimeCheck::HFSimpleTimeCheck(const std::pair<float,float> tlimits[2],
                                     const float energyWeights[2*HFAnodeStatus::N_POSSIBLE_STATES-1][2],
                                     const unsigned i_soiPhase,
                                     const float i_timeShift,
                                     const float i_triseIfNoTDC,
                                     const float i_tfallIfNoTDC,
                                     const float i_minChargeForUndershoot,
                                     const float i_minChargeForOvershoot,
                                     const bool rejectAllFailures,
                                     const bool alwaysCalculateQAsymmetry)
    : soiPhase_(i_soiPhase),
      timeShift_(i_timeShift),
      triseIfNoTDC_(i_triseIfNoTDC),
      tfallIfNoTDC_(i_tfallIfNoTDC),
      minChargeForUndershoot_(i_minChargeForUndershoot),
      minChargeForOvershoot_(i_minChargeForOvershoot),
      rejectAllFailures_(rejectAllFailures),
      alwaysQAsym_(alwaysCalculateQAsymmetry)
{
    tlimits_[0] = tlimits[0];
    tlimits_[1] = tlimits[1];
    float* to = &energyWeights_[0][0];
    const float* from = &energyWeights[0][0];
    memcpy(to, from, sizeof(energyWeights_));
}

unsigned HFSimpleTimeCheck::determineAnodeStatus(
    const unsigned ianode, const HFQIE10Info& anode, bool*) const
{
    // Check if this anode has a dataframe error
    if (!anode.isDataframeOK())
        return HFAnodeStatus::HARDWARE_ERROR;

    // Check the time limits
    float trise = anode.timeRising();
    const bool timeIsKnown = !HcalSpecialTimes::isSpecial(trise);
    trise += timeShift_;
    if (timeIsKnown &&
        tlimits_[ianode].first <= trise && trise <= tlimits_[ianode].second)
        return HFAnodeStatus::OK;
    else
        return HFAnodeStatus::FAILED_TIMING;
}

unsigned HFSimpleTimeCheck::mapStatusIntoIndex(const unsigned states[2]) const
{
    unsigned eStates[2];
    eStates[0] = states[0];
    eStates[1] = states[1];
    if (!rejectAllFailures_)
        for (unsigned i=0; i<2; ++i)
            if (eStates[i] == HFAnodeStatus::FAILED_TIMING ||
                eStates[i] == HFAnodeStatus::FAILED_OTHER)
                eStates[i] = HFAnodeStatus::OK;
    if (eStates[0] == HFAnodeStatus::OK)
        return eStates[1];
    else if (eStates[1] == HFAnodeStatus::OK)
        return HFAnodeStatus::N_POSSIBLE_STATES + eStates[0] - 1;
    else
        return UINT_MAX;
}

HFRecHit HFSimpleTimeCheck::reconstruct(const HFPreRecHit& prehit,
                                        const HcalCalibrations& /* calibs */,
                                        const bool flaggedBadInDB[2],
                                        const bool expectSingleAnodePMT)
{
    HFRecHit rh;

    // Determine the status of each anode
    unsigned states[2] = {HFAnodeStatus::NOT_READ_OUT, HFAnodeStatus::NOT_READ_OUT};
    if (expectSingleAnodePMT)
        states[1] = HFAnodeStatus::NOT_DUAL;

    bool isTimingReliable[2] = {true, true};
    for (unsigned ianode=0; ianode<2; ++ianode)
    {
        if (flaggedBadInDB[ianode])
            states[ianode] = HFAnodeStatus::FLAGGED_BAD;
        else
        {
            const HFQIE10Info* anodeInfo = prehit.getHFQIE10Info(ianode);
            if (anodeInfo)
                states[ianode] = determineAnodeStatus(ianode, *anodeInfo,
                                                      &isTimingReliable[ianode]);
        }
    }

    // Reconstruct energy and time
    const unsigned lookupInd = mapStatusIntoIndex(states);
    if (lookupInd != UINT_MAX)
    {
        // In this scope, at least one of states[i] is HFAnodeStatus::OK
        // or was mapped into that status by "mapStatusIntoIndex" method
        //
        const float* weights = &energyWeights_[lookupInd][0];
        float energy = 0.f, tfallWeightedEnergySum = 0.f, triseWeightedEnergySum = 0.f;
        float tfallWeightedSum = 0.f, triseWeightedSum = 0.f;
        float tfallSum = 0.f, triseSum = 0.f;
        unsigned tfallCount = 0, triseCount = 0;

        for (unsigned ianode=0; ianode<2; ++ianode)
        {
            const HFQIE10Info* anodeInfo = prehit.getHFQIE10Info(ianode);
            if (anodeInfo && weights[ianode] > 0.f)
            {
                const float weightedEnergy = weights[ianode]*anodeInfo->energy();
                energy += weightedEnergy;

                if (isTimingReliable[ianode] &&
                    states[ianode] != HFAnodeStatus::FAILED_TIMING)
                {
                    float trise = anodeInfo->timeRising();
                    if (!HcalSpecialTimes::isSpecial(trise))
                    {
                        trise += timeShift_;
                        triseSum += trise;
                        ++triseCount;
                        if (weightedEnergy > 0.f)
                        {
                            triseWeightedSum += trise*weightedEnergy;
                            triseWeightedEnergySum += weightedEnergy;
                        }
                    }

                    float tfall = anodeInfo->timeFalling();
                    if (!HcalSpecialTimes::isSpecial(tfall))
                    {
                        tfall += timeShift_;
                        tfallSum += tfall;
                        ++tfallCount;
                        if (weightedEnergy > 0.f)
                        {
                            tfallWeightedSum += tfall*weightedEnergy;
                            tfallWeightedEnergySum += weightedEnergy;
                        }
                    }
                }
            }
        }

        bool triseFromTDC = false;
        const float trise = build_rechit_time(
            triseWeightedEnergySum, triseWeightedSum, triseSum,
            triseCount, triseIfNoTDC_, &triseFromTDC);

        bool tfallFromTDC = false;
        const float tfall = build_rechit_time(
            tfallWeightedEnergySum, tfallWeightedSum, tfallSum,
            tfallCount, tfallIfNoTDC_, &tfallFromTDC);

        rh = HFRecHit(prehit.id(), energy, trise, tfall);
        HFRecHitAuxSetter::setAux(prehit, states, soiPhase_, &rh);

        // Set the "timing from TDC" flag
        const uint32_t flag = triseFromTDC ? 1U : 0U;
        rh.setFlagField(flag, HcalPhase1FlagLabels::TimingFromTDC);
    }

    return rh;
}
