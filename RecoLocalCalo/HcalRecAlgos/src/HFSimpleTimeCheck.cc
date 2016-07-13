#include <cstring>
#include <climits>

#include "RecoLocalCalo/HcalRecAlgos/interface/HFSimpleTimeCheck.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFRecHitAuxSetter.h"

// Rechit status bit assignments
// #include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

HFSimpleTimeCheck::HFSimpleTimeCheck(const std::pair<float,float> tlimits[2],
                                     const float energyWeights[2*HFAnodeStatus::N_POSSIBLE_STATES-1][2],
                                     const unsigned i_soiPhase,
                                     const float i_timeShift,
                                     const float i_triseIfNoTDC,
                                     const float i_tfallIfNoTDC,
                                     const bool rejectAllFailures)
    : soiPhase_(i_soiPhase),
      timeShift_(i_timeShift),
      triseIfNoTDC_(i_triseIfNoTDC),
      tfallIfNoTDC_(i_tfallIfNoTDC),
      rejectAllFailures_(rejectAllFailures)
{
    tlimits_[0] = tlimits[0];
    tlimits_[1] = tlimits[1];
    float* to = &energyWeights_[0][0];
    const float* from = &energyWeights[0][0];
    memcpy(to, from, sizeof(energyWeights_));
}

unsigned HFSimpleTimeCheck::determineAnodeStatus(
    const unsigned ianode, const HFQIE10Info& anode) const
{
    // Check if there is a hardware error on any of the anodes
    const unsigned nRaw = anode.nRaw();
    if (nRaw)
    {
        bool hardwareOK = true;
        unsigned sampleToCheck = anode.soi();
        if (sampleToCheck < nRaw)
        {
            const QIE10DataFrame::Sample s(anode.getRaw(sampleToCheck));
            hardwareOK = s.ok();
        }
        else
        {
            // This should not normally happen, but we still
            // need to do something reasonable here
            for (sampleToCheck = 0; sampleToCheck < nRaw; ++sampleToCheck)
            {
                const QIE10DataFrame::Sample s(anode.getRaw(sampleToCheck));
                if (!s.ok())
                    hardwareOK = false;
            }
        }
        if (!hardwareOK)
            return HFAnodeStatus::HARDWARE_ERROR;
    }

    // Check the time limits
    float trise = anode.timeRising();
    const bool timeIsKnown = !(trise == HFQIE10Info::UNKNOWN_T_UNDERSHOOT ||
                               trise == HFQIE10Info::UNKNOWN_T_OVERSHOOT);
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

    for (unsigned ianode=0; ianode<2; ++ianode)
    {
        const HFQIE10Info* anodeInfo = prehit.getHFQIE10Info(ianode);
        if (anodeInfo)
        {
            if (flaggedBadInDB[ianode])
                states[ianode] = HFAnodeStatus::FLAGGED_BAD;
            else
                states[ianode] = determineAnodeStatus(ianode, *anodeInfo);
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
        float energy = 0.f, t = 0.f, tfall = 0.f, weightedEnergySum = 0.f;
        float tsum = 0.f, tfallsum = 0.f;
        unsigned knownTimeCount = 0;

        for (unsigned ianode=0; ianode<2; ++ianode)
        {
            const HFQIE10Info* anodeInfo = prehit.getHFQIE10Info(ianode);
            if (anodeInfo)
            {
                const float weightedEnergy = weights[ianode]*anodeInfo->energy();
                energy += weightedEnergy;
                float trising = anodeInfo->timeRising();
                const bool timeIsKnown = !(trising == HFQIE10Info::UNKNOWN_T_UNDERSHOOT ||
                                           trising == HFQIE10Info::UNKNOWN_T_OVERSHOOT);
                if (timeIsKnown)
                {
                    trising += timeShift_;
                    tsum += trising;
                    const float tfalling = anodeInfo->timeFalling() + timeShift_;
                    tfallsum += tfalling;

                    if (weightedEnergy > 0.f)
                    {
                        weightedEnergySum += weightedEnergy;
                        t += trising*weightedEnergy;
                        tfall += tfalling*weightedEnergy;
                    }

                    ++knownTimeCount;
                }
            }
        }

        // uint32_t timingFromTDC = 1;
        if (weightedEnergySum > 0.f)
        {
            // Normally, determine TDC rise and fall time
            // by using energy-weighted anode values
            t /= weightedEnergySum;
            tfall /= weightedEnergySum;
        }
        else if (knownTimeCount)
        {
            // But if energies are negative, use simple averages
            t = tsum/knownTimeCount;
            tfall = tfallsum/knownTimeCount;
        }
        else
        {
            // If we are here, neither anode provided valid time
            // information
            t = triseIfNoTDC_;
            tfall = tfallIfNoTDC_;
            // timingFromTDC = 0;
        }

        rh = HFRecHit(prehit.id(), energy, t, tfall);
        HFRecHitAuxSetter::setAux(prehit, states, soiPhase_, &rh);

        // When Phase 1 rechit status bit assignments are understood,
        // set the "timing from TDC" flag as follows:
        //
        // rh.setFlagField(timingFromTDC, HcalCaloFlagLabels::?properFlagName?);
    }

    return rh;
}
