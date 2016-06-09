#include <cstring>
#include <climits>

#include "RecoLocalCalo/HcalRecAlgos/interface/HFSimpleTimeCheck.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFRecHitAuxSetter.h"

HFSimpleTimeCheck::HFSimpleTimeCheck(const std::pair<float,float> tlimits[2],
                                     const float energyWeights[2*HFAnodeStatus::N_POSSIBLE_STATES-1][2],
                                     const bool rejectAllFailures)
    : rejectAllFailures_(rejectAllFailures)
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
        const QIE10DataFrame::Sample s(anode.getRaw(nRaw - 1));
        if (!s.ok())
            return HFAnodeStatus::HARDWARE_ERROR;
    }

    // Check the time limits
    const float trise = anode.timeRising();
    if (tlimits_[ianode].first <= trise && trise <= tlimits_[ianode].second)
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
                                        const bool flaggedBadInDB[2])
{
    HFRecHit rh;

    // Determine the status of each anode
    unsigned states[2] = {HFAnodeStatus::NOT_READ_OUT, HFAnodeStatus::NOT_READ_OUT};
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
        unsigned anodeCount = 0;

        for (unsigned ianode=0; ianode<2; ++ianode)
        {
            const HFQIE10Info* anodeInfo = prehit.getHFQIE10Info(ianode);
            if (anodeInfo)
            {
                const float weightedEnergy = weights[ianode]*anodeInfo->energy();
                energy += weightedEnergy;
                const float trising = anodeInfo->timeRising();
                tsum += trising;
                const float tfalling = anodeInfo->timeFalling();
                tfallsum += tfalling;

                if (weightedEnergy > 0.f)
                {
                    weightedEnergySum += weightedEnergy;
                    t += trising*weightedEnergy;
                    tfall += tfalling*weightedEnergy;
                }

                ++anodeCount;
            }
        }

        if (weightedEnergySum > 0.f)
        {
            // Normally, determine TDC rise and fall time
            // by using energy-weighted anode values
            t /= weightedEnergySum;
            tfall /= weightedEnergySum;
        }
        else
        {
            // But if energies are negative, use simple averages
            t = tsum/anodeCount;
            tfall = tfallsum/anodeCount;
        }

        rh = HFRecHit(prehit.id(), energy, t, tfall);
        HFRecHitAuxSetter::setAux(prehit, states, &rh);
    }

    return rh;
}
