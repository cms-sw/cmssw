#include <cmath>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFFlexibleTimeCheck.h"

unsigned HFFlexibleTimeCheck::determineAnodeStatus(
    const unsigned ianode, const HFQIE10Info& anode,
    bool* isTimingReliable) const
{
    // Return quickly if this anode has a dataframe error
    if (!anode.isDataframeOK())
        return HFAnodeStatus::HARDWARE_ERROR;

    // Require minimum charge for reliable timing measurement
    const float charge = anode.charge();
    const float minCharge = ianode ? pmtInfo_->minCharge1() :
                                     pmtInfo_->minCharge0();
    if (charge < minCharge)
    {
        *isTimingReliable = false;
        return HFAnodeStatus::OK;
    }

    // Check the rise time information
    float trise = anode.timeRising();
    if (HcalSpecialTimes::isSpecial(trise))
        return HFAnodeStatus::FAILED_TIMING;
    trise += timeShift();

    // Figure out the timing cuts
    const AbsHcalFunctor& minTimeShape = pmtInfo_->cut(
        ianode ? HFPhase1PMTData::T_1_MIN : HFPhase1PMTData::T_0_MIN);
    const AbsHcalFunctor& maxTimeShape = pmtInfo_->cut(
        ianode ? HFPhase1PMTData::T_1_MAX : HFPhase1PMTData::T_0_MAX);

    // Apply the timing cuts
    if (minTimeShape(charge) <= trise && trise <= maxTimeShape(charge))
        return HFAnodeStatus::OK;
    else
        return HFAnodeStatus::FAILED_TIMING;
}

HFRecHit HFFlexibleTimeCheck::reconstruct(const HFPreRecHit& prehit,
                                          const HcalCalibrations& calibs,
                                          const bool flaggedBadInDB[2],
                                          const bool expectSingleAnodePMT)
{
    // The algorithm must be configured by now
    if (!algoConf_)
        throw cms::Exception("HFPhase1BadConfig")
            << "In HFFlexibleTimeCheck::reconstruct: algorithm is not configured";

    // Fetch the configuration for this PMT
    pmtInfo_ = &algoConf_->at(prehit.id());
    
    // Run the algorithm from the base class
    HFRecHit rh = HFSimpleTimeCheck::reconstruct(
        prehit, calibs, flaggedBadInDB, expectSingleAnodePMT);

    // Check the charge asymmetry
    bool passesAsymmetryCut = true;
    const HFQIE10Info* first = prehit.getHFQIE10Info(0U);
    const HFQIE10Info* second = prehit.getHFQIE10Info(1U);
    if (first && second)
    {
        const float q1 = first->charge();
        const float q2 = second->charge();
        const float qsum = q1 + q2;
        if (qsum > 0.f && qsum >= pmtInfo_->minChargeAsymm())
        {
            const float asymm = (q2 - q1)/qsum;
            const float minAsymm = (pmtInfo_->cut(HFPhase1PMTData::ASYMM_MIN))(qsum);
            const float maxAsymm = (pmtInfo_->cut(HFPhase1PMTData::ASYMM_MAX))(qsum);
            passesAsymmetryCut = minAsymm <= asymm && asymm <= maxAsymm;
        }
    }

    if (!passesAsymmetryCut)
    {
        // Set the relevant rechit flag
    }

    return rh;
}
