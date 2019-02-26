#include <cmath>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"
#include "DataFormats/HcalRecHit/interface/CaloRecHitAuxSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFRecHitAuxSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFFlexibleTimeCheck.h"

// Phase 1 rechit status bit assignments
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"

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

    // Special handling of under/overshoot TDC values, as well
    // as of DLL failures
    float trise = anode.timeRising();
    if ((trise == HcalSpecialTimes::UNKNOWN_T_UNDERSHOOT &&
         charge < minChargeForUndershoot()) ||
        (trise == HcalSpecialTimes::UNKNOWN_T_OVERSHOOT &&
         charge < minChargeForOvershoot()) ||
        trise == HcalSpecialTimes::UNKNOWN_T_DLL_FAILURE)
    {
        *isTimingReliable = false;
        return HFAnodeStatus::OK;
    }

    // Check if the rise time information is meaningful
    if (HcalSpecialTimes::isSpecial(trise))
        return HFAnodeStatus::FAILED_TIMING;

    // Figure out the timing cuts for this PMT
    const AbsHcalFunctor& minTimeShape = pmtInfo_->cut(
        ianode ? HFPhase1PMTData::T_1_MIN : HFPhase1PMTData::T_0_MIN);
    const AbsHcalFunctor& maxTimeShape = pmtInfo_->cut(
        ianode ? HFPhase1PMTData::T_1_MAX : HFPhase1PMTData::T_0_MAX);

    // Apply the timing cuts
    trise += timeShift();
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

    // Fetch the algorithm configuration data for this PMT
    pmtInfo_ = &algoConf_->at(prehit.id());

    // Run the reconstruction algorithm from the base class
    HFRecHit rh = HFSimpleTimeCheck::reconstruct(
        prehit, calibs, flaggedBadInDB, expectSingleAnodePMT);

    if (rh.id().rawId())
    {
        // Check the charge asymmetry between the two anodes
        bool setAsymmetryFlag = true;
        if (!alwaysCalculatingQAsym())
        {
            using namespace CaloRecHitAuxSetter;

            const unsigned st0 = getField(rh.aux(), HFRecHitAuxSetter::MASK_STATUS,
                                          HFRecHitAuxSetter::OFF_STATUS);
            const unsigned st1 = getField(rh.getAuxHF(), HFRecHitAuxSetter::MASK_STATUS,
                                          HFRecHitAuxSetter::OFF_STATUS);
            setAsymmetryFlag = st0 == HFAnodeStatus::OK && st1 == HFAnodeStatus::OK;
        }

        if (setAsymmetryFlag)
        {
            bool passesAsymmetryCut = true;
            const std::pair<float,bool> qAsymm =
                prehit.chargeAsymmetry(pmtInfo_->minChargeAsymm());
            if (qAsymm.second)
            {
                const float q = prehit.charge();
                const float minAsymm = (pmtInfo_->cut(HFPhase1PMTData::ASYMM_MIN))(q);
                const float maxAsymm = (pmtInfo_->cut(HFPhase1PMTData::ASYMM_MAX))(q);
                passesAsymmetryCut = minAsymm <= qAsymm.first && qAsymm.first <= maxAsymm;
            }
            if (!passesAsymmetryCut)
                rh.setFlagField(1U, HcalPhase1FlagLabels::HFSignalAsymmetry);
        }
    }

    return rh;
}
