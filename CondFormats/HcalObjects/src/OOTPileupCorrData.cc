#include <climits>

#include "boost/serialization/export.hpp"

#include "CondFormats/HcalObjects/interface/OOTPileupCorrData.h"

OOTPileupCorrData::OOTPileupCorrData(
    const std::vector<OOTPileupCorrDataFcn>& corrs,
    const std::vector<uint32_t>& iEtaLimits,
    const double chargeLimit,
    const int requireFirstTS,
    const int requireNTS,
    const bool readjustTiming)
    : corrs_(corrs),
      iEtaLimits_(iEtaLimits),
      chargeLimit_(chargeLimit),
      requireFirstTS_(requireFirstTS),
      requireNTS_(requireNTS),
      readjustTiming_(readjustTiming)
{
    if (!validate()) throw cms::Exception(
        "Invalid OOTPileupCorrData constructor arguments");
}

bool OOTPileupCorrData::validate() const
{
    const std::size_t nLimits(iEtaLimits_.size());
    if (!nLimits)
        return false;
    if (nLimits >= static_cast<std::size_t>(UINT_MAX))
        return false;
    for (std::size_t i=0; i<nLimits-1; ++i)
        if (!(iEtaLimits_[i] < iEtaLimits_[i+1]))
            return false;
    if (corrs_.size() != nLimits + 1)
        return false;
    return true;
}

void OOTPileupCorrData::apply(const HcalDetId& id,
    const double* inputCharge, const unsigned lenInputCharge,
    const BunchXParameter* /* bcParams */, const unsigned /* lenBcParams */,
    const unsigned firstTimeSlice, const unsigned nTimeSlices,
    double* correctedCharge, const unsigned lenCorrectedCharge,
    bool* pulseShapeCorrApplied, bool* leakCorrApplied,
    bool* readjustTiming) const
{
    // Check the arguments
    if (inputCharge == 0 || correctedCharge == 0 ||
        lenCorrectedCharge < lenInputCharge ||
        pulseShapeCorrApplied == 0 || leakCorrApplied == 0 ||
        readjustTiming == 0)
        throw cms::Exception(
            "Invalid arguments in OOTPileupCorrData::apply");

    for (unsigned i=0; i<lenInputCharge; ++i)
        correctedCharge[i] = inputCharge[i];

    // Check whether the charge corrections should actually be applied
    const bool fixCharge = (requireFirstTS_ < 0 || 
                            requireFirstTS_ == static_cast<int32_t>(firstTimeSlice)) &&
                           (requireNTS_ < 0 ||
                            requireNTS_ == static_cast<int32_t>(nTimeSlices));
    if (fixCharge)
        apply(id, correctedCharge, firstTimeSlice);

    *pulseShapeCorrApplied = false;
    *leakCorrApplied = fixCharge;
    *readjustTiming = readjustTiming_;
}

BOOST_CLASS_EXPORT_IMPLEMENT(OOTPileupCorrData)
