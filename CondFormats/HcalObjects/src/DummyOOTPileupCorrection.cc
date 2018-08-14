#include "FWCore/Utilities/interface/Exception.h"

#include "boost/serialization/export.hpp"

#include "CondFormats/HcalObjects/interface/DummyOOTPileupCorrection.h"

void DummyOOTPileupCorrection::apply(
    const HcalDetId& /* id */,
    const double* inputCharge, const unsigned lenInputCharge,
    const BunchXParameter* /* bcParams */, unsigned /* lenBcParams */,
    unsigned /* firstTimeSlice */, unsigned /* nTimeSlices */,
    double* correctedCharge, const unsigned lenCorrectedCharge,
    bool* pulseShapeCorrApplied, bool* leakCorrApplied,
    bool* readjustTiming) const
{
    // Check the arguments
    if (inputCharge == nullptr || correctedCharge == nullptr ||
        lenCorrectedCharge < lenInputCharge ||
        pulseShapeCorrApplied == nullptr || leakCorrApplied == nullptr ||
        readjustTiming == nullptr)
        throw cms::Exception(
            "Invalid arguments in DummyOOTPileupCorrection::apply");

    // Perform the correction
    for (unsigned i=0; i<lenInputCharge; ++i)
        correctedCharge[i] = scale_ * inputCharge[i];

    // Tell the code that runs after this which additional
    // corrections should be discarded
    *pulseShapeCorrApplied = false;
    *leakCorrApplied = false;

    // Tell the code that runs after this whether corrected
    // amplitudes should be used for timing calculations
    *readjustTiming = false;
}

bool DummyOOTPileupCorrection::isEqual(const AbsOOTPileupCorrection& otherBase) const
{
    // Note the use of static_cast rather than dynamic_cast below.
    // static_cast works faster and it is guaranteed to succeed here.
    const DummyOOTPileupCorrection& r = 
        static_cast<const DummyOOTPileupCorrection&>(otherBase);
    return descr_ == r.descr_ && scale_ == r.scale_;
}

BOOST_CLASS_EXPORT_IMPLEMENT(DummyOOTPileupCorrection)
