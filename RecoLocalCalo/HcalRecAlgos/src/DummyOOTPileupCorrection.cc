#include "RecoLocalCalo/HcalRecAlgos/interface/DummyOOTPileupCorrection.h"

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

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
    if (inputCharge == 0 || correctedCharge == 0 ||
        lenCorrectedCharge < lenInputCharge ||
        pulseShapeCorrApplied == 0 || leakCorrApplied == 0 ||
        readjustTiming == 0)
        throw cms::Exception("InvalidArgument")
            << "Invalid arguments in DummyOOTPileupCorrection::apply\n";

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

bool DummyOOTPileupCorrection::write(std::ostream& of) const
{
    gs::write_string(of, descr_);
    gs::write_pod(of, scale_);
    return !of.fail();
}

DummyOOTPileupCorrection* DummyOOTPileupCorrection::read(const gs::ClassId& id,
                                                         std::istream& in)
{
    // Class id for the current version of this class
    static const gs::ClassId myId(gs::ClassId::makeId<DummyOOTPileupCorrection>());

    // As this class is final, it is sufficient to just check that
    // the class id is correct. For more complicated situations
    // (i.e., if class could be an intermediate base), see example
    // code coming with the "Geners" package.
    myId.ensureSameId(id);

    // Read back the object info
    std::string s;
    gs::read_string(in, &s);

    double scale;
    gs::read_pod(in, &scale);

    // Check the status of the stream
    if (in.fail()) throw gs::IOReadFailure("In DummyOOTPileupCorrection::read: "
                                           "input stream failure");
    // Return a new object on the heap
    return new DummyOOTPileupCorrection(s, scale);
}

bool DummyOOTPileupCorrection::isEqual(const AbsOOTPileupCorrection& otherBase) const
{
    // Note the use of static_cast rather than dynamic_cast below.
    // static_cast works faster and it is guaranteed to succeed here.
    const DummyOOTPileupCorrection& r = 
        static_cast<const DummyOOTPileupCorrection&>(otherBase);
    return descr_ == r.descr_ && scale_ == r.scale_;
}
