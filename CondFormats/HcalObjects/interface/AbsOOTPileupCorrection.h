#ifndef CondFormats_HcalObjects_AbsOOTPileupCorrection_h
#define CondFormats_HcalObjects_AbsOOTPileupCorrection_h

#include <typeinfo>

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/export.hpp"

// Archive headers are needed here for the serialization registration to work.
// <cassert> is needed for the archive headers to work.
#if !defined(__GCCXML__)
#include <cassert>
#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"
#endif /* #if !defined(__GCCXML__) */

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

// The contents of the "BunchXParameter" class are not well
// defined yet. Potentially, they could be as simple as 0/1 flag
// for beam gap/normal bunch crossing or they can include bunch
// crossing lumi data. We need to figure out how to get this
// information.
class BunchXParameter;

class AbsOOTPileupCorrection
{
public:
    inline virtual ~AbsOOTPileupCorrection() {}

    // Main correction application method to be implemented by
    // derived classes. Arguments are as follows:
    //
    //  id              -- HCAL detector id.
    //
    //  inputCharge     -- This array contains pedestal-subtracted input
    //                     charge. Gain factors converting charge into
    //                     energy might already be applied by the caller.
    //                     The caler should collaborate with this class
    //                     providing the right arguments here, depending on
    //                     what "inputIsEnergy" method returns.
    //
    //  lenInputCharge  -- Length of the "inputCharge" array.
    //
    //  bcParams        -- Information about bunch crossings, in the order
    //                     corresponding to "inputCharge".
    //
    //  lenBcParams     -- Length of the "bunchParams" array. Must not be
    //                     less than lenInputCharge.
    //
    //  firstTimeSlice  -- The first time slice that will be used for energy
    //                     reconstruction.
    //
    //  nTimeSlices     -- Number of time slices that will be used for energy
    //                     reconstruction. The sum firstTimeSlice + nTimeSlices
    //                     must not exceed lenInputCharge.
    //
    //  correctedCharge -- Array of corrected charges (or energies), to be
    //                     filled on output.
    //
    //  lenCorrectedCharge -- Length of the "correctedCharge" array. Must not
    //                     be less than lenInputCharge.
    //
    //  pulseShapeCorrApplied -- *pulseShapeCorrApplied should be set "true"
    //                     if the algorithm effectively performs the phase-based
    //                     amplitude correction, so that additional correction
    //                     of this type is not needed. If additional correction
    //                     has to be applied, this flag should be set "false".
    //
    //  leakCorrApplied -- *leakCorrApplied should be set "true" if the
    //                     algorithm effectively performs a correction for
    //                     charge leakage into the time slice before the
    //                     first one used for energy reconstruction. If not,
    //                     this flag should be set "false".
    //
    //  readjustTiming  -- *readjustTiming should be set "true" if one should
    //                     use OOT pileup corrected energies to derive hit time.
    //                     To use the original, uncorrected energies set this
    //                     to "false".
    //
    // Some of the input arguments may be ignored by derived classes.
    //
    virtual void apply(const HcalDetId& id,
                       const double* inputCharge, unsigned lenInputCharge,
                       const BunchXParameter* bcParams, unsigned lenBcParams,
                       unsigned firstTimeSlice, unsigned nTimeSlices,
                       double* correctedCharge, unsigned lenCorrectedCharge,
                       bool* pulseShapeCorrApplied, bool* leakCorrApplied,
                       bool* readjustTiming) const = 0;

    // Another method to be overriden by derived classes. This method
    // tells whether fC -> GeV conversion should be performed before
    // calling the "apply" method or after. If "true" is returned,
    // it is assumed that the input is in GeV (this also takes into
    // account potentially different gains per cap id).
    virtual bool inputIsEnergy() const = 0;

    // Comparison operators. Note that they are not virtual and should
    // not be overriden by derived classes. These operators are very
    // useful for I/O testing.
    inline bool operator==(const AbsOOTPileupCorrection& r) const
        {return (typeid(*this) == typeid(r)) && this->isEqual(r);}
    inline bool operator!=(const AbsOOTPileupCorrection& r) const
        {return !(*this == r);}

protected:
    // Method needed to compare objects for equality.
    // Must be implemented by derived classes.
    virtual bool isEqual(const AbsOOTPileupCorrection&) const = 0;

private:
    friend class boost::serialization::access;
    template <typename Ar> 
    inline void serialize(Ar& ar, unsigned /* version */) {}
};

BOOST_CLASS_EXPORT_KEY(AbsOOTPileupCorrection)

#endif // CondFormats_HcalObjects_AbsOOTPileupCorrection_h
