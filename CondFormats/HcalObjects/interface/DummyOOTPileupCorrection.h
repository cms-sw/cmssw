#ifndef CondFormats_HcalObjects_DummyOOTPileupCorrection_h
#define CondFormats_HcalObjects_DummyOOTPileupCorrection_h

#include <string>

#include "boost/serialization/version.hpp"

#include "CondFormats/HcalObjects/interface/AbsOOTPileupCorrection.h"

class DummyOOTPileupCorrection : public AbsOOTPileupCorrection
{
public:
    // Constructor
    inline DummyOOTPileupCorrection(const std::string& itemDescription,
                                    const double scale)
        : descr_(itemDescription), scale_(scale) {}

    // Destructor
    inline ~DummyOOTPileupCorrection() override {}

    // Inspectors
    inline const std::string& description() const {return descr_;}
    inline double getScale() const {return scale_;}

    // Main correction function
    void apply(const HcalDetId& id,
               const double* inputCharge, unsigned lenInputCharge,
               const BunchXParameter* bcParams, unsigned lenBcParams,
               unsigned firstTimeSlice, unsigned nTimeSlices,
               double* correctedCharge, unsigned lenCorrectedCharge,
               bool* pulseShapeCorrApplied, bool* leakCorrApplied,
               bool* readjustTiming) const override;

    // Are we using charge or energy?
    inline bool inputIsEnergy() const override {return false;}

protected:
    // Comparison function must be implemented
    bool isEqual(const AbsOOTPileupCorrection& otherBase) const override;

public:
    // Default constructor needed for serialization.
    // Do not use in application code.
    inline DummyOOTPileupCorrection() {}

private:
    std::string descr_;
    double scale_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        boost::serialization::base_object<AbsOOTPileupCorrection>(*this);
        ar & descr_ & scale_;
    }
};

BOOST_CLASS_VERSION(DummyOOTPileupCorrection, 1)
BOOST_CLASS_EXPORT_KEY(DummyOOTPileupCorrection)

#endif // CondFormats_HcalObjects_DummyOOTPileupCorrection_h
