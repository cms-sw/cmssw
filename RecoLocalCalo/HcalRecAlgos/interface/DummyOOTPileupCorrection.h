#ifndef RecoLocalCalo_HcalRecAlgos_DummyOOTPileupCorrection_h
#define RecoLocalCalo_HcalRecAlgos_DummyOOTPileupCorrection_h

// This is a "final" class. No other classes should be derived from it.

#include <string>

#include "RecoLocalCalo/HcalRecAlgos/interface/AbsOOTPileupCorrection.h"

class DummyOOTPileupCorrection final : public AbsOOTPileupCorrection
{
public:
    // Constructor
    inline DummyOOTPileupCorrection(const std::string& itemDescription,
                                    const double scale)
        : descr_(itemDescription), scale_(scale) {}

    // Destructor
    inline ~DummyOOTPileupCorrection() {}

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
               bool* readjustTiming) const;

    // Are we using charge or energy?
    inline bool inputIsEnergy() const {return false;}

    // Methods related to I/O
    inline gs::ClassId classId() const {return gs::ClassId(*this);}
    bool write(std::ostream& of) const;

    static inline const char* classname() {return "DummyOOTPileupCorrection";}
    static inline unsigned version() {return 1;}
    static DummyOOTPileupCorrection* read(const gs::ClassId& id,
                                          std::istream& in);
protected:
    // Comparison function must be implemented
    bool isEqual(const AbsOOTPileupCorrection& otherBase) const;

private:
    // Disable default constructor
    DummyOOTPileupCorrection();

    std::string descr_;
    double scale_;
};

#endif // RecoLocalCalo_HcalRecAlgos_DummyOOTPileupCorrection_h
