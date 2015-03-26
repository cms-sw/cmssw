#ifndef CondFormats_HcalObjects_OOTPileupCorrDataFcn_h_
#define CondFormats_HcalObjects_OOTPileupCorrDataFcn_h_

#include "CondFormats/HcalObjects/interface/PiecewiseScalingPolynomial.h"
#include "CondFormats/HcalObjects/interface/ScalingExponential.h"

class OOTPileupCorrDataFcn
{
public:
    inline OOTPileupCorrDataFcn() {}

    inline OOTPileupCorrDataFcn(const PiecewiseScalingPolynomial& a1,
                                const PiecewiseScalingPolynomial& a2,
                                const PiecewiseScalingPolynomial& a3,
                                const ScalingExponential& a_1)
        : a1_(a1), a2_(a2), a3_(a3), a_1_(a_1) {}

    inline bool operator==(const OOTPileupCorrDataFcn& r) const
        {return a1_ == r.a1_ && a2_ == r.a2_ && a3_ == r.a3_ && a_1_ == r.a_1_;}

    inline bool operator!=(const OOTPileupCorrDataFcn& r) const
        {return !(*this == r);}

    inline void pucorrection(double *ts, const int tsTrig) const
    {
        // TS4 correction with functions a1(x) and a2(x)
        double ts20 = ts[tsTrig-2];
        double ts30 = ts[tsTrig-1] - a_1_(ts[tsTrig]); // ts[3] correction with a_1(x)
        double ts21 = ts20 > 0 ? a2_(ts20) : 0; 
        double ts22 = ts20 > 0 ? a1_(ts20) : 0;
        double ts321 = ts30 - ts22 > 0 ? a1_(ts30 - ts22) : 0; 
        ts[tsTrig] -= (ts321 + ts21); // ts[4] after pu correction
  
        // ts5 estimation from ts4
        ts[tsTrig+1] = a1_(ts[tsTrig]);
    }

    // Access the correction functions
    inline const PiecewiseScalingPolynomial& getA1() const {return a1_;}
    inline const PiecewiseScalingPolynomial& getA2() const {return a2_;}
    inline const PiecewiseScalingPolynomial& getA3() const {return a3_;}
    inline const ScalingExponential& getA_1() const {return a_1_;}

private:
    PiecewiseScalingPolynomial a1_, a2_, a3_;
    ScalingExponential a_1_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        ar & a1_ & a2_ & a3_ & a_1_;
    }
};

BOOST_CLASS_VERSION(OOTPileupCorrDataFcn, 1)

#endif // CondFormats_HcalObjects_OOTPileupCorrDataFcn_h_
