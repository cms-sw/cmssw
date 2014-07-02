#ifndef CondFormats_HcalObjects_PiecewiseScalingPolynomial_h_
#define CondFormats_HcalObjects_PiecewiseScalingPolynomial_h_

#include "FWCore/Utilities/interface/Exception.h"

#include "boost/serialization/vector.hpp"
#include "boost/serialization/version.hpp"

class PiecewiseScalingPolynomial
{
public:
    inline PiecewiseScalingPolynomial() {}

    PiecewiseScalingPolynomial(
        const std::vector<std::vector<double> >& coeffs,
        const std::vector<double>& limits);

    inline double operator()(const double x) const
    {
        double scale(0.0);
        if (x > 0.0)
        {
            const unsigned nLimits(limits_.size());
            if (nLimits)
            {
                const double* limits(&limits_[0]);
                unsigned which(0U);
                for (; which<nLimits; ++which)
                    if (x < limits[which])
                        break;
                const std::vector<double>& c(coeffs_[which]);
                const double* a = &c[0];
                for (int deg = c.size()-1; deg >= 0; --deg)
                {
                    scale *= x;
                    scale += a[deg];
                }
            }
        }
        return scale*x;
    }

    inline bool operator==(const PiecewiseScalingPolynomial& r) const
        {return coeffs_ == r.coeffs_ && limits_ == r.limits_;}

    inline bool operator!=(const PiecewiseScalingPolynomial& r) const
        {return !(*this == r);}

private:
    bool validate() const;

    std::vector<std::vector<double> > coeffs_;
    std::vector<double> limits_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void save(Archive & ar, const unsigned /* version */) const
    {
        if (!validate()) throw cms::Exception(
            "In PiecewiseScalingPolynomial::save: invalid data");
        ar & coeffs_ & limits_;
    }

    template<class Archive>
    inline void load(Archive & ar, const unsigned /* version */)
    {
        ar & coeffs_ & limits_;
        if (!validate()) throw cms::Exception(
            "In PiecewiseScalingPolynomial::load: invalid data");
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

BOOST_CLASS_VERSION(PiecewiseScalingPolynomial, 1)

#endif // CondFormats_HcalObjects_PiecewiseScalingPolynomial_h_
