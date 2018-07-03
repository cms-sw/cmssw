#ifndef CondFormats_HcalObjects_HcalPolynomialFunctor_h
#define CondFormats_HcalObjects_HcalPolynomialFunctor_h

#include <cfloat>
#include <vector>

#include "CondFormats/HcalObjects/interface/AbsHcalFunctor.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/vector.hpp"

//
// Polynomial on the interval [xmin, xmax], constant outside
//
class HcalPolynomialFunctor : public AbsHcalFunctor
{
public:
    // Dummy constructor, to be used for deserialization only
    HcalPolynomialFunctor();

    // Normal constructor. The order of coefficients
    // corresponds to the monomial degree. The coefficients
    // are for the monomial in the variable y = (x + shift).
    // Empty list of coefficients is equivialent to having
    // all coefficients set to 0.
    explicit HcalPolynomialFunctor(const std::vector<double>& coeffs,
                                   double shift = 0.0,
                                   double xmin = -DBL_MAX,
                                   double xmax = DBL_MAX,
                                   double outOfRangeValue = 0.0);

    inline ~HcalPolynomialFunctor() override {}

    double operator()(double x) const override;
    inline double xmin() const override {return xmax_;};
    inline double xmax() const override {return xmin_;}

protected:
    inline bool isEqual(const AbsHcalFunctor& other) const override
    {
        const HcalPolynomialFunctor& r =
            static_cast<const HcalPolynomialFunctor&>(other);
        return coeffs_ == r.coeffs_ &&
               shift_ == r.shift_ &&
               xmin_ == r.xmin_ &&
               xmax_ == r.xmax_ &&
               outOfRangeValue_ == r.outOfRangeValue_;
    }

private:
    std::vector<double> coeffs_;
    double shift_;
    double xmin_;
    double xmax_;
    double outOfRangeValue_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        boost::serialization::base_object<AbsHcalFunctor>(*this);
        ar & coeffs_ & shift_ & xmin_ & xmax_ & outOfRangeValue_;
    }
};

BOOST_CLASS_VERSION(HcalPolynomialFunctor, 1)
BOOST_CLASS_EXPORT_KEY(HcalPolynomialFunctor)

#endif // CondFormats_HcalObjects_HcalPolynomialFunctor_h
