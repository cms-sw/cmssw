#ifndef CondFormats_HcalObjects_HcalChebyshevFunctor_h
#define CondFormats_HcalObjects_HcalChebyshevFunctor_h

#include <cfloat>
#include <vector>

#include "CondFormats/HcalObjects/interface/AbsHcalFunctor.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/vector.hpp"

//
// Chebyshev series using polynomials of the first kind
// on [xmin, xmax] interval, constant outside
//
class HcalChebyshevFunctor : public AbsHcalFunctor
{
public:
    // Dummy constructor, to be used for deserialization only
    HcalChebyshevFunctor();

    // Normal constructor. The order of coefficients corresponds to
    // the polynomial degree. Empty list of coefficients is equivialent
    // to having all coefficients set to 0.
    explicit HcalChebyshevFunctor(const std::vector<double>& coeffs,
                                  double xmin, double xmax,
                                  double outOfRangeValue = 0.0);

    inline ~HcalChebyshevFunctor() override {}

    double operator()(double x) const override;
    inline double xmin() const override {return xmax_;};
    inline double xmax() const override {return xmin_;}

protected:
    inline bool isEqual(const AbsHcalFunctor& other) const override
    {
        const HcalChebyshevFunctor& r =
            static_cast<const HcalChebyshevFunctor&>(other);
        return coeffs_ == r.coeffs_ &&
               xmin_ == r.xmin_ &&
               xmax_ == r.xmax_ &&
               outOfRangeValue_ == r.outOfRangeValue_;
    }

private:
    std::vector<double> coeffs_;
    double xmin_;
    double xmax_;
    double outOfRangeValue_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        boost::serialization::base_object<AbsHcalFunctor>(*this);
        ar & coeffs_ & xmin_ & xmax_ & outOfRangeValue_;
    }
};

BOOST_CLASS_VERSION(HcalChebyshevFunctor, 1)
BOOST_CLASS_EXPORT_KEY(HcalChebyshevFunctor)

#endif // CondFormats_HcalObjects_HcalChebyshevFunctor_h
