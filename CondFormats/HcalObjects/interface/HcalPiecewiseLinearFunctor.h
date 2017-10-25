#ifndef CondFormats_HcalObjects_HcalPiecewiseLinearFunctor_h
#define CondFormats_HcalObjects_HcalPiecewiseLinearFunctor_h

#include <vector>
#include <utility>

#include "CondFormats/HcalObjects/interface/AbsHcalFunctor.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/vector.hpp"

//
// Simple piecewise linear interpolator.
// Will invert the curve if needed.
//
class HcalPiecewiseLinearFunctor : public AbsHcalFunctor
{
public:
    // Dummy constructor, to be used for deserialization only
    HcalPiecewiseLinearFunctor();

    // Abscissae are the first elements of the pairs.
    // Interpolated values are the second elements.
    // The order of the points is arbitrary -- they will
    // be sorted internally anyway in the order of
    // increasing abscissae.
    //
    // Argument "leftExtrapolationLinear" determines
    // whether the extrapolation to the left of the smallest
    // abscissa is going to be constant or linear.
    //
    // Argument "rightExtrapolationLinear" determines
    // whether the extrapolation to the right of the largest
    // abscissa is going to be constant or linear.
    //
    HcalPiecewiseLinearFunctor(const std::vector<std::pair<double, double> >& points,
                               bool leftExtrapolationLinear,
                               bool rightExtrapolationLinear);

    inline ~HcalPiecewiseLinearFunctor() override {}

    double operator()(double x) const override;
    double xmin() const override;
    double xmax() const override;

    // Check if the interpolated values are strictly increasing or decreasing
    bool isStrictlyMonotonous() const;

    // For strictly monotonous functors,
    // we will be able to generate the inverse
    HcalPiecewiseLinearFunctor inverse() const;

protected:
    inline bool isEqual(const AbsHcalFunctor& other) const override
    {
        const HcalPiecewiseLinearFunctor& r =
            static_cast<const HcalPiecewiseLinearFunctor&>(other);
        return abscissae_ == r.abscissae_ &&
               values_ == r.values_ &&
               leftExtrapolationLinear_ == r.leftExtrapolationLinear_ &&
               rightExtrapolationLinear_ == r.rightExtrapolationLinear_;
    }

private:
    std::vector<double> abscissae_;
    std::vector<double> values_;
    bool leftExtrapolationLinear_;
    bool rightExtrapolationLinear_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        boost::serialization::base_object<AbsHcalFunctor>(*this);
        ar & abscissae_
           & values_
           & leftExtrapolationLinear_
           & rightExtrapolationLinear_;
    }
};

BOOST_CLASS_VERSION(HcalPiecewiseLinearFunctor, 1)
BOOST_CLASS_EXPORT_KEY(HcalPiecewiseLinearFunctor)

#endif // CondFormats_HcalObjects_HcalPiecewiseLinearFunctor_h
