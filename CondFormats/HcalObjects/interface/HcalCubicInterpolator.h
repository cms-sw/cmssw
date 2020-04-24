#ifndef CondFormats_HcalObjects_HcalCubicInterpolator_h
#define CondFormats_HcalObjects_HcalCubicInterpolator_h

#include <vector>
#include <tuple>

#include "CondFormats/HcalObjects/interface/AbsHcalFunctor.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/vector.hpp"

//
// Cubic Hermite spline interpolator in 1-d. See, for example,
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
//
class HcalCubicInterpolator : public AbsHcalFunctor
{
public:
    // Order: abscissa, desired value, derivative
    typedef std::tuple<double,double,double> Triple;

    // Dummy constructor, to be used for deserialization only
    HcalCubicInterpolator();

    // Normal constructor from the set of interpolated points.
    // The points will be sorted internally, so they can be
    // given in arbitrary order.
    explicit HcalCubicInterpolator(const std::vector<Triple>& points);

    inline ~HcalCubicInterpolator() override {}

    double operator()(double x) const override;
    double xmin() const override;
    double xmax() const override;

    // Cubic approximation to the inverse curve (note, not the real
    // solution of the direct cubic equation). Use at your own risc.
    HcalCubicInterpolator approximateInverse() const;

protected:
    inline bool isEqual(const AbsHcalFunctor& other) const override
    {
        const HcalCubicInterpolator& r =
            static_cast<const HcalCubicInterpolator&>(other);
        return abscissae_ == r.abscissae_ &&
               values_ == r.values_ &&
               derivatives_ == r.derivatives_;
    }

private:
    std::vector<double> abscissae_;
    std::vector<double> values_;
    std::vector<double> derivatives_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        boost::serialization::base_object<AbsHcalFunctor>(*this);
        ar & abscissae_ & values_ & derivatives_;
    }
};

BOOST_CLASS_VERSION(HcalCubicInterpolator, 1)
BOOST_CLASS_EXPORT_KEY(HcalCubicInterpolator)

#endif // CondFormats_HcalObjects_HcalCubicInterpolator_h
