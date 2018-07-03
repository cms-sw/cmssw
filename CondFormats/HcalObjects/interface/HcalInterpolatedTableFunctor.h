#ifndef CondFormats_HcalObjects_HcalInterpolatedTableFunctor_h
#define CondFormats_HcalObjects_HcalInterpolatedTableFunctor_h

#include "CondFormats/HcalObjects/interface/HcalPiecewiseLinearFunctor.h"

//
// Simple linear interpolator from equidistant points.
// Need O(1) operations no matter how many points are used.
//
class HcalInterpolatedTableFunctor : public AbsHcalFunctor
{
public:
    // Dummy constructor, to be used for deserialization only
    HcalInterpolatedTableFunctor();

    // The vector of values must have at least two elements
    // (for xmin and xmax).
    //
    // Argument "leftExtrapolationLinear" determines
    // whether the extrapolation to the left of xmin
    // is going to be constant or linear.
    //
    // Argument "rightExtrapolationLinear" determines
    // whether the extrapolation to the right of xmax
    // is going to be constant or linear.
    //
    HcalInterpolatedTableFunctor(const std::vector<double>& values,
                                 double xmin, double xmax,
                                 bool leftExtrapolationLinear,
                                 bool rightExtrapolationLinear);

    inline ~HcalInterpolatedTableFunctor() override {}

    double operator()(double x) const override;
    inline double xmin() const override {return xmin_;}
    inline double xmax() const override {return xmax_;}

    // Check if the interpolated values are strictly increasing or decreasing
    bool isStrictlyMonotonous() const;

    // For strictly monotonous functors,
    // we will be able to generate the inverse
    HcalPiecewiseLinearFunctor inverse() const;

protected:
    inline bool isEqual(const AbsHcalFunctor& other) const override
    {
        const HcalInterpolatedTableFunctor& r =
            static_cast<const HcalInterpolatedTableFunctor&>(other);
        return values_ == r.values_ &&
               xmin_ == r.xmin_ && xmax_ == r.xmax_ &&
               leftExtrapolationLinear_ == r.leftExtrapolationLinear_ &&
               rightExtrapolationLinear_ == r.rightExtrapolationLinear_;
    }

private:
    std::vector<double> values_;
    double xmin_;
    double xmax_;
    bool leftExtrapolationLinear_;
    bool rightExtrapolationLinear_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        boost::serialization::base_object<AbsHcalFunctor>(*this);
        ar & values_ & xmin_ & xmax_
           & leftExtrapolationLinear_
           & rightExtrapolationLinear_;
    }
};

BOOST_CLASS_VERSION(HcalInterpolatedTableFunctor, 1)
BOOST_CLASS_EXPORT_KEY(HcalInterpolatedTableFunctor)

#endif // CondFormats_HcalObjects_HcalInterpolatedTableFunctor_h
