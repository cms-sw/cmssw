#ifndef CondFormats_HcalObjects_HcalConstFunctor_h
#define CondFormats_HcalObjects_HcalConstFunctor_h

#include "CondFormats/HcalObjects/interface/AbsHcalFunctor.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"

//
// A functor returning a constant value
//
class HcalConstFunctor : public AbsHcalFunctor
{
public:
    // Dummy constructor, to be used for deserialization only
    HcalConstFunctor();

    // Normal constructor
    explicit HcalConstFunctor(const double value);

    inline ~HcalConstFunctor() override {}

    double operator()(double x) const override;

protected:
    inline bool isEqual(const AbsHcalFunctor& other) const override
    {
        const HcalConstFunctor& r = static_cast<const HcalConstFunctor&>(other);
        return value_ == r.value_;
    }

private:
    double value_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        boost::serialization::base_object<AbsHcalFunctor>(*this);
        ar & value_;
    }
};

BOOST_CLASS_VERSION(HcalConstFunctor, 1)
BOOST_CLASS_EXPORT_KEY(HcalConstFunctor)

#endif // CondFormats_HcalObjects_HcalConstFunctor_h
