#ifndef CondFormats_HcalObjects_HBHENegativeEFilter_h_
#define CondFormats_HcalObjects_HBHENegativeEFilter_h_

#include <vector>
#include <utility>
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/cstdint.hpp"
#include "boost/serialization/utility.hpp"
#include "boost/serialization/access.hpp"
#include "boost/serialization/split_member.hpp"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/PiecewiseScalingPolynomial.h"

class HBHENegativeEFilter
{
public:
    inline HBHENegativeEFilter() : minCharge_(0.), tFirst_(0), tLast_(0) {}

    // If the vector of cuts is empty, the filter will be disabled
    HBHENegativeEFilter(const std::vector<PiecewiseScalingPolynomial>& a1vec,
                        const std::vector<PiecewiseScalingPolynomial>& a2vec,
                        const std::vector<uint32_t>& iEtaLimits,
                        const std::vector<std::pair<double,double> >& cut,
                        double minCharge, unsigned firstTimeSlice,
                        unsigned lastTimeSlice);

    // Does the sequence of time slices pass the filter?
    bool checkPassFilter(const HcalDetId& id,
                         const double* ts, unsigned lenTS) const;

    // Examing various filter data elements
    inline const PiecewiseScalingPolynomial& getA1(const HcalDetId& id) const
        {return a1v_.at(getEtaIndex(id));}
    inline const PiecewiseScalingPolynomial& getA2(const HcalDetId& id) const
        {return a2v_.at(getEtaIndex(id));}
    inline const std::vector<uint32_t>& getEtaLimits() const
        {return iEtaLimits_;}
    inline const std::vector<std::pair<double,double> >& getCut() const
        {return cut_;}
    inline double getMinCharge() const {return minCharge_;}
    inline unsigned getFirstTimeSlice() const {return tFirst_;}
    inline unsigned getLastTimeSlice() const {return tLast_;}
    inline bool isEnabled() const {return !cut_.empty();}

    // Comparison operators
    bool operator==(const HBHENegativeEFilter& r) const;
    inline bool operator!=(const HBHENegativeEFilter& r) const
        {return !(*this == r);}

private:
    unsigned getEtaIndex(const HcalDetId& id) const;
    bool validate() const;

    std::vector<PiecewiseScalingPolynomial> a1v_;
    std::vector<PiecewiseScalingPolynomial> a2v_;
    std::vector<uint32_t> iEtaLimits_;
    std::vector<std::pair<double,double> > cut_;
    double minCharge_;
    uint32_t tFirst_;
    uint32_t tLast_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void save(Archive & ar, const unsigned /* version */) const
    {
        if (!validate()) throw cms::Exception(
            "In HBHENegativeEFilter::save: invalid data");
        ar & a1v_ & a2v_ & iEtaLimits_ & cut_ & minCharge_ & tFirst_ & tLast_;
    }

    template<class Archive>
    inline void load(Archive & ar, const unsigned /* version */)
    {
        ar & a1v_ & a2v_ & iEtaLimits_ & cut_ & minCharge_ & tFirst_ & tLast_;
        if (!validate()) throw cms::Exception(
            "In HBHENegativeEFilter::load: invalid data");
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

BOOST_CLASS_VERSION(HBHENegativeEFilter, 1)

#endif // CondFormats_HcalObjects_HBHENegativeEFilter_h_
