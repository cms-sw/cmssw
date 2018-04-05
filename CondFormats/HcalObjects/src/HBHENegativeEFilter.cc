#include <cmath>
#include <climits>

#include "CondFormats/HcalObjects/interface/HBHENegativeEFilter.h"

HBHENegativeEFilter::HBHENegativeEFilter(
    const std::vector<PiecewiseScalingPolynomial>& a1vec,
    const std::vector<PiecewiseScalingPolynomial>& a2vec,
    const std::vector<uint32_t>& iEtaLimits,
    const std::vector<std::pair<double,double> >& cut,
    const double minCharge,
    const unsigned firstTimeSlice,
    const unsigned lastTimeSlice)
    : a1v_(a1vec),
      a2v_(a2vec),
      iEtaLimits_(iEtaLimits),
      cut_(cut),
      minCharge_(minCharge),
      tFirst_(firstTimeSlice),
      tLast_(lastTimeSlice)
{
    if (!validate()) throw cms::Exception(
        "Invalid HBHENegativeEFilter constructor arguments");
}

bool HBHENegativeEFilter::validate() const
{
    if (cut_.empty())
        return true;

    const std::size_t nLimits(iEtaLimits_.size());
    if (nLimits >= static_cast<std::size_t>(UINT_MAX - 1U))
        return false;
    for (std::size_t i=1; i<nLimits; ++i)
        if (!(iEtaLimits_[i-1] < iEtaLimits_[i]))
            return false;

    if (a1v_.size() != nLimits + 1)
        return false;
    if (a2v_.size() != nLimits + 1)
        return false;

    const std::size_t sz = cut_.size();
    if (sz >= static_cast<std::size_t>(UINT_MAX - 1U))
        return false;
    for (std::size_t i=1; i<sz; ++i)
        if (!(cut_[i-1U].first < cut_[i].first))
            return false;

    if (tFirst_ < 2U)
        return false;
    if (!(tFirst_ <= tLast_))
        return false;

    return true;
}

bool HBHENegativeEFilter::operator==(const HBHENegativeEFilter& r) const
{
    if (cut_.empty() && r.cut_.empty())
        return true;
    else
        return a1v_ == r.a1v_ &&
               a2v_ == r.a2v_ &&
               iEtaLimits_ == r.iEtaLimits_ &&
               cut_ == r.cut_ &&
               minCharge_ == r.minCharge_ &&
               tFirst_ == r.tFirst_ &&
               tLast_ == r.tLast_;
}

unsigned HBHENegativeEFilter::getEtaIndex(const HcalDetId& id) const
{
    const unsigned nLimits = iEtaLimits_.size();
    unsigned which(0U);
    if (nLimits)
    {
        const uint32_t uEta = std::abs(id.ieta());
        const uint32_t* limits(&iEtaLimits_[0]);
        for (; which<nLimits; ++which)
            if (uEta < limits[which])
                break;
    }
    return which;
}

bool HBHENegativeEFilter::checkPassFilter(const HcalDetId& id,
                                          const double* ts, const unsigned lenTS) const
{
    bool passes = true;
    const unsigned sz = cut_.size();
    if (sz)
    {
        double chargeInWindow = 0.0;
        for (unsigned i=tFirst_; i<=tLast_ && i<lenTS; ++i)
            chargeInWindow += ts[i];
        if (chargeInWindow >= minCharge_)
        {
            // Figure out the cut value for this charge
            const std::pair<double,double>* cut = &cut_[0];
            double cutValue = cut[0].second;
            if (sz > 1U)
            {
                // First point larger than charge
                unsigned largerPoint = 0;
                for (; cut[largerPoint].first <= chargeInWindow; ++largerPoint) {}

                // Constant extrapolation beyond min and max coords
                if (largerPoint >= sz)
                    cutValue = cut[sz - 1U].second;
                else if (largerPoint)
                {
                    const double slope = (cut[largerPoint].second - cut[largerPoint-1U].second)/
                                         (cut[largerPoint].first - cut[largerPoint-1U].first);
                    cutValue = cut[largerPoint-1U].second + slope*
                        (chargeInWindow - cut[largerPoint-1U].first);
                }
            }

            // Compare the modified time slices with the cut
            const unsigned itaIdx = getEtaIndex(id);
            const PiecewiseScalingPolynomial& a1(a1v_[itaIdx]);
            const PiecewiseScalingPolynomial& a2(a2v_[itaIdx]);

            for (unsigned i=tFirst_; i<=tLast_ && i<lenTS && passes; ++i)
            {
                const double ecorr = ts[i] - a1(ts[i-1U]) - a2(ts[i-2U]);
                passes = ecorr >= cutValue;
            }
        }
    }
    return passes;
}
