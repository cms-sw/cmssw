#include <climits>

#include "CondFormats/HcalObjects/interface/PiecewiseScalingPolynomial.h"

PiecewiseScalingPolynomial::PiecewiseScalingPolynomial(
    const std::vector<std::vector<double> >& coeffs,
    const std::vector<double>& limits)
    : coeffs_(coeffs),
      limits_(limits)
{
    if (!validate()) throw cms::Exception(
        "Invalid PiecewiseScalingPolynomial constructor arguments");
}

bool PiecewiseScalingPolynomial::validate() const
{
    const std::size_t nLimits(limits_.size());
    if (!nLimits)
        return false;
    if (nLimits >= static_cast<std::size_t>(UINT_MAX))
        return false;
    if (limits_[0] <= 0.0)
        return false;
    for (std::size_t i=0; i<nLimits-1; ++i)
        if (!(limits_[i] < limits_[i+1]))
            return false;
    if (coeffs_.size() != nLimits + 1)
        return false;
    for (std::size_t i=0; i<=nLimits; ++i)
    {
        if (coeffs_[i].empty())
            return false;
        if (coeffs_[i].size() >= static_cast<std::size_t>(INT_MAX))
            return false;
    }
    return true;
}
