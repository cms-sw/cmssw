#include "DataFormats/HcalRecHit/interface/HFPreRecHit.h"

HFPreRecHit::HFPreRecHit()
    : hasInfo_{false, false}
{
}

HFPreRecHit::HFPreRecHit(const HcalDetId& id, const HFQIE10Info* first,
                         const HFQIE10Info* second)
    : id_(id), hasInfo_{false, false}
{
    if (first)
    {
        hfQIE10Info_[0] = *first;
        hasInfo_[0] = true;
    }
    if (second)
    {
        hfQIE10Info_[1] = *second;
        hasInfo_[1] = true;
    }
}

float HFPreRecHit::charge() const
{
    float q = 0.f;
    for (unsigned i=0; i<2; ++i)
        if (hasInfo_[i])
            q += hfQIE10Info_[i].charge();
    return q;
}

float HFPreRecHit::energy() const
{
    float e = 0.f;
    for (unsigned i=0; i<2; ++i)
        if (hasInfo_[i])
            e += hfQIE10Info_[i].energy();
    return e;
}

const HFQIE10Info* HFPreRecHit::getHFQIE10Info(const unsigned index) const
{
    if (index < 2 && hasInfo_[index])
        return &hfQIE10Info_[index];
    else
        return nullptr;
}

std::pair<float,bool> HFPreRecHit::chargeAsymmetry(const float chargeThreshold) const
{
    std::pair<float,bool> result(0.f, false);
    if (hasInfo_[0] && hasInfo_[1])
    {
        const float q0 = hfQIE10Info_[0].charge();
        const float q1 = hfQIE10Info_[1].charge();
        const float qsum = q0 + q1;
        if (qsum > 0.f && qsum >= chargeThreshold)
        {
            result.first = (q1 - q0)/qsum;
            result.second = true;
        }
    }
    return result;
}

std::pair<float,bool> HFPreRecHit::energyAsymmetry(const float energyThreshold) const
{
    std::pair<float,bool> result(0.f, false);
    if (hasInfo_[0] && hasInfo_[1])
    {
        const float e0 = hfQIE10Info_[0].energy();
        const float e1 = hfQIE10Info_[1].energy();
        const float esum = e0 + e1;
        if (esum > 0.f && esum >= energyThreshold)
        {
            result.first = (e1 - e0)/esum;
            result.second = true;
        }
    }
    return result;
}
