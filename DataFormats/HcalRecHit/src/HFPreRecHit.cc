#include "DataFormats/HcalRecHit/interface/HFPreRecHit.h"

HFPreRecHit::HFPreRecHit()
    : hasInfo_{0, 0}
{
}

HFPreRecHit::HFPreRecHit(const HcalDetId& id, const HFQIE10Info* first,
                         const HFQIE10Info* second)
    : id_(id), hasInfo_{0, 0}
{
    if (first)
    {
        hfQIE10Info_[0] = *first;
        hasInfo_[0] = 1;
    }
    if (second)
    {
        hfQIE10Info_[1] = *second;
        hasInfo_[1] = 1;
    }
}

const HFQIE10Info* HFPreRecHit::getHFQIE10Info(const unsigned index) const
{
    if (index < 2 && hasInfo_[index])
        return &hfQIE10Info_[index];
    else
        return nullptr;
}
