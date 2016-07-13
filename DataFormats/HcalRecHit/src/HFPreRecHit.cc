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

const HFQIE10Info* HFPreRecHit::getHFQIE10Info(const unsigned index) const
{
    if (index < 2 && hasInfo_[index])
        return &hfQIE10Info_[index];
    else
        return nullptr;
}
