#include "DataFormats/HcalRecHit/interface/HFQIE10Info.h"

HFQIE10Info::HFQIE10Info()
    : adc_(0),
      charge_(0.f),
      energy_(0.f),
      timeRising_(0.f),
      timeFalling_(-1.f)
{
}

HFQIE10Info::HFQIE10Info(const HcalDetId& id, const int i_adc,
                         const float i_charge, const float i_energy,
                         const float i_timeRising, const float i_timeFalling)
    : id_(id),
      adc_(i_adc),
      charge_(i_charge),
      energy_(i_energy),
      timeRising_(i_timeRising),
      timeFalling_(i_timeFalling)
{
}
