#include "DataFormats/HcalRecHit/interface/HFQIE10Info.h"

HFQIE10Info::HFQIE10Info()
    : charge_(0.f),
      energy_(0.f),
      timeRising_(0.f),
      timeFalling_(-1.f),
      raw_{INVALID_RAW, INVALID_RAW, INVALID_RAW},
      nRaw_(0)
{
}

HFQIE10Info::HFQIE10Info(const HcalDetId& id,
                         const float i_charge, const float i_energy,
                         const float i_timeRising, const float i_timeFalling,
                         const raw_type* rawData, const unsigned nData)
    : id_(id),
      charge_(i_charge),
      energy_(i_energy),
      timeRising_(i_timeRising),
      timeFalling_(i_timeFalling),
      raw_{INVALID_RAW, INVALID_RAW, INVALID_RAW},
      nRaw_(nData)
{
    if (nRaw_ > N_RAW_MAX)
        nRaw_ = N_RAW_MAX;
    for (unsigned i=0; i<nRaw_; ++i)
        raw_[i] = rawData[i];
}
