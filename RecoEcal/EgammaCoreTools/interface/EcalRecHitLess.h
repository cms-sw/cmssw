#ifndef RecoECAL_ECALClusters_EcalRecHitLess_h
#define RecoECAL_ECALClusters_EcalRecHitLess_h

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

// Less than operator for sorting EcalRecHits according to energy.
class EcalRecHitLess : public std::binary_function<EcalRecHit, EcalRecHit, bool>
{
 public:
  bool operator()(EcalRecHit x, EcalRecHit y)
    {
      return (x.energy() > y.energy());
    }
};

#endif

