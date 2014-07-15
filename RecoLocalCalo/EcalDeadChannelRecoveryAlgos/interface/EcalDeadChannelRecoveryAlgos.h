#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryAlgos_HH
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_ECalDeadChannelRecoveryAlgos_HH

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <string>

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryNN.h"

template <typename DetIdT> class EcalDeadChannelRecoveryAlgos {
 public:
  void setCaloTopology(const CaloTopology *topology);
  EcalRecHit correct(const DetIdT id,
                     const EcalRecHitCollection &hit_collection,
                     std::string algo, double Sum8Cut, bool *AccFlag);

 private:
  EcalDeadChannelRecoveryNN<DetIdT> nn;
};
#endif
