#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryAlgos_HH
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryAlgos_HH

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryBDTG.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

template <typename DetIdT>
class EcalDeadChannelRecoveryAlgos {
public:
  void setParameters(const edm::ParameterSet &ps);
  void setCaloTopology(const CaloTopology *topology);
  float correct(const DetIdT id,
                const EcalRecHitCollection &hit_collection,
                std::string algo,
                double single8Cut,
                double sum8Cut,
                bool *accFlag);

private:
  EcalDeadChannelRecoveryBDTG<DetIdT> bdtg_;
};
#endif
