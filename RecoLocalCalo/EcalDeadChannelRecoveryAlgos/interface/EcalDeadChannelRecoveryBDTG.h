#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryBDTG_H
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryBDTG_H

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

#include <string>
#include <memory>

template <typename DetIdT>
class EcalDeadChannelRecoveryBDTG {
public:
  void setParameters(const edm::ParameterSet &ps);
  void setCaloTopology(const CaloTopology *topo) { topology_ = topo; }

  double recover(
      const DetIdT id, const EcalRecHitCollection &hit_collection, double single8Cut, double sum8Cut, bool &acceptFlag);

private:
  const CaloTopology *topology_;

  std::unique_ptr<const GBRForest> gbrForestNoCrack_;
  std::unique_ptr<const GBRForest> gbrForestCrack_;
};

#endif
