#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryBDTG_H
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryBDTG_H

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "CommonTools/MVAUtils/interface/TMVAZipReader.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TMVA/Reader.h"

#include <string>
#include <memory>

template <typename DetIdT>
class EcalDeadChannelRecoveryBDTG {
public:
  EcalDeadChannelRecoveryBDTG();
  ~EcalDeadChannelRecoveryBDTG();

  void setParameters(const edm::ParameterSet &ps);
  void setCaloTopology(const CaloTopology *topo) { topology_ = topo; }

  double recover(
      const DetIdT id, const EcalRecHitCollection &hit_collection, double single8Cut, double sum8Cut, bool *acceptFlag);

  void loadFile();
  void addVariables(TMVA::Reader *reader);

private:
  const CaloTopology *topology_;
  struct XtalMatrix {
    std::array<float, 9> rEn, ieta, iphi;
    float sumE8;
  };

  XtalMatrix mx_;

  edm::FileInPath bdtWeightFileNoCracks_;
  edm::FileInPath bdtWeightFileCracks_;

  std::unique_ptr<TMVA::Reader> readerNoCrack;
  std::unique_ptr<TMVA::Reader> readerCrack;
};

#endif
