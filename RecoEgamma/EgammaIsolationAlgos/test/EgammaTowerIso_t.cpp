#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

int main() {
  CaloTowerCollection towers;

  float cutEx[2]{1.f, 2.f}, cutIn[2]{0.5f, 0.5f};
  EgammaTowerIsolationNew<2> iso(cutEx, cutIn, towers);

  reco::SuperCluster cand;
  EgammaTowerIsolationNew<2>::Sum sum;
  iso.compute(
      true, sum, cand, static_cast<CaloTowerDetId const*>(nullptr), static_cast<CaloTowerDetId const*>(nullptr));

  return 0;
}
