#ifndef ReciEgamma_PhotonIdentification_PhotonXGBoostEstimator_h
#define ReciEgamma_PhotonIdentification_PhotonXGBoostEstimator_h

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/XGBoost/interface/XGBooster.h"

class PhotonXGBoostEstimator {
public:
  PhotonXGBoostEstimator(const edm::FileInPath& weightsFile, int best_ntree_limit);
  ~PhotonXGBoostEstimator();

  float computeMva(float rawEnergyIn,
                   float r9In,
                   float sigmaIEtaIEtaIn,
                   float etaWidthIn,
                   float phiWidthIn,
                   float s4In,
                   float etaIn,
                   float hOvrEIn,
                   float ecalPFIsoIn) const;

private:
  std::unique_ptr<pat::XGBooster> booster_;
  int best_ntree_limit_ = -1;
  std::string config_;
};

#endif
