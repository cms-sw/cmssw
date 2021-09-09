//--------------------------------------------------------------------------------------------------
//
// EGEnergyCorrector
//
// Helper Class for applying regression-based energy corrections with optimized BDT implementation
//
// Authors: J.Bendavid
//--------------------------------------------------------------------------------------------------

#ifndef EGAMMATOOLS_EGEnergyCorrector_H
#define EGAMMATOOLS_EGEnergyCorrector_H

#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include <array>
#include <memory>

class GBRForest;

class EGEnergyCorrector {
public:
  struct Initializer {
    std::shared_ptr<const GBRForest> readereb_;
    std::shared_ptr<const GBRForest> readerebvariance_;
    std::shared_ptr<const GBRForest> readeree_;
    std::shared_ptr<const GBRForest> readereevariance_;
  };

  ~EGEnergyCorrector() = default;
  EGEnergyCorrector() = default;
  explicit EGEnergyCorrector(Initializer) noexcept;

  std::pair<double, double> CorrectedEnergyWithError(const reco::Photon &p,
                                                     const reco::VertexCollection &vtxcol,
                                                     EcalClusterLazyTools &clustertools,
                                                     CaloGeometry const &caloGeometry);

  std::pair<double, double> CorrectedEnergyWithErrorV3(const reco::Photon &p,
                                                       const reco::VertexCollection &vtxcol,
                                                       double rho,
                                                       EcalClusterLazyTools &clustertools,
                                                       CaloGeometry const &caloGeometry,
                                                       bool applyRescale = false);

protected:
  std::shared_ptr<const GBRForest> fReadereb;
  std::shared_ptr<const GBRForest> fReaderebvariance;
  std::shared_ptr<const GBRForest> fReaderee;
  std::shared_ptr<const GBRForest> fReadereevariance;

  std::array<float, 73> fVals;
};

#endif
