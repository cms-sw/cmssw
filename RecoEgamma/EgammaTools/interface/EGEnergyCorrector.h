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

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include <array>

class GBRForest;

class EGEnergyCorrector {
public:
  ~EGEnergyCorrector();

  void Initialize(const edm::EventSetup &iSetup, std::string regweights, bool weightsFromDB = false);
  bool IsInitialized() const { return fIsInitialized; }

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
  const GBRForest *fReadereb = nullptr;
  const GBRForest *fReaderebvariance = nullptr;
  const GBRForest *fReaderee = nullptr;
  const GBRForest *fReadereevariance = nullptr;

  bool fIsInitialized = false;
  bool fOwnsForests = false;
  std::array<float, 73> fVals;
};

#endif
