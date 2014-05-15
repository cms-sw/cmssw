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
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

class GBRForest;

class EGEnergyCorrector {
  public:
    EGEnergyCorrector();
    ~EGEnergyCorrector();

    void Initialize(const edm::EventSetup &iSetup, std::string regweights, bool weightsFromDB=false);
    Bool_t IsInitialized() const { return fIsInitialized; }

    std::pair<double,double> CorrectedEnergyWithError(const reco::Photon &p, const reco::VertexCollection& vtxcol, EcalClusterLazyTools &clustertools, const edm::EventSetup &es);
    std::pair<double,double> CorrectedEnergyWithError(const reco::GsfElectron &e, const reco::VertexCollection& vtxcol, EcalClusterLazyTools &clustertools, const edm::EventSetup &es);

    std::pair<double,double> CorrectedEnergyWithErrorV3(const reco::Photon &p, const reco::VertexCollection& vtxcol, double rho, EcalClusterLazyTools &clustertools, const edm::EventSetup &es, bool applyRescale = false);
    std::pair<double,double> CorrectedEnergyWithErrorV3(const reco::GsfElectron &e, const reco::VertexCollection& vtxcol, double rho, EcalClusterLazyTools &clustertools, const edm::EventSetup &es);

  protected:
    const GBRForest *fReadereb;
    const GBRForest *fReaderebvariance;
    const GBRForest *fReaderee;
    const GBRForest *fReadereevariance;

    Bool_t fIsInitialized;
    Bool_t fOwnsForests;
    Float_t *fVals;

    EcalClusterLocal _ecalLocal;

    };


#endif
