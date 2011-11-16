//--------------------------------------------------------------------------------------------------
// $Id $
//
// EGEnergyCorrector
//
// Helper Class for applying regression-based energy corrections with optimized BDT implementation
//
// Authors: J.Bendavid
//--------------------------------------------------------------------------------------------------

#ifndef EGAMMATOOLS_EGEnergyCorrector_H
#define EGAMMATOOLS_EGEnergyCorrector_H


#include "PhotonFix.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


class GBRForest;
class EcalClusterLazyTools;

class EGEnergyCorrector {
  public:
    EGEnergyCorrector();
    ~EGEnergyCorrector(); 

    void Initialize(const edm::EventSetup &iSetup, std::string regweights, bool weightsFromDB=false);
    Bool_t IsInitialized() const { return fIsInitialized; }
    
    std::pair<double,double> CorrectedEnergyWithError(const reco::Photon &p);
    std::pair<double,double> CorrectedEnergyWithError(const reco::GsfElectron &e, EcalClusterLazyTools &clustertools);
    
  protected:
    const GBRForest *fReadereb;
    const GBRForest *fReaderebvariance;
    const GBRForest *fReaderee;
    const GBRForest *fReadereevariance;      
    
    Bool_t fIsInitialized;
    Bool_t fOwnsForests;
    Float_t *fVals;
    
    };


#endif
