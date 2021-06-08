#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronIDHeavyObjectCache_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronIDHeavyObjectCache_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

namespace reco { 
  class BeamSpot;
  class PreId; 
}

namespace lowptgsfeleid {
  
  class Features {
  public:
    // KF track
    float trk_p_ = -1.;
    float trk_nhits_ = -1.;
    float trk_chi2red_ = -1.;
    // GSF track
    float gsf_nhits_ = -1.;
    float gsf_chi2red_ = -1.;
    // SC 
    float sc_E_ = -1.;
    float sc_eta_ = -1.;
    float sc_etaWidth_ = -1.;
    float sc_phiWidth_ = -1.;
    // Track-cluster matching
    float match_seed_dEta_ = -1.;
    float match_eclu_EoverP_ = -1.;
    float match_SC_EoverP_ = -1.;
    float match_SC_dEta_ = -1.;
    float match_SC_dPhi_ = -1.;
    // Shower shape vars
    float shape_full5x5_sigmaIetaIeta_ = -1.;
    float shape_full5x5_sigmaIphiIphi_ = -1.;
    float shape_full5x5_HoverE_ = -1.;
    float shape_full5x5_r9_ = -1.;
    float shape_full5x5_circularity_ = -1.;
    // Misc
    float rho_ = -1.;
    float brem_frac_ = -1.;
    float ele_pt_ = -1.;
  public:
    std::vector<float> get();
    inline void set( const reco::GsfElectronRef& ele, double rho ) { set(*ele, rho); }
    void set( const reco::GsfElectron& ele, double rho );
  };
  
  class HeavyObjectCache {

  public:

    HeavyObjectCache( const edm::ParameterSet& );

    std::vector<std::string> modelNames() const { return names_; }

    double eval( const std::string& name, const reco::GsfElectronRef&, double rho ) const;
    
  private:

    std::vector<std::string> names_;
    std::vector< std::unique_ptr<const GBRForest> > models_;
    std::vector<double> thresholds_;

  };
}

#endif // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronIDHeavyObjectCache_h
