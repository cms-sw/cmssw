#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedHeavyObjectCache_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedHeavyObjectCache_h

#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include <vector>

namespace reco {
  class BeamSpot;
  class PreId;
}  // namespace reco

namespace lowptgsfeleseed {

  class Features {
  public:
    float trk_pt_ = -1.;
    float trk_eta_ = -1.;
    float trk_phi_ = -1.;
    float trk_p_ = -1.;
    float trk_nhits_ = -1.;
    float trk_high_quality_ = -1.;
    float trk_chi2red_ = -1.;
    float rho_ = -1.;
    float ktf_ecal_cluster_e_ = -1.;
    float ktf_ecal_cluster_deta_ = -42.;
    float ktf_ecal_cluster_dphi_ = -42.;
    float ktf_ecal_cluster_e3x3_ = -1.;
    float ktf_ecal_cluster_e5x5_ = -1.;
    float ktf_ecal_cluster_covEtaEta_ = -42.;
    float ktf_ecal_cluster_covEtaPhi_ = -42.;
    float ktf_ecal_cluster_covPhiPhi_ = -42.;
    float ktf_ecal_cluster_r9_ = -0.1;
    float ktf_ecal_cluster_circularity_ = -0.1;
    float ktf_hcal_cluster_e_ = -1.;
    float ktf_hcal_cluster_deta_ = -42.;
    float ktf_hcal_cluster_dphi_ = -42.;
    float preid_gsf_dpt_ = -1.;
    float preid_trk_gsf_chiratio_ = -1.;
    float preid_gsf_chi2red_ = -1.;
    float trk_dxy_sig_ = -1.;  // must be last (not used by unbiased model)
  public:
    std::vector<float> get();
    void set(const reco::PreId& ecal,
             const reco::PreId& hcal,
             double rho,
             const reco::BeamSpot& spot,
             noZS::EcalClusterLazyTools& ecalTools);
  };

  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet&);

    std::vector<std::string> modelNames() const { return names_; }

    bool eval(const std::string& name,
              reco::PreId& ecal,
              reco::PreId& hcal,
              double rho,
              const reco::BeamSpot& spot,
              noZS::EcalClusterLazyTools& ecalTools) const;

  private:
    std::vector<std::string> names_;
    std::vector<std::unique_ptr<const GBRForest> > models_;
    std::vector<double> thresholds_;
  };
}  // namespace lowptgsfeleseed

#endif  // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedHeavyObjectCache_h
