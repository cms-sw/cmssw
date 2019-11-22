#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "LowPtGsfElectronSeedHeavyObjectCache.h"

#include <string>

namespace lowptgsfeleseed {

  ////////////////////////////////////////////////////////////////////////////////
  //
  std::vector<float> Features::get() {
    std::vector<float> output = {trk_pt_,
                                 trk_eta_,
                                 trk_phi_,
                                 trk_p_,
                                 trk_nhits_,
                                 trk_high_quality_,
                                 trk_chi2red_,
                                 rho_,
                                 ktf_ecal_cluster_e_,
                                 ktf_ecal_cluster_deta_,
                                 ktf_ecal_cluster_dphi_,
                                 ktf_ecal_cluster_e3x3_,
                                 ktf_ecal_cluster_e5x5_,
                                 ktf_ecal_cluster_covEtaEta_,
                                 ktf_ecal_cluster_covEtaPhi_,
                                 ktf_ecal_cluster_covPhiPhi_,
                                 ktf_ecal_cluster_r9_,
                                 ktf_ecal_cluster_circularity_,
                                 ktf_hcal_cluster_e_,
                                 ktf_hcal_cluster_deta_,
                                 ktf_hcal_cluster_dphi_,
                                 preid_gsf_dpt_,
                                 preid_trk_gsf_chiratio_,
                                 preid_gsf_chi2red_,
                                 trk_dxy_sig_};
    return output;
  }

  ////////////////////////////////////////////////////////////////////////////////
  //
  void Features::set(const reco::PreId& ecal,
                     const reco::PreId& hcal,
                     double rho,
                     const reco::BeamSpot& spot,
                     noZS::EcalClusterLazyTools& tools) {
    // Tracks
    reco::TrackRef trk = ecal.trackRef();
    if (trk.isNonnull()) {
      trk_pt_ = trk->pt();
      trk_eta_ = trk->eta();
      trk_phi_ = trk->phi();
      trk_p_ = trk->p();
      trk_nhits_ = static_cast<float>(trk->found());
      trk_high_quality_ = static_cast<float>(trk->quality(reco::TrackBase::qualityByName("highPurity")));
      trk_chi2red_ = trk->normalizedChi2();
      if (trk->dxy(spot) > 0.) {
        trk_dxy_sig_ = trk->dxyError() / trk->dxy(spot);  //@@ to be consistent with the training based on 94X MC
      }
      ktf_ecal_cluster_dphi_ *= trk->charge();  //@@ to be consistent with the training based on 94X MC
    }

    // Rho
    rho_ = static_cast<float>(rho);

    // ECAL clusters
    reco::PFClusterRef ecal_clu = ecal.clusterRef();
    if (ecal_clu.isNonnull()) {
      ktf_ecal_cluster_e_ = ecal_clu->energy();
      ktf_ecal_cluster_deta_ = ecal.geomMatching()[0];
      ktf_ecal_cluster_dphi_ = ecal.geomMatching()[1];
      ktf_ecal_cluster_e3x3_ = tools.e3x3(*ecal_clu);
      ktf_ecal_cluster_e5x5_ = tools.e5x5(*ecal_clu);
      auto covs = tools.localCovariances(*ecal_clu);
      ktf_ecal_cluster_covEtaEta_ = covs[0];
      ktf_ecal_cluster_covEtaPhi_ = covs[1];
      ktf_ecal_cluster_covPhiPhi_ = covs[2];
      if (ktf_ecal_cluster_e_ > 0.) {
        ktf_ecal_cluster_r9_ = ktf_ecal_cluster_e3x3_ / ktf_ecal_cluster_e_;
      }
      if (ktf_ecal_cluster_e5x5_ > 0.) {
        ktf_ecal_cluster_circularity_ = 1. - tools.e1x5(*ecal_clu) / ktf_ecal_cluster_e5x5_;
      } else {
        ktf_ecal_cluster_circularity_ = -0.1;
      }
    }

    // HCAL clusters
    reco::PFClusterRef hcal_clu = hcal.clusterRef();
    if (hcal_clu.isNonnull()) {
      ktf_hcal_cluster_e_ = hcal_clu->energy();
      ktf_hcal_cluster_deta_ = hcal.geomMatching()[0];
      ktf_hcal_cluster_dphi_ = hcal.geomMatching()[1];
    }

    // PreId
    preid_gsf_dpt_ = ecal.dpt();
    preid_trk_gsf_chiratio_ = ecal.chi2Ratio();
    preid_gsf_chi2red_ = ecal.gsfChi2();
  };

  ////////////////////////////////////////////////////////////////////////////////
  //
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {
    for (auto& name : conf.getParameter<std::vector<std::string> >("ModelNames")) {
      names_.push_back(name);
    }
    for (auto& weights : conf.getParameter<std::vector<std::string> >("ModelWeights")) {
      models_.push_back(createGBRForest(edm::FileInPath(weights)));
    }
    for (auto& thresh : conf.getParameter<std::vector<double> >("ModelThresholds")) {
      thresholds_.push_back(thresh);
    }
    if (names_.size() != models_.size()) {
      throw cms::Exception("Incorrect configuration")
          << "'ModelNames' size (" << names_.size() << ") != 'ModelWeights' size (" << models_.size() << ").\n";
    }
    if (models_.size() != thresholds_.size()) {
      throw cms::Exception("Incorrect configuration")
          << "'ModelWeights' size (" << models_.size() << ") != 'ModelThresholds' size (" << thresholds_.size()
          << ").\n";
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  //
  bool HeavyObjectCache::eval(const std::string& name,
                              reco::PreId& ecal,
                              reco::PreId& hcal,
                              double rho,
                              const reco::BeamSpot& spot,
                              noZS::EcalClusterLazyTools& ecalTools) const {
    std::vector<std::string>::const_iterator iter = std::find(names_.begin(), names_.end(), name);
    if (iter != names_.end()) {
      int index = std::distance(names_.begin(), iter);
      Features features;
      features.set(ecal, hcal, rho, spot, ecalTools);
      std::vector<float> inputs = features.get();
      float output = models_.at(index)->GetResponse(inputs.data());
      bool pass = output > thresholds_.at(index);
      ecal.setMVA(pass, output, index);
      return pass;
    } else {
      throw cms::Exception("Unknown model name")
          << "'Name given: '" << name << "'. Check against configuration file.\n";
    }
  }

}  // namespace lowptgsfeleseed
