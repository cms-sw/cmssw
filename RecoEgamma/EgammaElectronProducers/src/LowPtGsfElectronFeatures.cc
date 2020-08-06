#include "RecoEgamma/EgammaElectronProducers/interface/LowPtGsfElectronFeatures.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <TVector3.h>

namespace lowptgsfeleseed {

  ////////////////////////////////////////////////////////////////////////////////
  //
  std::vector<float> features(const reco::PreId& ecal,
                              const reco::PreId& hcal,
                              double rho,
                              const reco::BeamSpot& spot,
                              noZS::EcalClusterLazyTools& tools) {
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

    // Set contents of vector
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
  };

}  // namespace lowptgsfeleseed

namespace lowptgsfeleid {

  std::vector<float> features(edm::Ptr<reco::GsfElectron> const& ele, float rho, float unbiased) {
    float eid_rho = -999.;
    float eid_sc_eta = -999.;
    float eid_shape_full5x5_r9 = -999.;
    float eid_sc_etaWidth = -999.;
    float eid_sc_phiWidth = -999.;
    float eid_shape_full5x5_HoverE = -999.;
    float eid_trk_nhits = -999.;
    float eid_trk_chi2red = -999.;
    float eid_gsf_chi2red = -999.;
    float eid_brem_frac = -999.;
    float eid_gsf_nhits = -999.;
    float eid_match_SC_EoverP = -999.;
    float eid_match_eclu_EoverP = -999.;
    float eid_match_SC_dEta = -999.;
    float eid_match_SC_dPhi = -999.;
    float eid_match_seed_dEta = -999.;
    float eid_sc_E = -999.;
    float eid_trk_p = -999.;
    float gsf_mode_p = -999.;
    float core_shFracHits = -999.;
    float gsf_bdtout1 = -999.;
    float gsf_dr = -999.;
    float trk_dr = -999.;
    float sc_Nclus = -999.;
    float sc_clus1_nxtal = -999.;
    float sc_clus1_dphi = -999.;
    float sc_clus2_dphi = -999.;
    float sc_clus1_deta = -999.;
    float sc_clus2_deta = -999.;
    float sc_clus1_E = -999.;
    float sc_clus2_E = -999.;
    float sc_clus1_E_ov_p = -999.;
    float sc_clus2_E_ov_p = -999.;

    // KF tracks
    if (ele->core().isNonnull()) {
      reco::TrackRef trk = ele->closestCtfTrackRef();
      if (trk.isNonnull()) {
        eid_trk_p = (float)trk->p();
        eid_trk_nhits = (float)trk->found();
        eid_trk_chi2red = (float)trk->normalizedChi2();
        TVector3 trkTV3(0, 0, 0);
        trkTV3.SetPtEtaPhi(trk->pt(), trk->eta(), trk->phi());
        TVector3 eleTV3(0, 0, 0);
        eleTV3.SetPtEtaPhi(ele->pt(), ele->eta(), ele->phi());
        trk_dr = eleTV3.DeltaR(trkTV3);
      }
    }

    // GSF tracks
    if (ele->core().isNonnull()) {
      reco::GsfTrackRef gsf = ele->core()->gsfTrack();
      if (gsf.isNonnull()) {
        gsf_mode_p = gsf->pMode();
        eid_gsf_nhits = (float)gsf->found();
        eid_gsf_chi2red = gsf->normalizedChi2();
        TVector3 gsfTV3(0, 0, 0);
        gsfTV3.SetPtEtaPhi(gsf->ptMode(), gsf->etaMode(), gsf->phiMode());
        TVector3 eleTV3(0, 0, 0);
        eleTV3.SetPtEtaPhi(ele->pt(), ele->eta(), ele->phi());
        gsf_dr = eleTV3.DeltaR(gsfTV3);
      }
    }

    // Super clusters
    if (ele->core().isNonnull()) {
      reco::SuperClusterRef sc = ele->core()->superCluster();
      if (sc.isNonnull()) {
        eid_sc_E = sc->energy();
        eid_sc_eta = sc->eta();
        eid_sc_etaWidth = sc->etaWidth();
        eid_sc_phiWidth = sc->phiWidth();
        sc_Nclus = (float)sc->clustersSize();
      }
    }

    // Track-cluster matching
    if (ele.isNonnull()) {
      eid_match_seed_dEta = ele->deltaEtaSeedClusterTrackAtCalo();
      eid_match_eclu_EoverP = (1. / ele->ecalEnergy()) - (1. / ele->p());
      eid_match_SC_EoverP = ele->eSuperClusterOverP();
      eid_match_SC_dEta = ele->deltaEtaSuperClusterTrackAtVtx();
      eid_match_SC_dPhi = ele->deltaPhiSuperClusterTrackAtVtx();
    }

    // Shower shape vars
    if (ele.isNonnull()) {
      eid_shape_full5x5_HoverE = ele->full5x5_hcalOverEcal();
      eid_shape_full5x5_r9 = ele->full5x5_r9();
    }

    // Misc
    eid_rho = rho;

    if (ele.isNonnull()) {
      eid_brem_frac = ele->fbrem();
      core_shFracHits = (float)ele->shFracInnerHits();
    }

    // Unbiased BDT from ElectronSeed
    gsf_bdtout1 = unbiased;

    // Clusters
    if (ele->core().isNonnull()) {
      reco::GsfTrackRef gsf = ele->core()->gsfTrack();
      if (gsf.isNonnull()) {
        reco::SuperClusterRef sc = ele->core()->superCluster();
        if (sc.isNonnull()) {
          // Propagate electron track to ECAL surface
          double mass_ = 0.000511 * 0.000511;
          float p2 = pow(gsf->p(), 2);
          float energy = sqrt(mass_ + p2);
          math::XYZTLorentzVector mom = math::XYZTLorentzVector(gsf->px(), gsf->py(), gsf->pz(), energy);
          math::XYZTLorentzVector pos = math::XYZTLorentzVector(gsf->vx(), gsf->vy(), gsf->vz(), 0.);
          float field_z = 3.8;
          BaseParticlePropagator mypart(RawParticle(mom, pos, gsf->charge()), 0, 0, field_z);
          mypart.propagateToEcalEntrance(true);   // true only first half loop , false more than one loop
          bool reach_ECAL = mypart.getSuccess();  // 0 does not reach ECAL, 1 yes barrel, 2 yes endcaps

          // ECAL entry point for track
          GlobalPoint ecal_pos(
              mypart.particle().vertex().x(), mypart.particle().vertex().y(), mypart.particle().vertex().z());

          // Iterate through ECAL clusters and sort in energy
          int clusNum = 0;
          float maxEne1 = -1;
          float maxEne2 = -1;
          int i1 = -1;
          int i2 = -1;
          try {
            if (sc->clustersSize() > 0 && sc->clustersBegin() != sc->clustersEnd()) {
              for (auto& cluster : sc->clusters()) {
                if (cluster->energy() > maxEne1) {
                  maxEne1 = cluster->energy();
                  i1 = clusNum;
                }
                clusNum++;
              }
              if (sc->clustersSize() > 1) {
                clusNum = 0;
                for (auto& cluster : sc->clusters()) {
                  if (clusNum != i1) {
                    if (cluster->energy() > maxEne2) {
                      maxEne2 = cluster->energy();
                      i2 = clusNum;
                    }
                  }
                  clusNum++;
                }
              }
            }  // loop over clusters
          } catch (...) {
            std::cout << "exception caught clusNum=" << clusNum << " clus size" << sc->clustersSize()
                      << " energy=" << sc->energy() << std::endl;
          }

          // Initializations
          sc_clus1_nxtal = -999.;
          sc_clus1_dphi = -999.;
          sc_clus2_dphi = -999.;
          sc_clus1_deta = -999.;
          sc_clus2_deta = -999.;
          sc_clus1_E = -999.;
          sc_clus2_E = -999.;
          sc_clus1_E_ov_p = -999.;
          sc_clus2_E_ov_p = -999.;

          // track-clusters match
          clusNum = 0;
          try {
            if (sc->clustersSize() > 0 && sc->clustersBegin() != sc->clustersEnd()) {
              for (auto& cluster : sc->clusters()) {
                double pi_ = 3.1415926535;
                float deta = std::fabs(ecal_pos.eta() - cluster->eta());
                float dphi = std::fabs(ecal_pos.phi() - cluster->phi());
                if (dphi > pi_)
                  dphi -= 2 * pi_;
                if (ecal_pos.phi() - cluster->phi() < 0)
                  dphi = -dphi;
                if (ecal_pos.eta() - cluster->eta() < 0)
                  deta = -deta;

                if (clusNum == i1) {
                  sc_clus1_E = cluster->energy();
                  if (gsf->pMode() > 0)
                    sc_clus1_E_ov_p = cluster->energy() / gsf->pMode();
                  sc_clus1_nxtal = (float)cluster->size();
                  if (reach_ECAL > 0) {
                    sc_clus1_deta = deta;
                    sc_clus1_dphi = dphi;
                  }
                } else if (clusNum == i2) {
                  sc_clus2_E = cluster->energy();
                  if (gsf->pMode() > 0)
                    sc_clus2_E_ov_p = cluster->energy() / gsf->pMode();
                  if (reach_ECAL > 0) {
                    sc_clus2_deta = deta;
                    sc_clus2_dphi = dphi;
                  }
                }
                clusNum++;
              }
            }
          } catch (...) {
            std::cout << "caught an exception" << std::endl;
          }
        }
      }
    }  // clusters

    // Out-of-range
    if (eid_rho < 0)
      eid_rho = 0;
    if (eid_rho > 100)
      eid_rho = 100;
    if (eid_sc_eta < -5)
      eid_sc_eta = -5;
    if (eid_sc_eta > 5)
      eid_sc_eta = 5;
    if (eid_shape_full5x5_r9 < 0)
      eid_shape_full5x5_r9 = 0;
    if (eid_shape_full5x5_r9 > 2)
      eid_shape_full5x5_r9 = 2;
    if (eid_sc_etaWidth < 0)
      eid_sc_etaWidth = 0;
    if (eid_sc_etaWidth > 3.14)
      eid_sc_etaWidth = 3.14;
    if (eid_sc_phiWidth < 0)
      eid_sc_phiWidth = 0;
    if (eid_sc_phiWidth > 3.14)
      eid_sc_phiWidth = 3.14;
    if (eid_shape_full5x5_HoverE < 0)
      eid_shape_full5x5_HoverE = 0;
    if (eid_shape_full5x5_HoverE > 50)
      eid_shape_full5x5_HoverE = 50;
    if (eid_trk_nhits < -1)
      eid_trk_nhits = -1;
    if (eid_trk_nhits > 50)
      eid_trk_nhits = 50;
    if (eid_trk_chi2red < -1)
      eid_trk_chi2red = -1;
    if (eid_trk_chi2red > 50)
      eid_trk_chi2red = 50;
    if (eid_gsf_chi2red < -1)
      eid_gsf_chi2red = -1;
    if (eid_gsf_chi2red > 100)
      eid_gsf_chi2red = 100;
    if (eid_brem_frac < 0)
      eid_brem_frac = -1;
    if (eid_brem_frac > 1)
      eid_brem_frac = 1;
    if (eid_gsf_nhits < -1)
      eid_gsf_nhits = -1;
    if (eid_gsf_nhits > 50)
      eid_gsf_nhits = 50;
    if (eid_match_SC_EoverP < 0)
      eid_match_SC_EoverP = 0;
    if (eid_match_SC_EoverP > 100)
      eid_match_SC_EoverP = 100;
    if (eid_match_eclu_EoverP < -1.)
      eid_match_eclu_EoverP = -1.;
    if (eid_match_eclu_EoverP > 1.)
      eid_match_eclu_EoverP = 1.;
    if (eid_match_SC_dEta < -10)
      eid_match_SC_dEta = -10;
    if (eid_match_SC_dEta > 10)
      eid_match_SC_dEta = 10;
    if (eid_match_SC_dPhi < -3.14)
      eid_match_SC_dPhi = -3.14;
    if (eid_match_SC_dPhi > 3.14)
      eid_match_SC_dPhi = 3.14;
    if (eid_match_seed_dEta < -10)
      eid_match_seed_dEta = -10;
    if (eid_match_seed_dEta > 10)
      eid_match_seed_dEta = 10;
    if (eid_sc_E < 0)
      eid_sc_E = 0;
    if (eid_sc_E > 1000)
      eid_sc_E = 1000;
    if (eid_trk_p < -1)
      eid_trk_p = -1;
    if (eid_trk_p > 1000)
      eid_trk_p = 1000;
    if (gsf_mode_p < 0)
      gsf_mode_p = 0;
    if (gsf_mode_p > 1000)
      gsf_mode_p = 1000;
    if (core_shFracHits < 0)
      core_shFracHits = 0;
    if (core_shFracHits > 1)
      core_shFracHits = 1;
    if (gsf_bdtout1 < -20)
      gsf_bdtout1 = -20;
    if (gsf_bdtout1 > 20)
      gsf_bdtout1 = 20;
    if (gsf_dr < 0)
      gsf_dr = 5;
    if (gsf_dr > 5)
      gsf_dr = 5;
    if (trk_dr < 0)
      trk_dr = 5;
    if (trk_dr > 5)
      trk_dr = 5;
    if (sc_Nclus < 0)
      sc_Nclus = 0;
    if (sc_Nclus > 20)
      sc_Nclus = 20;
    if (sc_clus1_nxtal < 0)
      sc_clus1_nxtal = 0;
    if (sc_clus1_nxtal > 100)
      sc_clus1_nxtal = 100;
    if (sc_clus1_dphi < -3.14)
      sc_clus1_dphi = -5;
    if (sc_clus1_dphi > 3.14)
      sc_clus1_dphi = 5;
    if (sc_clus2_dphi < -3.14)
      sc_clus2_dphi = -5;
    if (sc_clus2_dphi > 3.14)
      sc_clus2_dphi = 5;
    if (sc_clus1_deta < -5)
      sc_clus1_deta = -5;
    if (sc_clus1_deta > 5)
      sc_clus1_deta = 5;
    if (sc_clus2_deta < -5)
      sc_clus2_deta = -5;
    if (sc_clus2_deta > 5)
      sc_clus2_deta = 5;
    if (sc_clus1_E < 0)
      sc_clus1_E = 0;
    if (sc_clus1_E > 1000)
      sc_clus1_E = 1000;
    if (sc_clus2_E < 0)
      sc_clus2_E = 0;
    if (sc_clus2_E > 1000)
      sc_clus2_E = 1000;
    if (sc_clus1_E_ov_p < 0)
      sc_clus1_E_ov_p = -1;
    if (sc_clus2_E_ov_p < 0)
      sc_clus2_E_ov_p = -1;

    // Set contents of vector
    std::vector<float> output = {eid_rho,
                                 eid_sc_eta,
                                 eid_shape_full5x5_r9,
                                 eid_sc_etaWidth,
                                 eid_sc_phiWidth,
                                 eid_shape_full5x5_HoverE,
                                 eid_trk_nhits,
                                 eid_trk_chi2red,
                                 eid_gsf_chi2red,
                                 eid_brem_frac,
                                 eid_gsf_nhits,
                                 eid_match_SC_EoverP,
                                 eid_match_eclu_EoverP,
                                 eid_match_SC_dEta,
                                 eid_match_SC_dPhi,
                                 eid_match_seed_dEta,
                                 eid_sc_E,
                                 eid_trk_p,
                                 gsf_mode_p,
                                 core_shFracHits,
                                 gsf_bdtout1,
                                 gsf_dr,
                                 trk_dr,
                                 sc_Nclus,
                                 sc_clus1_nxtal,
                                 sc_clus1_dphi,
                                 sc_clus2_dphi,
                                 sc_clus1_deta,
                                 sc_clus2_deta,
                                 sc_clus1_E,
                                 sc_clus2_E,
                                 sc_clus1_E_ov_p,
                                 sc_clus2_E_ov_p};
    return output;
  }

  ////////////////////////////////////////////////////////////////////////////////
  //
  std::vector<float> features(edm::Ref<std::vector<reco::GsfElectron> > const& ele, float rho, float unbiased) {
    return features(edm::refToPtr(ele), rho, unbiased);
  }

  ////////////////////////////////////////////////////////////////////////////////
  //
  std::vector<float> features(edm::Ref<edm::View<reco::GsfElectron> > const& ele, float rho, float unbiased) {
    return features(edm::refToPtr(ele), rho, unbiased);
  }

}  // namespace lowptgsfeleid
