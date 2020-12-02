#include "RecoEgamma/EgammaElectronProducers/interface/LowPtGsfElectronFeatures.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TVector3.h"
#include <algorithm>
#include <iostream>

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
    const auto& trk = ecal.trackRef();  // reco::TrackRef
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
    const auto& ecal_clu = ecal.clusterRef();  // reco::PFClusterRef
    if (ecal_clu.isNonnull()) {
      ktf_ecal_cluster_e_ = ecal_clu->energy();
      ktf_ecal_cluster_deta_ = ecal.geomMatching()[0];
      ktf_ecal_cluster_dphi_ = ecal.geomMatching()[1];
      ktf_ecal_cluster_e3x3_ = tools.e3x3(*ecal_clu);
      ktf_ecal_cluster_e5x5_ = tools.e5x5(*ecal_clu);
      const auto& covs = tools.localCovariances(*ecal_clu);
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
    const auto& hcal_clu = hcal.clusterRef();  // reco::PFClusterRef
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

  std::vector<float> features_V1(reco::GsfElectron const& ele, float rho, float unbiased, float field_z) {
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
    if (ele.core().isNonnull()) {
      reco::TrackRef trk = ele.closestCtfTrackRef();
      if (trk.isNonnull()) {
        eid_trk_p = (float)trk->p();
        eid_trk_nhits = (float)trk->found();
        eid_trk_chi2red = (float)trk->normalizedChi2();
        TVector3 trkTV3(0, 0, 0);
        trkTV3.SetPtEtaPhi(trk->pt(), trk->eta(), trk->phi());
        TVector3 eleTV3(0, 0, 0);
        eleTV3.SetPtEtaPhi(ele.pt(), ele.eta(), ele.phi());
        trk_dr = eleTV3.DeltaR(trkTV3);
      }
    }

    // GSF tracks
    if (ele.core().isNonnull()) {
      reco::GsfTrackRef gsf = ele.core()->gsfTrack();
      if (gsf.isNonnull()) {
        gsf_mode_p = gsf->pMode();
        eid_gsf_nhits = (float)gsf->found();
        eid_gsf_chi2red = gsf->normalizedChi2();
        TVector3 gsfTV3(0, 0, 0);
        gsfTV3.SetPtEtaPhi(gsf->ptMode(), gsf->etaMode(), gsf->phiMode());
        TVector3 eleTV3(0, 0, 0);
        eleTV3.SetPtEtaPhi(ele.pt(), ele.eta(), ele.phi());
        gsf_dr = eleTV3.DeltaR(gsfTV3);
      }
    }

    // Super clusters
    if (ele.core().isNonnull()) {
      reco::SuperClusterRef sc = ele.core()->superCluster();
      if (sc.isNonnull()) {
        eid_sc_E = sc->energy();
        eid_sc_eta = sc->eta();
        eid_sc_etaWidth = sc->etaWidth();
        eid_sc_phiWidth = sc->phiWidth();
        sc_Nclus = sc->clustersSize();
      }
    }

    // Track-cluster matching
    eid_match_seed_dEta = ele.deltaEtaSeedClusterTrackAtCalo();
    eid_match_eclu_EoverP = (1. / ele.ecalEnergy()) - (1. / ele.p());
    eid_match_SC_EoverP = ele.eSuperClusterOverP();
    eid_match_SC_dEta = ele.deltaEtaSuperClusterTrackAtVtx();
    eid_match_SC_dPhi = ele.deltaPhiSuperClusterTrackAtVtx();

    // Shower shape vars
    eid_shape_full5x5_HoverE = ele.full5x5_hcalOverEcal();
    eid_shape_full5x5_r9 = ele.full5x5_r9();

    // Misc
    eid_rho = rho;

    eid_brem_frac = ele.fbrem();
    core_shFracHits = ele.shFracInnerHits();

    // Unbiased BDT from ElectronSeed
    gsf_bdtout1 = unbiased;

    // Clusters
    if (ele.core().isNonnull()) {
      reco::GsfTrackRef gsf = ele.core()->gsfTrack();
      if (gsf.isNonnull()) {
        reco::SuperClusterRef sc = ele.core()->superCluster();
        if (sc.isNonnull()) {
          // Propagate electron track to ECAL surface
          double mass2 = 0.000511 * 0.000511;
          float p2 = pow(gsf->p(), 2);
          float energy = sqrt(mass2 + p2);
          XYZTLorentzVector mom = XYZTLorentzVector(gsf->px(), gsf->py(), gsf->pz(), energy);
          XYZTLorentzVector pos = XYZTLorentzVector(gsf->vx(), gsf->vy(), gsf->vz(), 0.);
          BaseParticlePropagator propagator(RawParticle(mom, pos, gsf->charge()), 0, 0, field_z);
          propagator.propagateToEcalEntrance(true);   // true only first half loop , false more than one loop
          bool reach_ECAL = propagator.getSuccess();  // 0 does not reach ECAL, 1 yes barrel, 2 yes endcaps
                                                      // ECAL entry point for track
          GlobalPoint ecal_pos(propagator.particle().x(), propagator.particle().y(), propagator.particle().z());

          // Track-cluster matching for most energetic clusters
          sc_clus1_nxtal = -999;
          sc_clus1_dphi = -999.;
          sc_clus2_dphi = -999.;
          sc_clus1_deta = -999.;
          sc_clus2_deta = -999.;
          sc_clus1_E = -999.;
          sc_clus2_E = -999.;
          sc_clus1_E_ov_p = -999.;
          sc_clus2_E_ov_p = -999.;
          trackClusterMatching(*sc,
                               *gsf,
                               reach_ECAL,
                               ecal_pos,
                               sc_clus1_nxtal,
                               sc_clus1_dphi,
                               sc_clus2_dphi,
                               sc_clus1_deta,
                               sc_clus2_deta,
                               sc_clus1_E,
                               sc_clus2_E,
                               sc_clus1_E_ov_p,
                               sc_clus2_E_ov_p);
          sc_clus1_nxtal = (int)sc_clus1_nxtal;

        }  // sc.isNonnull()
      }    // gsf.isNonnull()
    }      // clusters

    // Out-of-range
    eid_rho = std::clamp(eid_rho, 0.f, 100.f);
    eid_sc_eta = std::clamp(eid_sc_eta, -5.f, 5.f);
    eid_shape_full5x5_r9 = std::clamp(eid_shape_full5x5_r9, 0.f, 2.f);
    eid_sc_etaWidth = std::clamp(eid_sc_etaWidth, 0.f, 3.14f);
    eid_sc_phiWidth = std::clamp(eid_sc_phiWidth, 0.f, 3.14f);
    eid_shape_full5x5_HoverE = std::clamp(eid_shape_full5x5_HoverE, 0.f, 50.f);
    eid_trk_nhits = std::clamp(eid_trk_nhits, -1.f, 50.f);
    eid_trk_chi2red = std::clamp(eid_trk_chi2red, -1.f, 50.f);
    eid_gsf_chi2red = std::clamp(eid_gsf_chi2red, -1.f, 100.f);
    if (eid_brem_frac < 0.)
      eid_brem_frac = -1.;  //
    if (eid_brem_frac > 1.)
      eid_brem_frac = 1.;  //
    eid_gsf_nhits = std::clamp(eid_gsf_nhits, -1.f, 50.f);
    eid_match_SC_EoverP = std::clamp(eid_match_SC_EoverP, 0.f, 100.f);
    eid_match_eclu_EoverP = std::clamp(eid_match_eclu_EoverP, -1.f, 1.f);
    eid_match_SC_dEta = std::clamp(eid_match_SC_dEta, -10.f, 10.f);
    eid_match_SC_dPhi = std::clamp(eid_match_SC_dPhi, -3.14f, 3.14f);
    eid_match_seed_dEta = std::clamp(eid_match_seed_dEta, -10.f, 10.f);
    eid_sc_E = std::clamp(eid_sc_E, 0.f, 1000.f);
    eid_trk_p = std::clamp(eid_trk_p, -1.f, 1000.f);
    gsf_mode_p = std::clamp(gsf_mode_p, 0.f, 1000.f);
    core_shFracHits = std::clamp(core_shFracHits, 0.f, 1.f);
    gsf_bdtout1 = std::clamp(gsf_bdtout1, -20.f, 20.f);
    if (gsf_dr < 0.)
      gsf_dr = 5.;  //
    if (gsf_dr > 5.)
      gsf_dr = 5.;  //
    if (trk_dr < 0.)
      trk_dr = 5.;  //
    if (trk_dr > 5.)
      trk_dr = 5.;  //
    sc_Nclus = std::clamp(sc_Nclus, 0.f, 20.f);
    sc_clus1_nxtal = std::clamp(sc_clus1_nxtal, 0.f, 100.f);
    sc_clus1_dphi = std::clamp(sc_clus1_dphi, -3.14f, 3.14f);
    sc_clus2_dphi = std::clamp(sc_clus2_dphi, -3.14f, 3.14f);
    sc_clus1_deta = std::clamp(sc_clus1_deta, -5.f, 5.f);
    sc_clus2_deta = std::clamp(sc_clus2_deta, -5.f, 5.f);
    sc_clus1_E = std::clamp(sc_clus1_E, 0.f, 1000.f);
    sc_clus2_E = std::clamp(sc_clus2_E, 0.f, 1000.f);
    if (sc_clus1_E_ov_p < 0.)
      sc_clus1_E_ov_p = -1.;  //
    if (sc_clus2_E_ov_p < 0.)
      sc_clus2_E_ov_p = -1.;  //

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
  // feature list for original models (2019Aug07 and earlier)
  std::vector<float> features_V0(reco::GsfElectron const& ele, float rho, float unbiased) {
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
    if (ele.core().isNonnull()) {
      const auto& trk = ele.closestCtfTrackRef();  // reco::TrackRef
      if (trk.isNonnull()) {
        eid_trk_p = (float)trk->p();
        eid_trk_nhits = (float)trk->found();
        eid_trk_chi2red = (float)trk->normalizedChi2();
        TVector3 trkTV3(0, 0, 0);
        trkTV3.SetPtEtaPhi(trk->pt(), trk->eta(), trk->phi());
        TVector3 eleTV3(0, 0, 0);
        eleTV3.SetPtEtaPhi(ele.pt(), ele.eta(), ele.phi());
        trk_dr = eleTV3.DeltaR(trkTV3);
      }
    }

    // GSF tracks
    if (ele.core().isNonnull()) {
      const auto& gsf = ele.core()->gsfTrack();  // reco::GsfTrackRef
      if (gsf.isNonnull()) {
        gsf_mode_p = gsf->pMode();
        eid_gsf_nhits = (float)gsf->found();
        eid_gsf_chi2red = gsf->normalizedChi2();
        TVector3 gsfTV3(0, 0, 0);
        gsfTV3.SetPtEtaPhi(gsf->ptMode(), gsf->etaMode(), gsf->phiMode());
        TVector3 eleTV3(0, 0, 0);
        eleTV3.SetPtEtaPhi(ele.pt(), ele.eta(), ele.phi());
        gsf_dr = eleTV3.DeltaR(gsfTV3);
      }
    }

    // Super clusters
    if (ele.core().isNonnull()) {
      const auto& sc = ele.core()->superCluster();  // reco::SuperClusterRef
      if (sc.isNonnull()) {
        eid_sc_E = sc->energy();
        eid_sc_eta = sc->eta();
        eid_sc_etaWidth = sc->etaWidth();
        eid_sc_phiWidth = sc->phiWidth();
        sc_Nclus = (float)sc->clustersSize();
      }
    }

    // Track-cluster matching
    eid_match_seed_dEta = ele.deltaEtaSeedClusterTrackAtCalo();
    eid_match_eclu_EoverP = (1. / ele.ecalEnergy()) - (1. / ele.p());
    eid_match_SC_EoverP = ele.eSuperClusterOverP();
    eid_match_SC_dEta = ele.deltaEtaSuperClusterTrackAtVtx();
    eid_match_SC_dPhi = ele.deltaPhiSuperClusterTrackAtVtx();

    // Shower shape vars
    eid_shape_full5x5_HoverE = ele.full5x5_hcalOverEcal();
    eid_shape_full5x5_r9 = ele.full5x5_r9();

    // Misc
    eid_rho = rho;

    eid_brem_frac = ele.fbrem();
    core_shFracHits = (float)ele.shFracInnerHits();

    // Unbiased BDT from ElectronSeed
    gsf_bdtout1 = unbiased;

    // Clusters
    if (ele.core().isNonnull()) {
      const auto& gsf = ele.core()->gsfTrack();  // reco::GsfTrackRef
      if (gsf.isNonnull()) {
        const auto& sc = ele.core()->superCluster();  // reco::SuperClusterRef
        if (sc.isNonnull()) {
          // Propagate electron track to ECAL surface
          double mass2 = 0.000511 * 0.000511;
          float p2 = pow(gsf->p(), 2);
          float energy = sqrt(mass2 + p2);
          math::XYZTLorentzVector mom = math::XYZTLorentzVector(gsf->px(), gsf->py(), gsf->pz(), energy);
          math::XYZTLorentzVector pos = math::XYZTLorentzVector(gsf->vx(), gsf->vy(), gsf->vz(), 0.);
          float field_z = 3.8;
          BaseParticlePropagator mypart(RawParticle(mom, pos, gsf->charge()), 0, 0, field_z);
          mypart.propagateToEcalEntrance(true);   // true only first half loop , false more than one loop
          bool reach_ECAL = mypart.getSuccess();  // 0 does not reach ECAL, 1 yes barrel, 2 yes endcaps

          // ECAL entry point for track
          GlobalPoint ecal_pos(
              mypart.particle().vertex().x(), mypart.particle().vertex().y(), mypart.particle().vertex().z());

          // Track-cluster matching for most energetic clusters
          sc_clus1_nxtal = -999.;
          sc_clus1_dphi = -999.;
          sc_clus2_dphi = -999.;
          sc_clus1_deta = -999.;
          sc_clus2_deta = -999.;
          sc_clus1_E = -999.;
          sc_clus2_E = -999.;
          sc_clus1_E_ov_p = -999.;
          sc_clus2_E_ov_p = -999.;
          trackClusterMatching(*sc,
                               *gsf,
                               reach_ECAL,
                               ecal_pos,
                               sc_clus1_nxtal,
                               sc_clus1_dphi,
                               sc_clus2_dphi,
                               sc_clus1_deta,
                               sc_clus2_deta,
                               sc_clus1_E,
                               sc_clus2_E,
                               sc_clus1_E_ov_p,
                               sc_clus2_E_ov_p);

        }  // sc.isNonnull()
      }    // gsf.isNonnull()
    }      // clusters

    // Out-of-range
    eid_sc_eta = std::clamp(eid_sc_eta, -5.f, 5.f);
    eid_shape_full5x5_r9 = std::clamp(eid_shape_full5x5_r9, 0.f, 2.f);
    eid_sc_etaWidth = std::clamp(eid_sc_etaWidth, 0.f, 3.14f);
    eid_sc_phiWidth = std::clamp(eid_sc_phiWidth, 0.f, 3.14f);
    eid_shape_full5x5_HoverE = std::clamp(eid_shape_full5x5_HoverE, 0.f, 50.f);
    eid_trk_nhits = std::clamp(eid_trk_nhits, -1.f, 50.f);
    eid_trk_chi2red = std::clamp(eid_trk_chi2red, -1.f, 50.f);
    eid_gsf_chi2red = std::clamp(eid_gsf_chi2red, -1.f, 100.f);
    if (eid_brem_frac < 0.)
      eid_brem_frac = -1.;  //
    if (eid_brem_frac > 1.)
      eid_brem_frac = 1.;  //
    eid_gsf_nhits = std::clamp(eid_gsf_nhits, -1.f, 50.f);
    eid_match_SC_EoverP = std::clamp(eid_match_SC_EoverP, 0.f, 100.f);
    eid_match_eclu_EoverP = std::clamp(eid_match_eclu_EoverP, -1.f, 1.f);
    eid_match_SC_dEta = std::clamp(eid_match_SC_dEta, -10.f, 10.f);
    eid_match_SC_dPhi = std::clamp(eid_match_SC_dPhi, -3.14f, 3.14f);
    eid_match_seed_dEta = std::clamp(eid_match_seed_dEta, -10.f, 10.f);
    eid_sc_E = std::clamp(eid_sc_E, 0.f, 1000.f);
    eid_trk_p = std::clamp(eid_trk_p, -1.f, 1000.f);
    gsf_mode_p = std::clamp(gsf_mode_p, 0.f, 1000.f);
    core_shFracHits = std::clamp(core_shFracHits, 0.f, 1.f);
    gsf_bdtout1 = std::clamp(gsf_bdtout1, -20.f, 20.f);
    if (gsf_dr < 0.)
      gsf_dr = 5.;  //
    if (gsf_dr > 5.)
      gsf_dr = 5.;  //
    if (trk_dr < 0.)
      trk_dr = 5.;  //
    if (trk_dr > 5.)
      trk_dr = 5.;  //
    sc_Nclus = std::clamp(sc_Nclus, 0.f, 20.f);
    sc_clus1_nxtal = std::clamp(sc_clus1_nxtal, 0.f, 100.f);
    if (sc_clus1_dphi < -3.14)
      sc_clus1_dphi = -5.;  //
    if (sc_clus1_dphi > 3.14)
      sc_clus1_dphi = 5.;  //
    if (sc_clus2_dphi < -3.14)
      sc_clus2_dphi = -5.;  //
    if (sc_clus2_dphi > 3.14)
      sc_clus2_dphi = 5.;  //
    sc_clus1_deta = std::clamp(sc_clus1_deta, -5.f, 5.f);
    sc_clus2_deta = std::clamp(sc_clus2_deta, -5.f, 5.f);
    sc_clus1_E = std::clamp(sc_clus1_E, 0.f, 1000.f);
    sc_clus2_E = std::clamp(sc_clus2_E, 0.f, 1000.f);
    if (sc_clus1_E_ov_p < 0.)
      sc_clus1_E_ov_p = -1.;  //
    if (sc_clus2_E_ov_p < 0.)
      sc_clus2_E_ov_p = -1.;  //

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
  // Find most energetic clusters
  void findEnergeticClusters(
      reco::SuperCluster const& sc, int& clusNum, float& maxEne1, float& maxEne2, int& i1, int& i2) {
    if (sc.clustersSize() > 0 && sc.clustersBegin() != sc.clustersEnd()) {
      for (auto const& cluster : sc.clusters()) {
        if (cluster->energy() > maxEne1) {
          maxEne1 = cluster->energy();
          i1 = clusNum;
        }
        clusNum++;
      }
      if (sc.clustersSize() > 1) {
        clusNum = 0;
        for (auto const& cluster : sc.clusters()) {
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
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Track-cluster matching for most energetic clusters
  void trackClusterMatching(reco::SuperCluster const& sc,
                            reco::GsfTrack const& gsf,
                            bool const& reach_ECAL,
                            GlobalPoint const& ecal_pos,
                            float& sc_clus1_nxtal,
                            float& sc_clus1_dphi,
                            float& sc_clus2_dphi,
                            float& sc_clus1_deta,
                            float& sc_clus2_deta,
                            float& sc_clus1_E,
                            float& sc_clus2_E,
                            float& sc_clus1_E_ov_p,
                            float& sc_clus2_E_ov_p) {
    // Iterate through ECAL clusters and sort in energy
    int clusNum = 0;
    float maxEne1 = -1;
    float maxEne2 = -1;
    int i1 = -1;
    int i2 = -1;
    findEnergeticClusters(sc, clusNum, maxEne1, maxEne2, i1, i2);

    // track-clusters match
    clusNum = 0;
    if (sc.clustersSize() > 0 && sc.clustersBegin() != sc.clustersEnd()) {
      for (auto const& cluster : sc.clusters()) {
        float deta = ecal_pos.eta() - cluster->eta();
        float dphi = reco::deltaPhi(ecal_pos.phi(), cluster->phi());
        if (clusNum == i1) {
          sc_clus1_E = cluster->energy();
          if (gsf.pMode() > 0)
            sc_clus1_E_ov_p = cluster->energy() / gsf.pMode();
          sc_clus1_nxtal = (float)cluster->size();
          if (reach_ECAL > 0) {
            sc_clus1_deta = deta;
            sc_clus1_dphi = dphi;
          }
        } else if (clusNum == i2) {
          sc_clus2_E = cluster->energy();
          if (gsf.pMode() > 0)
            sc_clus2_E_ov_p = cluster->energy() / gsf.pMode();
          if (reach_ECAL > 0) {
            sc_clus2_deta = deta;
            sc_clus2_dphi = dphi;
          }
        }
        clusNum++;
      }
    }
  }

}  // namespace lowptgsfeleid
