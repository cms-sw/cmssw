#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "helper.h"
#include <limits>
#include <algorithm>
#include "KinVtxFitter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

class BToV0TrkLLBuilder : public edm::global::EDProducer<> {
  // perhaps we need better structure here (begin run etc)
public:
  typedef std::vector<reco::TransientTrack> TransientTrackCollection;
  explicit BToV0TrkLLBuilder(const edm::ParameterSet &cfg)
      :  // selections
        pre_vtx_selection_{cfg.getParameter<std::string>("preVtxSelection")},
        post_vtx_selection_{cfg.getParameter<std::string>("postVtxSelection")},
        //inputs
        dileptons_{consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("dileptons"))},
        leptons_ttracks_{consumes<TransientTrackCollection>(cfg.getParameter<edm::InputTag>("leptonTransientTracks"))},
        V0s_ttracks_{consumes<TransientTrackCollection>(cfg.getParameter<edm::InputTag>("V0s_ttracks"))},
        V0s_{consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("V0s"))},
        pions_{consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("pions"))},
        pions_ttracks_{consumes<TransientTrackCollection>(cfg.getParameter<edm::InputTag>("pionsTransientTracks"))},
        beamspot_{consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))},
        vertex_src_{consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("offlinePrimaryVertexSrc"))} {
    //output
    produces<pat::CompositeCandidateCollection>();
  }

  ~BToV0TrkLLBuilder() override {}

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {}

private:
  // selections
  const StringCutObjectSelector<pat::CompositeCandidate> pre_vtx_selection_;   // cut on the di-lepton before the SV fit
  const StringCutObjectSelector<pat::CompositeCandidate> post_vtx_selection_;  // cut on the di-lepton after the SV fit

  const edm::EDGetTokenT<pat::CompositeCandidateCollection> dileptons_;
  const edm::EDGetTokenT<TransientTrackCollection> leptons_ttracks_;
  const edm::EDGetTokenT<TransientTrackCollection> V0s_ttracks_;
  const edm::EDGetTokenT<pat::CompositeCandidateCollection> V0s_;
  const edm::EDGetTokenT<pat::CompositeCandidateCollection> pions_;
  const edm::EDGetTokenT<TransientTrackCollection> pions_ttracks_;

  const edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  const edm::EDGetTokenT<reco::VertexCollection> vertex_src_;
};

void BToV0TrkLLBuilder::produce(edm::StreamID, edm::Event &evt, edm::EventSetup const &iSetup) const {
  //input
  edm::Handle<pat::CompositeCandidateCollection> dileptons;
  evt.getByToken(dileptons_, dileptons);

  edm::Handle<TransientTrackCollection> leptons_ttracks;
  evt.getByToken(leptons_ttracks_, leptons_ttracks);
  edm::Handle<pat::CompositeCandidateCollection> pions;
  evt.getByToken(pions_, pions);

  edm::Handle<TransientTrackCollection> pions_ttracks;
  evt.getByToken(pions_ttracks_, pions_ttracks);

  edm::Handle<pat::CompositeCandidateCollection> V0s;
  evt.getByToken(V0s_, V0s);

  edm::Handle<TransientTrackCollection> V0s_ttracks;
  evt.getByToken(V0s_ttracks_, V0s_ttracks);

  edm::Handle<reco::BeamSpot> beamspot;
  evt.getByToken(beamspot_, beamspot);

  edm::Handle<reco::VertexCollection> pvtxs;
  evt.getByToken(vertex_src_, pvtxs);

  edm::ESHandle<MagneticField> fieldHandle;
  const MagneticField *fMagneticField = fieldHandle.product();
  AnalyticalImpactPointExtrapolator extrapolator(fMagneticField);

  std::vector<int> used_lep1_id, used_lep2_id, used_pi_id, used_V0_id;

  // output
  std::unique_ptr<pat::CompositeCandidateCollection> ret_val(new pat::CompositeCandidateCollection());
  for (size_t V0_idx = 0; V0_idx < V0s->size(); ++V0_idx) {
    edm::Ptr<pat::CompositeCandidate> V0_ptr(V0s, V0_idx);
    math::PtEtaPhiMLorentzVector V0_p4(V0_ptr->userFloat("fitted_pt"),
                                       V0_ptr->userFloat("fitted_eta"),
                                       V0_ptr->userFloat("fitted_phi"),
                                       V0_ptr->userFloat("fitted_mass"));
    edm::Ptr<reco::Candidate> pi1_ptr = V0_ptr->userCand("trk1");
    edm::Ptr<reco::Candidate> pi2_ptr = V0_ptr->userCand("trk2");
    unsigned int pi1_idx = V0_ptr->userInt("trk1_idx");
    unsigned int pi2_idx = V0_ptr->userInt("trk2_idx");
    //    float pi1_dr = V0_ptr->userFloat("trk1_dr");
    //    float pi2_dr = V0_ptr->userFloat("trk2_dr");
    for (size_t pi_idx = 0; pi_idx < pions->size(); ++pi_idx) {
      edm::Ptr<pat::CompositeCandidate> pi_ptr(pions, pi_idx);
      if (pi1_idx == pi_idx || pi2_idx == pi_idx)
        continue;
      edm::Ptr<reco::Candidate> pi1_ptr(pions, pi1_idx);
      edm::Ptr<reco::Candidate> pi2_ptr(pions, pi2_idx);
      math::PtEtaPhiMLorentzVector pi_p4(pi_ptr->pt(), pi_ptr->eta(), pi_ptr->phi(), bph::PI_MASS);
      pat::CompositeCandidate cand;
      cand.setP4(pi_ptr->p4() + V0_p4);
      //      cand.setCharge(V0_ptr->userInt("fit_trk1_charge") + V0_ptr->userInt("fit_trk2_charge") + pi_ptr->charge());
      //      cand.addUserInt("fitted_charge",V0_ptr->userInt("fit_trk1_charge") + V0_ptr->userInt("fit_trk2_charge") + pi_ptr->charge());
      cand.addUserInt("pi_idx", pi_idx);
      cand.addUserInt("V0_idx", V0_idx);
      cand.addUserCand("pi", pi_ptr);
      cand.addUserCand("V0", V0_ptr);
      float dr = deltaR(pi_ptr->eta(), pi_ptr->phi(), V0_ptr->userFloat("fitted_eta"), V0_ptr->userFloat("fitted_phi"));
      cand.addUserFloat("V0pi_dr", dr);
      for (size_t ll_idx = 0; ll_idx < dileptons->size(); ++ll_idx) {
        edm::Ptr<pat::CompositeCandidate> ll_ptr(dileptons, ll_idx);
        edm::Ptr<reco::Candidate> l1_ptr = ll_ptr->userCand("l1");
        edm::Ptr<reco::Candidate> l2_ptr = ll_ptr->userCand("l2");
        int l1_idx = ll_ptr->userInt("l1_idx");
        int l2_idx = ll_ptr->userInt("l2_idx");
        cand.addUserCand("l1", l1_ptr);
        cand.addUserCand("l2", l2_ptr);
        cand.addUserCand("dilepton", ll_ptr);
        cand.addUserInt("l1_idx", l1_idx);
        cand.addUserInt("l2_idx", l2_idx);
        cand.addUserInt("ll_idx", ll_idx);
        cand.addUserCand("l1_ptr", l1_ptr);
        cand.addUserCand("l2_ptr", l2_ptr);
        auto lep1_p4 = l1_ptr->polarP4();
        auto lep2_p4 = l2_ptr->polarP4();
        lep1_p4.SetM(l1_ptr->mass());
        lep2_p4.SetM(l2_ptr->mass());
        auto pi_p4 = pi_ptr->polarP4();
        auto V0_p4 = V0_ptr->polarP4();
        pi_p4.SetM(bph::PI_MASS);
        V0_p4.SetM(V0_ptr->mass());
        cand.setP4(ll_ptr->p4() + pi_p4 + V0_p4);
        cand.setCharge(ll_ptr->charge() + V0_ptr->userInt("fit_trk1_charge") + V0_ptr->userInt("fit_trk2_charge") +
                       pi_ptr->charge());
        cand.addUserFloat("ll_V0_deltaR", reco::deltaR(*ll_ptr, *V0_ptr));
        cand.addUserFloat("ll_pi_deltaR", reco::deltaR(*ll_ptr, *pi_ptr));
        auto V0_dr_info = bph::min_max_dr({l1_ptr, l2_ptr, V0_ptr});
        cand.addUserFloat("V0_min_dr", V0_dr_info.first);
        cand.addUserFloat("V0_max_dr", V0_dr_info.second);
        auto pi_dr_info = bph::min_max_dr({l1_ptr, l2_ptr, pi_ptr});
        cand.addUserFloat("pi_min_dr", pi_dr_info.first);
        cand.addUserFloat("pi_max_dr", pi_dr_info.second);
        cand.addUserFloat("mIntermediate_unfitted", (pi_p4 + V0_p4).M());
        if (!pre_vtx_selection_(cand))
          continue;

        KinVtxFitter fitter;
        try {
          fitter = KinVtxFitter({leptons_ttracks->at(l1_idx),
                                 leptons_ttracks->at(l2_idx),
                                 pions_ttracks->at(pi_idx),
                                 V0s_ttracks->at(V0_idx)},
                                {l1_ptr->mass(), l2_ptr->mass(), bph::PI_MASS, V0_ptr->mass()},
                                {bph::LEP_SIGMA, bph::LEP_SIGMA, bph::PI_SIGMA, V0_ptr->userFloat("massErr")});

        } catch (const VertexException &e) {
          edm::LogWarning("KinematicFit") << "V0TrkLL Builder: Skipping candidate due to fit failure: " << e.what();
          continue;
        }
        if (!fitter.success())
          continue;

        cand.setVertex(
            reco::Candidate::Point(fitter.fitted_vtx().x(), fitter.fitted_vtx().y(), fitter.fitted_vtx().z()));

        TrajectoryStateOnSurface V0tsos =
            extrapolator.extrapolate(V0s_ttracks->at(V0_idx).impactPointState(), fitter.fitted_vtx());
        cand.addUserFloat("V0_dz", V0tsos.globalPosition().z() - fitter.fitted_vtx().z());
        cand.addUserFloat("V0_x", V0tsos.globalPosition().x());
        cand.addUserFloat("V0_y", V0tsos.globalPosition().y());
        cand.addUserFloat("V0_z", V0tsos.globalPosition().z());
        TrajectoryStateOnSurface pitsos =
            extrapolator.extrapolate(pions_ttracks->at(pi_idx).impactPointState(), fitter.fitted_vtx());
        cand.addUserFloat("pi_dz", pions_ttracks->at(pi_idx).track().vz() - fitter.fitted_vtx().z());  //
        cand.addUserFloat("pi_x", pions_ttracks->at(pi_idx).track().vx());  //pitsos.globalPosition().x());
        cand.addUserFloat("pi_y", pions_ttracks->at(pi_idx).track().vy());  //pitsos.globalPosition().y());
        cand.addUserFloat("pi_z", pions_ttracks->at(pi_idx).track().vz());  //pitsos.globalPosition().z());
        cand.addUserFloat(
            "V0trk_dz",
            V0tsos.globalPosition().z() - pions_ttracks->at(pi_idx).track().vz());  //pitsos.globalPosition().z());

        // vertex vars
        cand.addUserFloat("sv_chi2", fitter.chi2());
        cand.addUserFloat("sv_ndof", fitter.dof());
        cand.addUserFloat("sv_prob", fitter.prob());
        // refitted kinematic vars
        cand.addUserFloat("fitted_KstarPlus_mass", (fitter.daughter_p4(2) + fitter.daughter_p4(3)).mass());
        cand.addUserFloat("fitted_KstarPlus_pt", (fitter.daughter_p4(2) + fitter.daughter_p4(3)).pt());
        cand.addUserFloat("fitted_KstarPlus_eta", (fitter.daughter_p4(2) + fitter.daughter_p4(3)).eta());
        cand.addUserFloat("fitted_KstarPlus_phi", (fitter.daughter_p4(2) + fitter.daughter_p4(3)).phi());
        auto fit_p4 = fitter.fitted_p4();
        cand.addUserFloat("fitted_mass", fit_p4.mass());
        cand.addUserFloat("fitted_massErr", sqrt(fitter.fitted_candidate().kinematicParametersError().matrix()(6, 6)));
        cand.addUserFloat("fitted_mll_mass", (fitter.daughter_p4(0) + fitter.daughter_p4(1)).mass());
        cand.addUserFloat("fitted_mll_pt", (fitter.daughter_p4(0) + fitter.daughter_p4(1)).pt());
        cand.addUserFloat("fitted_mll_eta", (fitter.daughter_p4(0) + fitter.daughter_p4(1)).eta());
        cand.addUserFloat("fitted_mll_phi", (fitter.daughter_p4(0) + fitter.daughter_p4(1)).phi());
        cand.addUserFloat("fitted_pt", fit_p4.pt());
        cand.addUserFloat("fitted_eta", fit_p4.eta());
        cand.addUserFloat("fitted_phi", fit_p4.phi());

        const reco::BeamSpot &beamSpot = *beamspot;
        TrajectoryStateClosestToPoint theDCAXBS = fitter.fitted_candidate_ttrk().trajectoryStateClosestToPoint(
            GlobalPoint(beamSpot.position().x(), beamSpot.position().y(), beamSpot.position().z()));
        double DCAB0BS = -99.;
        double DCAB0BSErr = -99.;

        if (theDCAXBS.isValid() == true) {
          DCAB0BS = theDCAXBS.perigeeParameters().transverseImpactParameter();
          DCAB0BSErr = theDCAXBS.perigeeError().transverseImpactParameterError();
        }
        cand.addUserFloat("dca", DCAB0BS);
        cand.addUserFloat("dcaErr", DCAB0BSErr);

        cand.addUserFloat("vtx_x", cand.vx());
        cand.addUserFloat("vtx_y", cand.vy());
        cand.addUserFloat("vtx_z", cand.vz());
        cand.addUserFloat("vtx_ex", sqrt(fitter.fitted_vtx_uncertainty().cxx()));
        cand.addUserFloat("vtx_ey", sqrt(fitter.fitted_vtx_uncertainty().cyy()));
        cand.addUserFloat("vtx_ez", sqrt(fitter.fitted_vtx_uncertainty().czz()));
        // refitted daughters (leptons/tracks)
        std::vector<std::string> dnames{"l1", "l2", "pi", "V0"};
        for (size_t idaughter = 0; idaughter < dnames.size(); idaughter++) {
          cand.addUserFloat("fitted_" + dnames[idaughter] + "_pt", fitter.daughter_p4(idaughter).pt());
          cand.addUserFloat("fitted_" + dnames[idaughter] + "_eta", fitter.daughter_p4(idaughter).eta());
          cand.addUserFloat("fitted_" + dnames[idaughter] + "_phi", fitter.daughter_p4(idaughter).phi());
        }
        // other vars
        cand.addUserFloat("cos_theta_2D", bph::cos_theta_2D(fitter, *beamspot, cand.p4()));
        cand.addUserFloat("fitted_cos_theta_2D", bph::cos_theta_2D(fitter, *beamspot, fit_p4));
        auto lxy = bph::l_xy(fitter, *beamspot);
        cand.addUserFloat("l_xy", lxy.value());
        cand.addUserFloat("l_xy_unc", lxy.error());

        std::pair<bool, Measurement1D> cur2DIP_V0 =
            bph::signedTransverseImpactParameter(V0tsos, fitter.fitted_refvtx(), *beamspot);
        std::pair<bool, Measurement1D> cur3DIP_V0 =
            bph::signedImpactParameter3D(V0tsos, fitter.fitted_refvtx(), *beamspot, (*pvtxs)[0].position().z());
        std::pair<bool, Measurement1D> cur2DIP_pi =
            bph::signedTransverseImpactParameter(pitsos, fitter.fitted_refvtx(), *beamspot);
        std::pair<bool, Measurement1D> cur3DIP_pi =
            bph::signedImpactParameter3D(pitsos, fitter.fitted_refvtx(), *beamspot, (*pvtxs)[0].position().z());
        cand.addUserFloat("pi_svip2d", cur2DIP_pi.second.value());
        cand.addUserFloat("pi_svip2d_err", cur2DIP_pi.second.error());
        cand.addUserFloat("pi_svip3d", cur3DIP_pi.second.value());
        cand.addUserFloat("pi_svip3d_err", cur3DIP_pi.second.error());
        cand.addUserFloat("V0_svip2d", cur2DIP_V0.second.value());
        cand.addUserFloat("V0_svip2d_err", cur2DIP_V0.second.error());
        cand.addUserFloat("V0_svip3d", cur3DIP_V0.second.value());
        cand.addUserFloat("V0_svip3d_err", cur3DIP_V0.second.error());
        if (!post_vtx_selection_(cand))
          continue;

        // /////////////////////////////////////////////////////////////////
        // ///     Mass constrained fit START                            ///
        // /////////////////////////////////////////////////////////////////

        // Define variables
        bool sv_OK_withMC = false;
        float sv_chi2_withMC = cand.userFloat("sv_chi2");
        float sv_ndof_withMC = cand.userFloat("sv_ndof");
        float sv_prob_withMC = cand.userFloat("sv_prob");
        float fitted_mll_withMC = cand.userFloat("fitted_mll_mass");
        float fitted_pt_withMC = cand.userFloat("fitted_pt");
        float fitted_eta_withMC = cand.userFloat("fitted_eta");
        float fitted_phi_withMC = cand.userFloat("fitted_phi");
        float fitted_mass_withMC = cand.userFloat("fitted_mass");
        float fitted_massErr_withMC = cand.userFloat("fitted_massErr");
        float fitted_cos_theta_2D_withMC = cand.userFloat("fitted_cos_theta_2D");
        float l_xy_withMC = cand.userFloat("l_xy");
        float l_xy_unc_withMC = cand.userFloat("l_xy_unc");
        float vtx_x_withMC = cand.userFloat("vtx_x");
        float vtx_y_withMC = cand.userFloat("vtx_y");
        float vtx_z_withMC = cand.userFloat("vtx_z");
        float vtx_ex_withMC = cand.userFloat("vtx_ex");
        float vtx_ey_withMC = cand.userFloat("vtx_ey");
        float vtx_ez_withMC = cand.userFloat("vtx_ez");
        float fitted_l1_pt_withMC = cand.userFloat("fitted_l1_pt");
        float fitted_l1_eta_withMC = cand.userFloat("fitted_l1_eta");
        float fitted_l1_phi_withMC = cand.userFloat("fitted_l1_phi");
        float fitted_l2_pt_withMC = cand.userFloat("fitted_l2_pt");
        float fitted_l2_eta_withMC = cand.userFloat("fitted_l2_eta");
        float fitted_l2_phi_withMC = cand.userFloat("fitted_l2_phi");
        float fitted_V0_pt_withMC = cand.userFloat("fitted_V0_pt");
        float fitted_V0_eta_withMC = cand.userFloat("fitted_V0_eta");
        float fitted_V0_phi_withMC = cand.userFloat("fitted_V0_phi");
        float fitted_pi_pt_withMC = cand.userFloat("fitted_pi_pt");
        float fitted_pi_eta_withMC = cand.userFloat("fitted_pi_eta");
        float fitted_pi_phi_withMC = cand.userFloat("fitted_pi_phi");

        // Check dilepton mass from Bparticles to be in the jpsi bin
        const double dilepton_mass = ll_ptr->userFloat("fitted_mass");
        // const double dilepton_mass = (fitter.daughter_p4(0) + fitter.daughter_p4(1)).mass();
        const double jpsi_bin[2] = {2.8,
                                    3.2};  // {2.9, 3.2}; Start bin from 2.8 to be able to measure systematics later
        const double psi2s_bin[2] = {3.55, 3.8};
        if ((dilepton_mass > jpsi_bin[0] && dilepton_mass < jpsi_bin[1]) ||
            (dilepton_mass > psi2s_bin[0] && dilepton_mass < psi2s_bin[1])) {
          // JPsi  mass constrait
          // do mass constrained vertex fit

          ParticleMass JPsi_mass = 3.0969;   // Jpsi mass 3.096900±0.000006
          ParticleMass Psi2S_mass = 3.6861;  // Psi2S mass 3.6861093±0.0000034
          ParticleMass mass_constraint = (dilepton_mass < jpsi_bin[1]) ? JPsi_mass : Psi2S_mass;
          // Mass constraint is applied to the first two particles in the "particles" vector
          // Make sure that the first two particles are the ones you want to constrain
          KinVtxFitter constrained_fitter;
          try {
            constrained_fitter =
                KinVtxFitter({leptons_ttracks->at(l1_idx),
                              leptons_ttracks->at(l2_idx),
                              pions_ttracks->at(pi_idx),
                              V0s_ttracks->at(V0_idx)},
                             {l1_ptr->mass(), l2_ptr->mass(), bph::PI_MASS, V0_ptr->mass()},
                             {bph::LEP_SIGMA, bph::LEP_SIGMA, bph::PI_SIGMA, V0_ptr->userFloat("massErr")},
                             mass_constraint);
          } catch (const VertexException &e) {
            edm::LogWarning("KinematicFit")
                << "V0TrkLL Builder constraint: Skipping candidate due to fit failure: " << e.what();
            continue;
          }
          if (!constrained_fitter.success()) {
            // Save default values and continue
            cand.addUserInt("sv_OK_withMC", sv_OK_withMC);
            cand.addUserFloat("sv_chi2_withMC", sv_chi2_withMC);
            cand.addUserFloat("sv_ndof_withMC", sv_ndof_withMC);
            cand.addUserFloat("sv_prob_withMC", sv_prob_withMC);
            cand.addUserFloat("fitted_mll_withMC", fitted_mll_withMC);
            cand.addUserFloat("fitted_pt_withMC", fitted_pt_withMC);
            cand.addUserFloat("fitted_eta_withMC", fitted_eta_withMC);
            cand.addUserFloat("fitted_phi_withMC", fitted_phi_withMC);
            cand.addUserFloat("fitted_mass_withMC", fitted_mass_withMC);
            cand.addUserFloat("fitted_massErr_withMC", fitted_massErr_withMC);
            cand.addUserFloat("fitted_cos_theta_2D_withMC", fitted_cos_theta_2D_withMC);
            cand.addUserFloat("l_xy_withMC", l_xy_withMC);
            cand.addUserFloat("l_xy_unc_withMC", l_xy_unc_withMC);
            cand.addUserFloat("vtx_x_withMC", vtx_x_withMC);
            cand.addUserFloat("vtx_y_withMC", vtx_y_withMC);
            cand.addUserFloat("vtx_z_withMC", vtx_z_withMC);
            cand.addUserFloat("vtx_ex_withMC", vtx_ex_withMC);
            cand.addUserFloat("vtx_ey_withMC", vtx_ey_withMC);
            cand.addUserFloat("vtx_ez_withMC", vtx_ez_withMC);
            cand.addUserFloat("fitted_l1_pt_withMC", fitted_l1_pt_withMC);
            cand.addUserFloat("fitted_l1_eta_withMC", fitted_l1_eta_withMC);
            cand.addUserFloat("fitted_l1_phi_withMC", fitted_l1_phi_withMC);
            cand.addUserFloat("fitted_l2_pt_withMC", fitted_l2_pt_withMC);
            cand.addUserFloat("fitted_l2_eta_withMC", fitted_l2_eta_withMC);
            cand.addUserFloat("fitted_l2_phi_withMC", fitted_l2_phi_withMC);
            cand.addUserFloat("fitted_V0_pt_withMC", fitted_V0_pt_withMC);
            cand.addUserFloat("fitted_V0_eta_withMC", fitted_V0_eta_withMC);
            cand.addUserFloat("fitted_V0_phi_withMC", fitted_V0_phi_withMC);
            cand.addUserFloat("fitted_pi_pt_withMC", fitted_pi_pt_withMC);
            cand.addUserFloat("fitted_pi_eta_withMC", fitted_pi_eta_withMC);
            cand.addUserFloat("fitted_pi_phi_withMC", fitted_pi_phi_withMC);
            ret_val->push_back(cand);
            continue;
          }
          auto fit_p4_withMC = constrained_fitter.fitted_p4();
          sv_OK_withMC = constrained_fitter.success();
          sv_chi2_withMC = constrained_fitter.chi2();
          sv_ndof_withMC = constrained_fitter.dof();
          sv_prob_withMC = constrained_fitter.prob();
          fitted_mll_withMC = (constrained_fitter.daughter_p4(0) + constrained_fitter.daughter_p4(1)).mass();
          fitted_pt_withMC = fit_p4_withMC.pt();
          fitted_eta_withMC = fit_p4_withMC.eta();
          fitted_phi_withMC = fit_p4_withMC.phi();
          fitted_mass_withMC = constrained_fitter.fitted_candidate().mass();
          fitted_massErr_withMC = sqrt(constrained_fitter.fitted_candidate().kinematicParametersError().matrix()(6, 6));
          fitted_cos_theta_2D_withMC = bph::cos_theta_2D(constrained_fitter, *beamspot, fit_p4_withMC);
          auto lxy_withMC = bph::l_xy(constrained_fitter, *beamspot);
          l_xy_withMC = lxy_withMC.value();
          l_xy_unc_withMC = lxy_withMC.error();
          vtx_x_withMC = cand.vx();
          vtx_y_withMC = cand.vy();
          vtx_z_withMC = cand.vz();
          vtx_ex_withMC = sqrt(constrained_fitter.fitted_vtx_uncertainty().cxx());
          vtx_ey_withMC = sqrt(constrained_fitter.fitted_vtx_uncertainty().cyy());
          vtx_ez_withMC = sqrt(constrained_fitter.fitted_vtx_uncertainty().czz());
          fitted_l1_pt_withMC = constrained_fitter.daughter_p4(0).pt();
          fitted_l1_eta_withMC = constrained_fitter.daughter_p4(0).eta();
          fitted_l1_phi_withMC = constrained_fitter.daughter_p4(0).phi();
          fitted_l2_pt_withMC = constrained_fitter.daughter_p4(1).pt();
          fitted_l2_eta_withMC = constrained_fitter.daughter_p4(1).eta();
          fitted_l2_phi_withMC = constrained_fitter.daughter_p4(1).phi();
          fitted_V0_pt_withMC = constrained_fitter.daughter_p4(3).pt();
          fitted_V0_eta_withMC = constrained_fitter.daughter_p4(3).eta();
          fitted_V0_phi_withMC = constrained_fitter.daughter_p4(3).phi();
          fitted_pi_pt_withMC = constrained_fitter.daughter_p4(2).pt();
          fitted_pi_eta_withMC = constrained_fitter.daughter_p4(2).eta();
          fitted_pi_phi_withMC = constrained_fitter.daughter_p4(2).phi();
        }
        cand.addUserInt("sv_OK_withMC", sv_OK_withMC);
        cand.addUserFloat("sv_chi2_withMC", sv_chi2_withMC);
        cand.addUserFloat("sv_ndof_withMC", sv_ndof_withMC);
        cand.addUserFloat("sv_prob_withMC", sv_prob_withMC);
        cand.addUserFloat("fitted_mll_withMC", fitted_mll_withMC);
        cand.addUserFloat("fitted_pt_withMC", fitted_pt_withMC);
        cand.addUserFloat("fitted_eta_withMC", fitted_eta_withMC);
        cand.addUserFloat("fitted_phi_withMC", fitted_phi_withMC);
        cand.addUserFloat("fitted_mass_withMC", fitted_mass_withMC);
        cand.addUserFloat("fitted_massErr_withMC", fitted_massErr_withMC);
        cand.addUserFloat("fitted_cos_theta_2D_withMC", fitted_cos_theta_2D_withMC);
        cand.addUserFloat("l_xy_withMC", l_xy_withMC);
        cand.addUserFloat("l_xy_unc_withMC", l_xy_unc_withMC);
        cand.addUserFloat("vtx_x_withMC", vtx_x_withMC);
        cand.addUserFloat("vtx_y_withMC", vtx_y_withMC);
        cand.addUserFloat("vtx_z_withMC", vtx_z_withMC);
        cand.addUserFloat("vtx_ex_withMC", vtx_ex_withMC);
        cand.addUserFloat("vtx_ey_withMC", vtx_ey_withMC);
        cand.addUserFloat("vtx_ez_withMC", vtx_ez_withMC);
        cand.addUserFloat("fitted_l1_pt_withMC", fitted_l1_pt_withMC);
        cand.addUserFloat("fitted_l1_eta_withMC", fitted_l1_eta_withMC);
        cand.addUserFloat("fitted_l1_phi_withMC", fitted_l1_phi_withMC);
        cand.addUserFloat("fitted_l2_pt_withMC", fitted_l2_pt_withMC);
        cand.addUserFloat("fitted_l2_eta_withMC", fitted_l2_eta_withMC);
        cand.addUserFloat("fitted_l2_phi_withMC", fitted_l2_phi_withMC);
        cand.addUserFloat("fitted_V0_pt_withMC", fitted_V0_pt_withMC);
        cand.addUserFloat("fitted_V0_eta_withMC", fitted_V0_eta_withMC);
        cand.addUserFloat("fitted_V0_phi_withMC", fitted_V0_phi_withMC);
        cand.addUserFloat("fitted_pi_pt_withMC", fitted_pi_pt_withMC);
        cand.addUserFloat("fitted_pi_eta_withMC", fitted_pi_eta_withMC);
        cand.addUserFloat("fitted_pi_phi_withMC", fitted_pi_phi_withMC);

        // /////////////////////////////////////////////////////////////////
        // ///     Mass constrained fit END                              ///
        // /////////////////////////////////////////////////////////////////
        ret_val->push_back(cand);
        //	std::cout << "BGIKE" << endl;
      }  //     for(size_t ll_idx = 0; ll_idx < dileptons->size(); ++ll_idx)
    }  //   for(size_t pi_idx = 0; pi_idx < pions->size(); ++V0_idx)
  }  // for(size_t V0_idx = 0; V0_idx < V0s->size(); ++V0_idx)
  for (auto &cand : *ret_val) {
    cand.addUserInt("n_pi_used", std::count(used_pi_id.begin(), used_pi_id.end(), cand.userInt("pi_idx")));
    cand.addUserInt("n_V0_used", std::count(used_V0_id.begin(), used_V0_id.end(), cand.userInt("V0_idx")));
    cand.addUserInt("n_l1_used",
                    std::count(used_lep1_id.begin(), used_lep1_id.end(), cand.userInt("l1_idx")) +
                        std::count(used_lep2_id.begin(), used_lep2_id.end(), cand.userInt("l1_idx")));
    cand.addUserInt("n_l2_used",
                    std::count(used_lep1_id.begin(), used_lep1_id.end(), cand.userInt("l2_idx")) +
                        std::count(used_lep2_id.begin(), used_lep2_id.end(), cand.userInt("l2_idx")));
  }
  evt.put(std::move(ret_val));
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BToV0TrkLLBuilder);
