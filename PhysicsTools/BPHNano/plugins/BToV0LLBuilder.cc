/////////////////////////////// BToV0LLBuilder ///////////////////////////////
/// original authors: G Karathanasis (CERN),  G Melachroinos (NKUA)
/// takes rebuilt V0 cands and a dilepton collection and produces B mothers

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "KinVtxFitter.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "helper.h"

class BToV0LLBuilder : public edm::global::EDProducer<> {
  // perhaps we need better structure here (begin run etc)
public:
  typedef std::vector<reco::TransientTrack> TransientTrackCollection;

  explicit BToV0LLBuilder(const edm::ParameterSet &cfg)
      : bFieldToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
        pre_vtx_selection_{cfg.getParameter<std::string>("preVtxSelection")},
        post_vtx_selection_{cfg.getParameter<std::string>("postVtxSelection")},
        dileptons_{consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("dileptons"))},
        leptons_ttracks_{consumes<TransientTrackCollection>(cfg.getParameter<edm::InputTag>("leptonTransientTracks"))},
        v0s_{consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("v0s"))},
        v0_ttracks_{consumes<TransientTrackCollection>(cfg.getParameter<edm::InputTag>("v0TransientTracks"))},
        pu_tracks_(consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("PUtracks"))),
        beamspot_{consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))},
        dilepton_constraint_{cfg.getParameter<bool>("dileptonMassContraint")} {
    produces<pat::CompositeCandidateCollection>();
  }

  ~BToV0LLBuilder() override {}

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;

  // selection
  const StringCutObjectSelector<pat::CompositeCandidate> pre_vtx_selection_;
  const StringCutObjectSelector<pat::CompositeCandidate> post_vtx_selection_;

  // input
  const edm::EDGetTokenT<pat::CompositeCandidateCollection> dileptons_;
  const edm::EDGetTokenT<TransientTrackCollection> leptons_ttracks_;
  const edm::EDGetTokenT<pat::CompositeCandidateCollection> v0s_;
  const edm::EDGetTokenT<TransientTrackCollection> v0_ttracks_;
  const edm::EDGetTokenT<pat::CompositeCandidateCollection> pu_tracks_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  const bool dilepton_constraint_;
};

void BToV0LLBuilder::produce(edm::StreamID, edm::Event &evt, edm::EventSetup const &iSetup) const {
  // input
  edm::Handle<pat::CompositeCandidateCollection> dileptons;
  evt.getByToken(dileptons_, dileptons);
  edm::Handle<TransientTrackCollection> leptons_ttracks;
  evt.getByToken(leptons_ttracks_, leptons_ttracks);

  edm::Handle<pat::CompositeCandidateCollection> v0s;
  evt.getByToken(v0s_, v0s);
  edm::Handle<TransientTrackCollection> v0_ttracks;
  evt.getByToken(v0_ttracks_, v0_ttracks);

  edm::Handle<pat::CompositeCandidateCollection> pu_tracks;
  evt.getByToken(pu_tracks_, pu_tracks);

  edm::Handle<reco::BeamSpot> beamspot;
  evt.getByToken(beamspot_, beamspot);

  edm::ESHandle<MagneticField> fieldHandle;
  const auto &bField = iSetup.getData(bFieldToken_);
  AnalyticalImpactPointExtrapolator extrapolator(&bField);

  // output
  std::unique_ptr<pat::CompositeCandidateCollection> ret_val(new pat::CompositeCandidateCollection());

  // access V0
  for (size_t v0_idx = 0; v0_idx < v0s->size(); ++v0_idx) {
    edm::Ptr<pat::CompositeCandidate> v0_ptr(v0s, v0_idx);

    // access ll
    for (size_t ll_idx = 0; ll_idx < dileptons->size(); ++ll_idx) {
      edm::Ptr<pat::CompositeCandidate> ll_ptr(dileptons, ll_idx);
      edm::Ptr<reco::Candidate> l1_ptr = ll_ptr->userCand("l1");
      edm::Ptr<reco::Candidate> l2_ptr = ll_ptr->userCand("l2");
      int l1_idx = ll_ptr->userInt("l1_idx");
      int l2_idx = ll_ptr->userInt("l2_idx");

      pat::CompositeCandidate cand;
      cand.setP4(ll_ptr->p4() + v0_ptr->p4());
      cand.setCharge(ll_ptr->charge() + v0_ptr->charge());

      cand.addUserInt("l1_idx", l1_idx);
      cand.addUserInt("l2_idx", l2_idx);
      cand.addUserInt("ll_idx", ll_idx);
      cand.addUserInt("v0_idx", v0_idx);

      auto dr_info = bph::min_max_dr({l1_ptr, l2_ptr, v0_ptr});
      cand.addUserFloat("min_dr", dr_info.first);
      cand.addUserFloat("max_dr", dr_info.second);

      // built B
      if (!pre_vtx_selection_(cand))
        continue;

      KinVtxFitter fitter({leptons_ttracks->at(l1_idx), leptons_ttracks->at(l2_idx), v0_ttracks->at(v0_idx)},
                          {l1_ptr->mass(), l2_ptr->mass(), v0_ptr->mass()},
                          {bph::LEP_SIGMA, bph::LEP_SIGMA, v0_ptr->userFloat("massErr")});

      if (!fitter.success())
        continue;

      cand.setVertex(reco::Candidate::Point(fitter.fitted_vtx().x(), fitter.fitted_vtx().y(), fitter.fitted_vtx().z()));

      cand.addUserFloat("sv_chi2", fitter.chi2());
      cand.addUserFloat("sv_ndof", fitter.dof());
      cand.addUserFloat("sv_prob", fitter.prob());
      cand.addUserFloat("fitted_mll", (fitter.daughter_p4(0) + fitter.daughter_p4(1)).mass());
      cand.addUserFloat("fitted_v0_mass", fitter.daughter_p4(2).mass());

      auto fit_p4 = fitter.fitted_p4();
      cand.addUserFloat("fitted_pt", fit_p4.pt());
      cand.addUserFloat("fitted_eta", fit_p4.eta());
      cand.addUserFloat("fitted_phi", fit_p4.phi());
      cand.addUserFloat("fitted_mass", fitter.fitted_candidate().mass());
      cand.addUserFloat("fitted_massErr", sqrt(fitter.fitted_candidate().kinematicParametersError().matrix()(6, 6)));
      cand.addUserFloat("cos_theta_2D", bph::cos_theta_2D(fitter, *beamspot, cand.p4()));
      cand.addUserFloat("fitted_cos_theta_2D", bph::cos_theta_2D(fitter, *beamspot, fit_p4));

      auto lxy = bph::l_xy(fitter, *beamspot);
      cand.addUserFloat("l_xy", lxy.value());
      cand.addUserFloat("l_xy_unc", lxy.error());

      TrajectoryStateOnSurface tsos =
          extrapolator.extrapolate(v0_ttracks->at(v0_idx).impactPointState(), fitter.fitted_vtx());
      std::pair<bool, Measurement1D> cur2DIP =
          bph::signedTransverseImpactParameter(tsos, fitter.fitted_refvtx(), *beamspot);
      cand.addUserFloat("v0_svip2d", cur2DIP.second.value());
      cand.addUserFloat("v0_svip2d_err", cur2DIP.second.error());

      if (!post_vtx_selection_(cand))
        continue;

      cand.addUserFloat("vtx_x", cand.vx());
      cand.addUserFloat("vtx_y", cand.vy());
      cand.addUserFloat("vtx_z", cand.vz());

      const auto &covMatrix = fitter.fitted_vtx_uncertainty();
      cand.addUserFloat("vtx_cxx", covMatrix.cxx());
      cand.addUserFloat("vtx_cyy", covMatrix.cyy());
      cand.addUserFloat("vtx_czz", covMatrix.czz());
      cand.addUserFloat("vtx_cyx", covMatrix.cyx());
      cand.addUserFloat("vtx_czx", covMatrix.czx());
      cand.addUserFloat("vtx_czy", covMatrix.czy());

      // refitted daughters (leptons/tracks)
      std::vector<std::string> dnames{"l1", "l2", "v0"};
      for (size_t idaughter = 0; idaughter < dnames.size(); idaughter++) {
        cand.addUserFloat("fitted_" + dnames[idaughter] + "_pt", fitter.daughter_p4(idaughter).pt());
        cand.addUserFloat("fitted_" + dnames[idaughter] + "_eta", fitter.daughter_p4(idaughter).eta());
        cand.addUserFloat("fitted_" + dnames[idaughter] + "_phi", fitter.daughter_p4(idaughter).phi());
      }

      // compute isolation
      std::vector<float> isos = bph::TrackerIsolation(pu_tracks, cand, dnames);
      for (size_t idaughter = 0; idaughter < dnames.size(); idaughter++) {
        cand.addUserFloat(dnames[idaughter] + "_iso04", isos[idaughter]);
      }
      float constraint_sv_prob = -9;
      float constraint_pt = -9;
      float constraint_eta = -9;
      float constraint_phi = -9;
      float constraint_mass = -9;
      float constraint_massErr = -9;
      float constraint_mll = -9;
      const double dilepton_mass = ll_ptr->userFloat("fitted_mass");
      const double jpsi_bin[2] = {2.8, 3.35};
      const double psi2s_bin[2] = {3.45, 3.85};
      if (dilepton_constraint_ && ((dilepton_mass > jpsi_bin[0] && dilepton_mass < jpsi_bin[1]) ||
                                   (dilepton_mass > psi2s_bin[0] && dilepton_mass < psi2s_bin[1]))) {
        ParticleMass JPsi_mass = 3.0969;   // Jpsi mass 3.096900±0.000006
        ParticleMass Psi2S_mass = 3.6861;  // Psi2S mass 3.6861093±0.0000034
        ParticleMass mass_constraint = (dilepton_mass < jpsi_bin[1]) ? JPsi_mass : Psi2S_mass;

        // Mass constraint is applied to the first two particles in the
        // "particles" vector Make sure that the first two particles are the
        // ones you want to constrain
        KinVtxFitter constraint_fitter(
            {leptons_ttracks->at(l1_idx), leptons_ttracks->at(l2_idx), v0_ttracks->at(v0_idx)},
            {l1_ptr->mass(), l2_ptr->mass(), v0_ptr->mass()},
            {bph::LEP_SIGMA, bph::LEP_SIGMA, v0_ptr->userFloat("massErr")},
            mass_constraint);
        if (constraint_fitter.success()) {
          auto constraint_p4 = constraint_fitter.fitted_p4();
          constraint_sv_prob = constraint_fitter.prob();
          constraint_pt = constraint_p4.pt();
          constraint_eta = constraint_p4.eta();
          constraint_phi = constraint_p4.phi();
          constraint_mass = constraint_fitter.fitted_candidate().mass();
          constraint_massErr = sqrt(constraint_fitter.fitted_candidate().kinematicParametersError().matrix()(6, 6));
          constraint_mll = (constraint_fitter.daughter_p4(0) + constraint_fitter.daughter_p4(1)).mass();
        }
      }
      cand.addUserFloat("constraint_sv_prob", constraint_sv_prob);
      cand.addUserFloat("constraint_pt", constraint_pt);
      cand.addUserFloat("constraint_eta", constraint_eta);
      cand.addUserFloat("constraint_phi", constraint_phi);
      cand.addUserFloat("constraint_mass", constraint_mass);
      cand.addUserFloat("constraint_massErr", constraint_massErr);
      cand.addUserFloat("constraint_mll", constraint_mll);

      ret_val->push_back(cand);
    }
  }
  evt.put(std::move(ret_val));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BToV0LLBuilder);
