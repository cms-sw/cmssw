/////////////////////////////// DiTrackBuilder ///////////////////////////////
/// original authors: G Karathanasis (CERN),  G Melachroinos (NKUA)
// takes selected track collection and a mass hypothesis and produces ditrack ca
// -ndidates

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
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
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "helper.h"

class DiTrackBuilder : public edm::global::EDProducer<> {
public:
  typedef std::vector<reco::TransientTrack> TransientTrackCollection;

  explicit DiTrackBuilder(const edm::ParameterSet &cfg)
      : trk1_selection_{cfg.getParameter<std::string>("trk1Selection")},
        trk2_selection_{cfg.getParameter<std::string>("trk2Selection")},
        pre_vtx_selection_{cfg.getParameter<std::string>("preVtxSelection")},
        post_vtx_selection_{cfg.getParameter<std::string>("postVtxSelection")},
        pfcands_{consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("tracks"))},
        ttracks_{consumes<TransientTrackCollection>(cfg.getParameter<edm::InputTag>("transientTracks"))},
        beamspot_{consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))},
        trk1_mass_{cfg.getParameter<double>("trk1Mass")},
        trk2_mass_{cfg.getParameter<double>("trk2Mass")} {
    // output
    produces<pat::CompositeCandidateCollection>();
  }

  ~DiTrackBuilder() override {}

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  const StringCutObjectSelector<pat::CompositeCandidate> trk1_selection_;      // cuts on leading cand
  const StringCutObjectSelector<pat::CompositeCandidate> trk2_selection_;      // sub-leading cand
  const StringCutObjectSelector<pat::CompositeCandidate> pre_vtx_selection_;   // cut on the di-track before the SV fit
  const StringCutObjectSelector<pat::CompositeCandidate> post_vtx_selection_;  // cut on the di-track after the SV fit
  const edm::EDGetTokenT<pat::CompositeCandidateCollection>
      pfcands_;                                               // input PF cands this is sorted in pT in previous step
  const edm::EDGetTokenT<TransientTrackCollection> ttracks_;  // input TTracks of PF cands
  const edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  double trk1_mass_;
  double trk2_mass_;
};

void DiTrackBuilder::produce(edm::StreamID, edm::Event &evt, edm::EventSetup const &) const {
  // inputs
  edm::Handle<pat::CompositeCandidateCollection> pfcands;
  evt.getByToken(pfcands_, pfcands);
  edm::Handle<TransientTrackCollection> ttracks;
  evt.getByToken(ttracks_, ttracks);
  edm::Handle<reco::BeamSpot> beamspot;
  evt.getByToken(beamspot_, beamspot);

  // output
  std::unique_ptr<pat::CompositeCandidateCollection> ditrack_out(new pat::CompositeCandidateCollection());

  // main loop
  for (size_t trk1_idx = 0; trk1_idx < pfcands->size(); ++trk1_idx) {
    edm::Ptr<pat::CompositeCandidate> trk1_ptr(pfcands, trk1_idx);
    if (!trk1_selection_(*trk1_ptr))
      continue;

    for (size_t trk2_idx = trk1_idx + 1; trk2_idx < pfcands->size(); ++trk2_idx) {
      edm::Ptr<pat::CompositeCandidate> trk2_ptr(pfcands, trk2_idx);
      // if (trk1_ptr->charge() == trk2_ptr->charge()) continue;
      if (!trk2_selection_(*trk2_ptr))
        continue;

      pat::CompositeCandidate ditrack_cand;
      auto trk1_p4 = trk1_ptr->polarP4();
      auto trk2_p4 = trk2_ptr->polarP4();
      trk1_p4.SetM(bph::K_MASS);
      trk2_p4.SetM(bph::K_MASS);
      ditrack_cand.setP4(trk1_p4 + trk2_p4);
      ditrack_cand.setCharge(trk1_ptr->charge() + trk2_ptr->charge());
      ditrack_cand.addUserFloat("trk_deltaR", reco::deltaR(*trk1_ptr, *trk2_ptr));
      // save indices
      ditrack_cand.addUserInt("trk1_idx", trk1_idx);
      ditrack_cand.addUserInt("trk2_idx", trk2_idx);
      // save cands
      ditrack_cand.addUserCand("trk1", trk1_ptr);
      ditrack_cand.addUserCand("trk2", trk2_ptr);

      ditrack_cand.addUserFloat("trk_dz", trk1_ptr->vz() - trk2_ptr->vz());
      ditrack_cand.addUserFloat("unfitted_mass_KK", (trk1_p4 + trk2_p4).M());
      trk1_p4.SetM(bph::K_MASS);
      trk2_p4.SetM(bph::PI_MASS);
      ditrack_cand.addUserFloat("unfitted_mass_Kpi", (trk1_p4 + trk2_p4).M());
      trk2_p4.SetM(bph::K_MASS);
      trk1_p4.SetM(bph::PI_MASS);
      ditrack_cand.addUserFloat("unfitted_mass_piK", (trk1_p4 + trk2_p4).M());
      trk2_p4.SetM(bph::K_MASS);
      trk1_p4.SetM(bph::K_MASS);

      if (!pre_vtx_selection_(ditrack_cand))
        continue;

      KinVtxFitter fitter(
          {ttracks->at(trk1_idx), ttracks->at(trk2_idx)}, {bph::K_MASS, bph::K_MASS}, {bph::K_SIGMA, bph::K_SIGMA}
          // K and PI sigma equal...
      );
      if (!fitter.success())
        continue;
      ditrack_cand.addUserFloat("fitted_mass_KK", fitter.fitted_candidate().mass());
      ditrack_cand.addUserFloat("fitted_mass_KK_Err",
                                sqrt(fitter.fitted_candidate().kinematicParametersError().matrix()(6, 6)));
      // fits required in order to calculate the error of the mass for each mass
      // hypothesis.
      KinVtxFitter fitter_Kpi(
          {ttracks->at(trk1_idx), ttracks->at(trk2_idx)}, {bph::K_MASS, bph::PI_MASS}, {bph::K_SIGMA, bph::K_SIGMA}
          // K and PI sigma equal...
      );
      if (!fitter_Kpi.success())
        continue;
      ditrack_cand.addUserFloat("fitted_mass_Kpi", fitter_Kpi.fitted_candidate().mass());
      ditrack_cand.addUserFloat("fitted_mass_Kpi_Err",
                                sqrt(fitter_Kpi.fitted_candidate().kinematicParametersError().matrix()(6, 6)));
      KinVtxFitter fitter_piK(
          {ttracks->at(trk1_idx), ttracks->at(trk2_idx)}, {bph::PI_MASS, bph::K_MASS}, {bph::K_SIGMA, bph::K_SIGMA}
          // K and PI sigma equal...
      );
      if (!fitter_piK.success())
        continue;
      ditrack_cand.addUserFloat("fitted_mass_piK", fitter_piK.fitted_candidate().mass());
      ditrack_cand.addUserFloat("fitted_mass_piK_Err",
                                sqrt(fitter_piK.fitted_candidate().kinematicParametersError().matrix()(6, 6)));

      ditrack_cand.setVertex(
          reco::Candidate::Point(fitter.fitted_vtx().x(), fitter.fitted_vtx().y(), fitter.fitted_vtx().z()));
      // save quantities after fit
      auto lxy = bph::l_xy(fitter, *beamspot);
      ditrack_cand.addUserFloat("l_xy", lxy.value());
      ditrack_cand.addUserFloat("l_xy_unc", lxy.error());
      ditrack_cand.addUserInt("sv_ok", fitter.success() ? 1 : 0);
      auto fit_p4 = fitter.fitted_p4();
      ditrack_cand.addUserFloat("fitted_cos_theta_2D", bph::cos_theta_2D(fitter, *beamspot, fit_p4));
      // The following quantities do not independent on the mass hypothesis
      ditrack_cand.addUserFloat("sv_chi2", fitter.chi2());
      ditrack_cand.addUserFloat("sv_ndof", fitter.dof());
      ditrack_cand.addUserFloat("sv_prob", fitter.prob());
      ditrack_cand.addUserFloat("fitted_pt", fitter.fitted_candidate().globalMomentum().perp());
      ditrack_cand.addUserFloat("fitted_eta", fitter.fitted_candidate().globalMomentum().eta());
      ditrack_cand.addUserFloat("fitted_phi", fitter.fitted_candidate().globalMomentum().phi());
      ditrack_cand.addUserFloat("vtx_x", ditrack_cand.vx());
      ditrack_cand.addUserFloat("vtx_y", ditrack_cand.vy());
      ditrack_cand.addUserFloat("vtx_z", ditrack_cand.vz());
      const auto &covMatrix = fitter.fitted_vtx_uncertainty();
      ditrack_cand.addUserFloat("vtx_cxx", covMatrix.cxx());
      ditrack_cand.addUserFloat("vtx_cyy", covMatrix.cyy());
      ditrack_cand.addUserFloat("vtx_czz", covMatrix.czz());
      ditrack_cand.addUserFloat("vtx_cyx", covMatrix.cyx());
      ditrack_cand.addUserFloat("vtx_czx", covMatrix.czx());
      ditrack_cand.addUserFloat("vtx_czy", covMatrix.czy());

      // after fit selection
      if (!post_vtx_selection_(ditrack_cand))
        continue;
      ditrack_out->emplace_back(ditrack_cand);
    }  // end for(size_t trk2_idx = trk1_idx + 1
  }  // for(size_t trk1_idx = 0

  evt.put(std::move(ditrack_out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DiTrackBuilder);
