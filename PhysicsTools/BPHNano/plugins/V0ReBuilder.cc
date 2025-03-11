/////////////////////////////// BToV0LLBuilder ///////////////////////////////
/// original authors: G Karathanasis (CERN),  G Melachroinos (NKUA)
// takes V0 cabds from CMSSW and creates useful V0

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidateFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"

#include <limits>
#include <algorithm>
#include <vector>
#include <memory>
#include <map>
#include <string>

#include "KinVtxFitter.h"
#include "helper.h"

class V0ReBuilder : public edm::global::EDProducer<> {

  // perhaps we need better structure here (begin run etc)
public:
  typedef std::vector<reco::TransientTrack> TransientTrackCollection;
  typedef std::vector<reco::VertexCompositePtrCandidate> V0Collection;

  explicit V0ReBuilder(const edm::ParameterSet &cfg):
    theB_(esConsumes(edm::ESInputTag{"", "TransientTrackBuilder"})),
    trk_selection_{cfg.getParameter<std::string>("trkSelection")},
    pre_vtx_selection_{cfg.getParameter<std::string>("V0Selection")},
    post_vtx_selection_{cfg.getParameter<std::string>("postVtxSelection")},
    v0s_{consumes<V0Collection>( cfg.getParameter<edm::InputTag>("V0s") )},
    beamspot_{consumes<reco::BeamSpot>( cfg.getParameter<edm::InputTag>("beamSpot") )},
    track_match_{consumes<edm::Association<pat::CompositeCandidateCollection>>( cfg.getParameter<edm::InputTag>("track_match") )},
    isLambda_{cfg.getParameter<bool>("isLambda")}    
  {
    produces<pat::CompositeCandidateCollection>("SelectedV0Collection");
    produces<TransientTrackCollection>("SelectedV0TransientCollection");
  }

  ~V0ReBuilder() override {}

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {}

private:
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theB_;
  const StringCutObjectSelector<pat::PackedCandidate> trk_selection_;
  const StringCutObjectSelector<reco::VertexCompositePtrCandidate> pre_vtx_selection_;
  const StringCutObjectSelector<pat::CompositeCandidate> post_vtx_selection_;
  const edm::EDGetTokenT<V0Collection> v0s_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  const edm::EDGetTokenT<edm::Association<pat::CompositeCandidateCollection>> track_match_;
  const bool isLambda_;
};

void V0ReBuilder::produce(edm::StreamID, edm::Event &evt, edm::EventSetup const &iSetup) const {

  //input
  auto const theB = &iSetup.getData(theB_);
  edm::Handle<V0Collection> V0s;
  evt.getByToken(v0s_, V0s);
  edm::Handle<reco::BeamSpot> beamspot;
  evt.getByToken(beamspot_, beamspot);

  auto& track_match = evt.get(track_match_);

  // output
  std::unique_ptr<pat::CompositeCandidateCollection> ret_val(new pat::CompositeCandidateCollection());
  std::unique_ptr<TransientTrackCollection> trans_out( new TransientTrackCollection );

  size_t v0_idx = 0;
  for (reco::VertexCompositePtrCandidateCollection::const_iterator v0 = V0s->begin(); v0 != V0s->end(); v0++) {

    reco::VertexCompositePtrCandidate V0 = V0s->at(v0_idx);
    v0_idx++;

    // selection on V0s
    if (v0->numberOfDaughters() != 2) continue;
    if (!pre_vtx_selection_(V0)) continue;

    pat::PackedCandidate v0daughter1 = *(dynamic_cast<const pat::PackedCandidate *>(v0->daughter(0)));
    pat::PackedCandidate v0daughter2 = *(dynamic_cast<const pat::PackedCandidate *>(v0->daughter(1)));

    if (!v0daughter1.hasTrackDetails()) continue;
    if (!v0daughter2.hasTrackDetails()) continue;

    if (fabs(v0daughter1.pdgId()) != 211) continue;// This cut do not affect the Lambda->proton pion candidates
    if (fabs(v0daughter2.pdgId()) != 211) continue;// This cut do not affect the Lambda->proton pion candidates

    if (!trk_selection_(v0daughter1) || !trk_selection_(v0daughter2)) continue;

    reco::TransientTrack v0daughter1_ttrack; // 1st daughter, leading daughter to be assigned. Proton mass will be assigned for the Lambda->Proton Pion mode, Pion mass will be assigned for the Kshort->PionPion mode. 
    reco::TransientTrack v0daughter2_ttrack; // 2nd daughter, subleading daughter to be assigned. It hass always the pion mass

    if (v0daughter1.p()>v0daughter2.p())
    {
	v0daughter1_ttrack = theB->build(v0daughter1.bestTrack());
	v0daughter2_ttrack = theB->build(v0daughter2.bestTrack());
    }
    else
    {
	v0daughter1_ttrack = theB->build(v0daughter2.bestTrack());
	v0daughter2_ttrack = theB->build(v0daughter1.bestTrack());
    }

    float Track1_mass = (isLambda_) ? PROT_MASS : PI_MASS;
    float Track1_sigma = PI_SIGMA;
    float Track2_mass = PI_MASS;
    float Track2_sigma = PI_SIGMA;    
    // create V0 vertex
    KinVtxFitter fitter(
    {v0daughter1_ttrack, v0daughter2_ttrack},
    {Track1_mass, Track2_mass},
    {Track1_sigma,Track2_sigma} );

    if (!fitter.success()) continue;

    pat::CompositeCandidate cand;
    cand.setVertex( reco::Candidate::Point(
                      fitter.fitted_vtx().x(),
                      fitter.fitted_vtx().y(),
                      fitter.fitted_vtx().z()                                                         )
                  );
    auto fit_p4 = fitter.fitted_p4();
    cand.setP4(fit_p4);

    cand.setCharge(v0daughter1.charge() + v0daughter2.charge());
    cand.addUserFloat("sv_chi2", fitter.chi2());
    cand.addUserFloat("sv_prob", fitter.prob());
    cand.addUserFloat("fitted_mass", fitter.fitted_candidate().mass());
    cand.addUserFloat("massErr",
                      sqrt(fitter.fitted_candidate().kinematicParametersError().matrix()(6, 6)));
    cand.addUserFloat("cos_theta_2D",
                      cos_theta_2D(fitter, *beamspot, cand.p4()));
    cand.addUserFloat("fitted_cos_theta_2D",
                      cos_theta_2D(fitter, *beamspot, fit_p4));
    auto lxy = l_xy(fitter, *beamspot);
    cand.addUserFloat("l_xy", lxy.value());
    cand.addUserFloat("l_xy_unc", lxy.error());

    if (!post_vtx_selection_(cand)) continue;

    cand.addUserFloat("vtx_x", cand.vx());
    cand.addUserFloat("vtx_y", cand.vy());
    cand.addUserFloat("vtx_z", cand.vz());

    const auto& covMatrix = fitter.fitted_vtx_uncertainty();
    cand.addUserFloat("vtx_cxx", covMatrix.cxx());
    cand.addUserFloat("vtx_cyy", covMatrix.cyy());
    cand.addUserFloat("vtx_czz", covMatrix.czz());
    cand.addUserFloat("vtx_cyx", covMatrix.cyx());
    cand.addUserFloat("vtx_czx", covMatrix.czx());
    cand.addUserFloat("vtx_czy", covMatrix.czy());

    cand.addUserFloat("prefit_mass", v0->mass());
    int trk1 = 0;
    int trk2 = 1;
    if (fitter.daughter_p4(0).pt() < fitter.daughter_p4(1).pt()) {
      trk1 = 1;
      trk2 = 0;
    }
    cand.addUserFloat("trk1_pt", fitter.daughter_p4(trk1).pt());
    cand.addUserFloat("trk1_eta", fitter.daughter_p4(trk1).eta());
    cand.addUserFloat("trk1_phi", fitter.daughter_p4(trk1).phi());
    cand.addUserFloat("trk2_pt", fitter.daughter_p4(trk2).pt());
    cand.addUserFloat("trk2_eta", fitter.daughter_p4(trk2).eta());
    cand.addUserFloat("trk2_phi", fitter.daughter_p4(trk2).phi());

    // track match
    auto trk1_ptr = v0->daughterPtr(trk1);
    auto trk1_matched_ref = track_match.get(trk1_ptr.id(), trk1_ptr.key());
    auto trk2_ptr = v0->daughterPtr(trk2);
    auto trk2_matched_ref = track_match.get(trk2_ptr.id(), trk2_ptr.key());

    cand.addUserInt("trk1_idx", trk1_matched_ref.key());
    cand.addUserInt("trk2_idx", trk2_matched_ref.key());

    // save
    ret_val->push_back(cand);
    auto V0TT = fitter.fitted_candidate_ttrk();
    trans_out->emplace_back(V0TT);
  }

  evt.put(std::move(ret_val), "SelectedV0Collection");
  evt.put(std::move(trans_out), "SelectedV0TransientCollection");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(V0ReBuilder);
