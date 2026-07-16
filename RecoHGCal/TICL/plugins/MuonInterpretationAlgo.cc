#include "RecoHGCal/TICL/plugins/MuonInterpretationAlgo.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// v0 of the Muon-POG HGCAL muon interpretation. It captures the architecture (point the
// track into HGCAL, collect the tracksters around the trajectory, require a MIP / not-
// energetic signature, and consume the MIP tracksters) with a rule-based decision as a
// placeholder for the neural network. TODOs marked below are the upgrade path:
//   - replace the direction-based association with full track propagation to the HGCAL
//     layers (as GeneralInterpretationAlgo does) and per-layer layer-cluster collection;
//   - run the ONNX muon-ID network over those layer clusters (onnx_model_path_).

using namespace ticl;

MuonInterpretationAlgo::MuonInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : TICLInterpretationAlgoBase(conf, iC),
      delta_tk_ts_(conf.getParameter<double>("delta_tk_ts")),
      mip_energy_max_(conf.getParameter<double>("mip_energy_max")),
      onnx_model_path_(conf.getParameter<std::string>("onnx_model_path")),
      hgcons_(nullptr) {}

MuonInterpretationAlgo::~MuonInterpretationAlgo() {}

void MuonInterpretationAlgo::initialize(const HGCalDDDConstants *hgcons,
                                        const hgcal::RecHitTools rhtools,
                                        const edm::ESHandle<MagneticField> bfieldH,
                                        const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
  bfield_ = bfieldH;
  propagator_ = propH;
}

bool MuonInterpretationAlgo::isMuonLike(double nearbyEnergy, unsigned /*nNearbyTracksters*/) const {
  // TODO: when onnx_model_path_ is set, run the muon-ID network over the surrounding
  // layer clusters and use its score here. Until then, the Muon-POG "not energetic"
  // requirement is the decision: a muon deposits a MIP, so the tracksters around its
  // trajectory carry little energy.
  return nearbyEnergy < mip_energy_max_;
}

void MuonInterpretationAlgo::makeCandidates(const Inputs &input,
                                            edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                            std::vector<Trackster> &resultTracksters,
                                            std::vector<int> &resultCandidate,
                                            std::vector<bool> &maskedTracksters) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;
  if (maskedTracksters.size() < tracksters.size())
    maskedTracksters.resize(tracksters.size(), false);

  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    if (!maskTracks[iTrack])
      continue;
    const auto &tk = tracks[iTrack];
    // TODO: propagate the track to the HGCAL layers; here we use the outer track
    // direction (or the momentum direction) as the HGCAL entry direction, which is a
    // good approximation for a minimally-bending muon.
    const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
    const double tkEta = dir.eta();
    const double tkPhi = dir.phi();

    // Collect the tracksters whose barycenter lies within the (eta,phi) window around
    // the trajectory, and sum their raw energy (the "is it energetic?" measure).
    std::vector<unsigned> nearby;
    double nearbyEnergy = 0.;
    for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
      if (maskedTracksters[iTs])
        continue;
      const auto &bary = tracksters[iTs].barycenter();
      if (bary.eta() * tkEta < 0.)  // same endcap
        continue;
      if (reco::deltaR(bary.eta(), bary.phi(), tkEta, tkPhi) < delta_tk_ts_) {
        nearby.push_back(iTs);
        nearbyEnergy += tracksters[iTs].raw_energy();
      }
    }

    if (!isMuonLike(nearbyEnergy, nearby.size())) {
      // Trajectory points to a shower: this is not a muon. Flag it so the producer
      // routes it to the general interpretation instead of building a muon candidate.
      resultCandidate[iTrack] = kMuonRejected;
      continue;
    }

    // Muon: consume the MIP tracksters (mask them) and merge them into one trackster so
    // the producer can attach it to the muon candidate; the candidate energy itself is
    // taken from the track momentum by the producer.
    if (!nearby.empty()) {
      Trackster muonTrackster;
      for (unsigned iTs : nearby) {
        muonTrackster.mergeTracksters(tracksters[iTs]);
        maskedTracksters[iTs] = true;
      }
      resultCandidate[iTrack] = static_cast<int>(resultTracksters.size());
      resultTracksters.push_back(muonTrackster);
    } else {
      resultCandidate[iTrack] = -1;  // muon with no HGCAL deposit: track-only candidate
    }
  }
}

void MuonInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<double>("delta_tk_ts", 0.1)->setComment("(eta,phi) window to collect tracksters around the trajectory.");
  desc.add<double>("mip_energy_max", 10.0)
      ->setComment("Max summed raw energy [GeV] around the trajectory for a MIP-like (muon) signature.");
  desc.add<std::string>("onnx_model_path", "")
      ->setComment("ONNX muon-ID model; empty falls back to the rule-based MIP test.");
  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
