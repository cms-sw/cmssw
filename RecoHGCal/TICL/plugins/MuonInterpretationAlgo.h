#ifndef RecoHGCal_TICL_MuonInterpretationAlgo_h
#define RecoHGCal_TICL_MuonInterpretationAlgo_h

// Muon interpretation for TICL candidates. A muon crosses HGCAL as a MIP: it should
// NOT be built from the small calorimetric deposit it leaves (as the general
// interpretation would), but recognised as a muon and built from the track momentum,
// with its MIP tracksters consumed so they do not resurface as neutral candidates.
//
// This is where the Muon-POG HGCAL muon identification belongs. For each candidate
// track this algo:
//   1. points/propagates the track into HGCAL and collects the layer clusters /
//      tracksters in an (eta,phi) window around the trajectory;
//   2. checks the trajectory does NOT coincide with an energetic trackster (a muon is
//      MIP-like, not a shower) - the "not energetic" requirement;
//   3. runs a neural network over the surrounding layer clusters to decide muon vs not
//      (ONNX model, path configurable like the other TICL inference). The trained
//      weights are a Muon-POG deliverable; until a model is provided the decision falls
//      back to the rule-based MIP/not-energetic test so the algo is runnable.
// A track accepted as a muon has its MIP tracksters masked (consumed) and is reported
// so the producer builds a muon candidate from the track momentum.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <memory>
#include <string>

namespace cms::Ort {
  class ONNXRuntime;
}

namespace ticl {

  class MuonInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    MuonInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);
    ~MuonInterpretationAlgo() override;

    void makeCandidates(const Inputs &input,
                        edm::Handle<MtdHostCollection> inputTiming_h,
                        std::vector<Trackster> &resultTracksters,
                        std::vector<int> &resultCandidate,
                        std::vector<bool> &maskedTracksters) override;

    void initialize(const HGCalDDDConstants *hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    static void fillPSetDescription(edm::ParameterSetDescription &iDesc);

  private:
    // NN muon score over the layer clusters around the trajectory. Falls back to the
    // rule-based MIP/not-energetic decision when no ONNX model is configured.
    bool isMuonLike(double nearbyEnergy, unsigned nNearbyTracksters) const;

    // (eta,phi) window used to collect tracksters around the track direction.
    const double delta_tk_ts_;
    // Max summed raw energy of the tracksters around the trajectory for a MIP-like
    // (muon) signature; above this the track points to a shower and is not a muon.
    const double mip_energy_max_;
    // ONNX muon-ID model (empty -> rule-based fallback). Loaded via the global cache.
    const std::string onnx_model_path_;

    const HGCalDDDConstants *hgcons_;
    hgcal::RecHitTools rhtools_;
    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;
  };

}  // namespace ticl

#endif
