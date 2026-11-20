#ifndef RecoTICL_Linking_TracksterLinkingbyCornetto_h
#define RecoTICL_Linking_TracksterLinkingbyCornetto_h

// Trackster linking by a parent tree. The pair test, the parent search and the emission rule are in
// RecoTICL/Linking/interface/CornettoLinking.h.
//
// Pair test with anchor a and other trackster b: both are in the same endcap, |deta| and |dphi| are at most
// etaWindow, and b is inside the cone around the axis from the origin to the barycenter of a. An EM anchor uses the
// narrower EM cone. The time gate and the EM/hadronic type veto can refuse the pair.
//
// Grouping: the parent of b is the highest-ranked trackster a above b (more energy; at equal energy the smaller
// index) for which the pair (a, b) is compatible. Each trackster follows its parents to its root. It stays in the
// group of its root only if the pair (root, trackster) is compatible; else it is a group of its own. The optional
// second pass applies the same rule to the groups, with the merge cone. A group is emitted when it passes the
// emission rule. A trackster with a non-finite barycenter or energy is not emitted.

#include <cmath>
#include <vector>

#include "RecoTICL/Linking/interface/CornettoFeatures.h"
#include "RecoTICL/Linking/interface/CornettoGeometry.h"
#include "RecoTICL/Linking/interface/CornettoLinking.h"
#include "RecoTICL/Linking/interface/TracksterLinkingAlgoBase.h"

namespace ticl {

  class TracksterLinkingbyCornetto : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbyCornetto(const edm::ParameterSet& conf,
                               edm::ConsumesCollector iC,
                               cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);
    ~TracksterLinkingbyCornetto() override = default;

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;

    void initialize(const HGCalDDDConstants* hgcons,
                    const ticlgeom::Tools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override {}

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.add<float>("etaWindow", 0.3f)
          ->setComment("Max |deta| and |dphi| between the barycenters. Must be in (0, pi - 2 pi / 128).");
      iDesc.add<float>("maxLongitudinalDistance", 60.f)
          ->setComment("Max separation [cm] downstream of the anchor along its axis. Must be positive.");
      iDesc.add<float>("maxBackwardDistance", 20.f)
          ->setComment("Max separation [cm] upstream of the anchor along its axis. Must not be negative.");
      iDesc.add<float>("transverseRadius0", 4.f)
          ->setComment("Max transverse distance at zero separation [cm]. Must not be negative.");
      iDesc.add<float>("transverseSlope", 0.05f)
          ->setComment("Growth of the max transverse distance per cm of separation. Must not be negative.");
      iDesc.add<float>("emTransverseRadius0", 1.5f)
          ->setComment(
              "Max transverse distance at zero separation [cm] for an EM anchor. Must not be negative or above "
              "transverseRadius0.");
      iDesc.add<float>("emTransverseSlope", 0.05f)
          ->setComment(
              "Growth of the max transverse distance per cm of separation for an EM anchor. Must not be negative or "
              "above transverseSlope.");
      iDesc.add<float>("timeCompatibilityNSigma", 3.f)
          ->setComment(
              "Max |time difference at the anchor barycenter depth| in combined sigmas. Applies when both time "
              "errors are positive. Values <= 0 disable the gate.");
      iDesc.add<float>("timeResolutionFloor", 0.05f)
          ->setComment("Time resolution [ns] added in quadrature to the combined time error. Must not be negative.");
      iDesc.add<float>("typeVetoProbability", 0.6f)
          ->setComment(
              "EM/hadronic veto threshold: 0 disables the veto, else it must be in [0.5, 1]. A pair is refused if "
              "one EM probability is above this value and the other below 1 - this value. Applies only when both "
              "carry a PID.");
      iDesc
          .add<std::vector<int>>("emCategories",
                                 {static_cast<int>(Trackster::ParticleType::photon),
                                  static_cast<int>(Trackster::ParticleType::electron),
                                  static_cast<int>(Trackster::ParticleType::neutral_pion)})
          ->setComment(
              "Trackster::ParticleType slots of the EM probability: the sum of their PID probabilities divided by "
              "the sum over all slots. At least one value, each in [0, 7], no duplicates.");
      iDesc.add<float>("minEmittedEnergy", 1.f)
          ->setComment("Min raw energy [GeV] of an emitted group. Applies after linking.");
      iDesc.add<float>("minEmittedPt", 0.5f)
          ->setComment("Min raw pt [GeV] of an emitted group. Applies after linking.");
      iDesc.add<float>("minEmittedEnergyEM", 1.f)
          ->setComment("Min raw energy [GeV] of an emitted EM group (EM probability above typeVetoProbability).");
      iDesc.add<float>("minEmittedPtEM", 0.5f)->setComment("Min raw pt [GeV] of an emitted EM group.");
      iDesc.add<float>("minEmittedEnergyAnyPt", 0.f)
          ->setComment(
              "A group at or above this raw energy [GeV] is emitted at any pt. Values <= 0 disable it. The minimum "
              "energy still applies.");
      iDesc.add<bool>("mergeFragments", false)
          ->setComment("Second pass: an emitted group can join an emitted group of higher energy.");
      iDesc.add<bool>("mergeDropped", false)
          ->setComment("Second pass: a group that is not emitted can join an emitted group.");
      iDesc.add<bool>("mergeHadronicOnly", true)->setComment("Second pass: refuse a pair with an EM group.");
      iDesc.add<float>("mergeMinParentEnergy", 0.f)->setComment("Second pass: min raw energy [GeV] of a parent.");
      iDesc.add<float>("mergeMaxLongitudinalDistance", 60.f)
          ->setComment("Second pass: max separation [cm] downstream of the parent. Must be positive.");
      iDesc.add<float>("mergeMaxBackwardDistance", 20.f)
          ->setComment("Second pass: max separation [cm] upstream of the parent. Must not be negative.");
      iDesc.add<float>("mergeTransverseRadius0", 4.f)
          ->setComment("Second pass: max transverse distance at zero separation [cm]. Must not be negative.");
      iDesc.add<float>("mergeTransverseSlope", 0.05f)
          ->setComment(
              "Second pass: growth of the max transverse distance per cm of separation. Must not be negative.");
      iDesc.add<float>("mergeEMTransverseRadius0", 1.5f)
          ->setComment("Second pass: max transverse distance at zero separation [cm] for an EM parent.");
      iDesc.add<float>("mergeEMTransverseSlope", 0.05f)
          ->setComment("Second pass: growth of the max transverse distance per cm of separation for an EM parent.");
      TracksterLinkingAlgoBase::fillPSetDescription(iDesc);
    }

  private:
    using Features = cornetto::Features;

    Features computeFeatures(const Inputs& input) const;
    bool compatible(const Features& f, unsigned int a, unsigned int b) const {
      return cornetto::compatible(cuts_, cone_, emCone_, f, a, b);
    }
    bool mergeCompatible(const Features& g, unsigned int a, unsigned int b) const;
    bool emits(const Features& g, unsigned int k) const {
      return g.valid[k] && cornetto::emits(emission_, g.energy[k], g.energy[k] / std::cosh(g.eta[k]), g.em[k]);
    }
    // The groups after the second pass.
    std::vector<std::vector<unsigned int>> mergeGroups(const Features& f,
                                                       std::vector<std::vector<unsigned int>> groups) const;

    const cornetto::PairCuts cuts_;
    const cornetto::Cone cone_;
    const cornetto::Cone emCone_;  // cone of an EM anchor, inside cone_
    const std::vector<int> emCategories_;
    const cornetto::EmissionRule emission_;
    const bool mergeFragments_;
    const bool mergeDropped_;
    const bool mergeHadronicOnly_;
    const float mergeMinParentEnergy_;
    const cornetto::Cone mergeCone_;
    const cornetto::Cone mergeEMCone_;  // merge cone of an EM parent, inside mergeCone_
  };

}  // namespace ticl

#endif
