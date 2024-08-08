/*
TICL plugin for electron superclustering in HGCAL using a DNN. 
DNN designed and trained by Alessandro Tarabini.

Inputs are CLUE3D EM tracksters. Outputs are superclusters (as vectors of IDs of trackster)
"Seed trackster" : seed of supercluster, always highest pT trackster of supercluster, normally should be an electron
"Candidate trackster" : trackster that is considered for superclustering with a seed

Authors : Theo Cuisset <theo.cuisset@cern.ch>, Shamik Ghosh <shamik.ghosh@cern.ch>
Date : 11/2023
*/

#ifndef RecoHGCal_TICL_TracksterLinkingSuperClustering_H
#define RecoHGCal_TICL_TracksterLinkingSuperClustering_H

#include <vector>

namespace cms {
  namespace Ort {
    class ONNXRuntime;
  }
}  // namespace cms

#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "RecoHGCal/TICL/interface/SuperclusteringDNNInputs.h"

namespace ticl {
  class Trackster;

  class TracksterLinkingbySuperClusteringDNN : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbySuperClusteringDNN(const edm::ParameterSet& ps,
                                         edm::ConsumesCollector iC,
                                         cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);
    /* virtual */ ~TracksterLinkingbySuperClusteringDNN() override {}
    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;
    void initialize(const HGCalDDDConstants* hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

  private:
    bool checkExplainedVarianceRatioCut(ticl::Trackster const& ts) const;
    bool trackstersPassesPIDCut(const Trackster& ts) const;

    std::unique_ptr<AbstractSuperclusteringDNNInput> dnnInputs_;  // Helper class for DNN input features computation
    unsigned int inferenceBatchSize_;                             // Size of inference batches fed to DNN
    double
        nnWorkingPoint_;  // Working point for neural network (above this score, consider the trackster candidate for superclustering)
    float deltaEtaWindow_;                  // Delta eta window to consider trackster seed-candidate pairs for inference
    float deltaPhiWindow_;                  // Delta phi window
    float seedPtThreshold_;                 // Min pT for a trackster to be considered as supercluster seed
    float candidateEnergyThreshold_;        // Min energy for a trackster to be superclustered as candidate
    float explVarRatioCut_energyBoundary_;  // Boundary energy between low and high energy explVarRatio cut threshold
    float explVarRatioMinimum_lowEnergy_;  // Cut on explained variance ratio of tracksters to be considered as candidate, for trackster raw_energy < explVarRatioCut_energyBoundary
    float explVarRatioMinimum_highEnergy_;  // Cut on explained variance ratio of tracksters to be considered as candidate, for trackster raw_energy > explVarRatioCut_energyBoundary
    bool filterByTracksterPID_;
    std::vector<int> tracksterPIDCategoriesToFilter_;
    float PIDThreshold_;
  };

}  // namespace ticl

#endif
