#ifndef RecoHGCal_TICL_TracksterLinkingSuperClustering_H
#define RecoHGCal_TICL_TracksterLinkingSuperClustering_H
/*
TICL plugin for electron superclustering in HGCAL using a DNN. 
DNN designed and trained by Alessandro Tarabini.

Inputs are CLUE3D EM tracksters. Outputs are superclusters (as vectors of IDs of trackster)
"Seed trackster" : seed of supercluster, always highest pT trackster of supercluster, normally should be an electron
"Candidate trackster" : trackster that is considered for superclustering with a seed

Authors : Theo Cuisset <theo.cuisset@cern.ch>, Shamik Ghosh <shamik.ghosh@cern.ch>
Date : 11/2023

Modified by Felice Pantaleo <felice.pantaleo@cern.ch>
Improved memory usage and inference performance. 
Date: 02/2026

*/

#include <memory>
#include <vector>

namespace cms {
  namespace Ort {
    class ONNXRuntime;
  }
}  // namespace cms

#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "RecoHGCal/TICL/interface/SuperclusteringDNNInputs.h"
#include "DataFormats/HGCalReco/interface/TracksterFwd.h"

namespace ticl {
  class TracksterLinkingbySuperClusteringDNN : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbySuperClusteringDNN(const edm::ParameterSet& ps,
                                         edm::ConsumesCollector iC,
                                         cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);
    ~TracksterLinkingbySuperClusteringDNN() override = default;

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

    // --- Configuration
    std::unique_ptr<AbstractSuperclusteringDNNInput> dnnInputs_;
    unsigned int inferenceBatchSize_;
    double nnWorkingPoint_;
    float deltaEtaWindow_;
    float deltaPhiWindow_;
    float seedPtThreshold_;
    float candidateEnergyThreshold_;
    float explVarRatioCut_energyBoundary_;
    float explVarRatioMinimum_lowEnergy_;
    float explVarRatioMinimum_highEnergy_;
    bool filterByTracksterPID_;
    std::vector<int> tracksterPIDCategoriesToFilter_;
    float PIDThreshold_;
  };

}  // namespace ticl

#endif
