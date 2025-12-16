#ifndef RecoHGCal_TICL_GNNInterpretationAlgo_H_
#define RecoHGCal_TICL_GNNInterpretationAlgo_H_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "TMatrixDSym.h"
#include "TMatrixD.h"

namespace ticl {
  
  struct GraphEdge {
    unsigned target_index;  // Index of the neighbor (trackster)
    float weight;
  };
  
  struct GraphNode {
    unsigned index;                     // Index of the seed (track or trackster)
    bool isTrackster;                   // True if it's from seedingCollection
    std::vector<GraphEdge> neighbours;  // Edges to nearby tracksters
  };

  using NodeKey = std::pair<bool, int>;
  
  struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
      auto h1 = std::hash<T1>{}(p.first);
      auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
  };

  struct GraphData {
    std::vector<std::vector<float>> node_features;
    std::vector<std::pair<int, int>> edge_index;
    std::vector<std::vector<float>> edge_attr;
    std::unordered_map<NodeKey, size_t, pair_hash> nodeIndexToRow;
    int num_nodes;                                   
  };

  class GNNInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    GNNInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);

    ~GNNInterpretationAlgo() override;

    void makeCandidates(const Inputs &input,
                        edm::Handle<MtdHostCollection> inputTiming_h,
                        std::vector<Trackster> &resultTracksters,
                        std::vector<int> &resultCandidate) override;

    void initialize(const HGCalDDDConstants *hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    static void fillPSetDescription(edm::ParameterSetDescription &iDesc);

  private:
    void buildLayers();
    const std::unique_ptr<cms::Ort::ONNXRuntime> onnxLinkingRuntimeFirstDisk_;
    const cms::Ort::ONNXRuntime* onnxLinkingSessionFirstDisk_;
    const std::unique_ptr<cms::Ort::ONNXRuntime> onnxLinkingRuntimeInterfaceDisk_;
    const cms::Ort::ONNXRuntime* onnxLinkingSessionInterfaceDisk_;
    const std::vector<std::string> inputNames_;
    const std::vector<std::string> output_;

    Vector propagateTrackster(const Trackster &t,
                              const unsigned idx,
                              float zVal,
                              std::array<TICLLayerTile, 2> &tracksterTiles);

    std::pair<float, float> CalculateTrackstersError(const Trackster &trackster);
    std::vector<float> padFeatures(const std::vector<float>& core_feats,
				   size_t track_block_size,
				   size_t trackster_block_size,
				   bool isTrack);
    void constructNodeFromWindow(const edm::MultiSpan<Trackster> &tracksters,
				  const std::vector<std::tuple<Vector, unsigned, AlgebraicMatrix55>> &seeding,
				  const std::array<TICLLayerTile, 2> &tracksterTiles,
				  const std::vector<Vector> &tracksterPropPoints,
				  float delta,
				  unsigned trackstersSize,
				  std::vector<GraphNode> &graph);
    void printGraphSummary(const GraphData& graphData);
    void buildGraphFromNodes(const std::tuple<Vector, AlgebraicMatrix55, int>& TrackInfo,
			     const reco::Track &track,
			     const edm::MultiSpan<Trackster> &tracksters,
			     const std::vector<reco::CaloCluster> &clusters,
			     const std::vector<GraphNode>& nodeVec,
			     GraphData& outGraphData) ;

    const float del_tk_ts_;
    const float threshold_;
    
    const size_t track_block_size     = 10; // number of track features
    const size_t trackster_block_size = 13; // number of trackster features 

    const HGCalDDDConstants *hgcons_;

    std::unique_ptr<GeomDet> firstDisk_[2];
    std::unique_ptr<GeomDet> interfaceDisk_[2];

    hgcal::RecHitTools rhtools_;

    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;

  };

}  // namespace ticl

#endif
