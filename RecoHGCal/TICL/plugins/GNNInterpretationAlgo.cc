// Author: Mohamed Darwish
#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "RecoHGCal/TICL/plugins/GNNInterpretationAlgo.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

using namespace ticl;
using namespace cms::Ort;

using Vector = ticl::Trackster::Vector;

GNNInterpretationAlgo::~GNNInterpretationAlgo() {}

GNNInterpretationAlgo::GNNInterpretationAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector cc)
    : TICLInterpretationAlgoBase(conf, cc),
      onnxLinkingRuntimeFirstDisk_(std::make_unique<cms::Ort::ONNXRuntime>(
          conf.getParameter<edm::FileInPath>("onnxTrkLinkingModelFirstDisk").fullPath().c_str())),
      onnxLinkingRuntimeInterfaceDisk_(std::make_unique<cms::Ort::ONNXRuntime>(
          conf.getParameter<edm::FileInPath>("onnxTrkLinkingModelInterfaceDisk").fullPath().c_str())),
      inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
      output_(conf.getParameter<std::vector<std::string>>("output")),
      del_tk_ts_(conf.getParameter<double>("delta_tk_ts")),
      threshold_(conf.getParameter<double>("thr_gnn")) {
  onnxLinkingSessionFirstDisk_ = onnxLinkingRuntimeFirstDisk_.get();
  onnxLinkingSessionInterfaceDisk_ = onnxLinkingRuntimeInterfaceDisk_.get();
}

// Initialization
void GNNInterpretationAlgo::initialize(const HGCalDDDConstants* hgcons,
                                       const hgcal::RecHitTools rhtools,
                                       const edm::ESHandle<MagneticField> bfieldH,
                                       const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
  bfield_ = bfieldH;
  propagator_ = propH;

  buildLayers();
}
// Geometry construction
void GNNInterpretationAlgo::buildLayers() {
  // Build propagation disks at:
  //  - HGCal front face
  //  - CE-E CE-H interface

  const float z_front = hgcons_->waferZ(1, true);
  const auto r_front = hgcons_->rangeR(z_front, true);

  const float z_interface = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();
  const auto r_interface = hgcons_->rangeR(z_interface, true);

  for (int side = 0; side < 2; ++side) {
    const float sign = (side == 0 ? -1.f : 1.f);

    firstDisk_[side] = std::make_unique<GeomDet>(
        Disk::build(Disk::PositionType(0, 0, sign * z_front),
                    Disk::RotationType(),
                    SimpleDiskBounds(r_front.first, r_front.second, sign * z_front - 0.5f, sign * z_front + 0.5f))
            .get());

    interfaceDisk_[side] = std::make_unique<GeomDet>(
        Disk::build(Disk::PositionType(0, 0, sign * z_interface),
                    Disk::RotationType(),
                    SimpleDiskBounds(
                        r_interface.first, r_interface.second, sign * z_interface - 0.5f, sign * z_interface + 0.5f))
            .get());
  }
}

// Trackster propagation
Vector GNNInterpretationAlgo::propagateTrackster(const Trackster& t,
                                                 unsigned idx,
                                                 float zVal,
                                                 std::array<TICLLayerTile, 2>& tracksterTiles) {
  // needs only the positive Z co-ordinate of the surface to propagate to
  // the correct sign is calculated inside according to the barycenter of trackster

  // Propagation direction
  const Vector& barycenter = t.barycenter();

  // NOTE:
  // barycenter as direction for tracksters w/ poor PCA
  // propagation still done to get the cartesian coords
  // which are anyway converted to eta, phi in linking
  // -> can be simplified later

  //FP: disable PCA propagation for the moment and fallback to barycenter position
  // if (t.eigenvalues()[0] / t.eigenvalues()[1] < 20)

  Vector direction = barycenter.unit();

  // Ensure correct Z-side propagation
  zVal *= (barycenter.Z() > 0.f ? 1.f : -1.f);

  const float scale = (zVal - barycenter.Z()) / direction.Z();
  const Vector propPoint(scale * direction.X() + barycenter.X(), scale * direction.Y() + barycenter.Y(), zVal);

  // Fill spatial tiles for fast lookup
  const bool isPositiveZ = (propPoint.Eta() > 0.f);
  tracksterTiles[isPositiveZ].fill(propPoint.Eta(), propPoint.Phi(), idx);

  return propPoint;
}

std::pair<float, float> GNNInterpretationAlgo::CalculateTrackstersError(const Trackster& trackster) {
  const auto& barycenter = trackster.barycenter();
  const double x = barycenter.x(), y = barycenter.y(), z = barycenter.z();

  const auto& s = trackster.sigmasPCA();
  const double s1 = s[0] * s[0], s2 = s[1] * s[1], s3 = s[2] * s[2];

  const auto& v1 = trackster.eigenvectors()[0];
  const auto& v2 = trackster.eigenvectors()[1];
  const auto& v3 = trackster.eigenvectors()[2];

  // Covariance in XY from 3D
  const double cxx = s1 * v1.x() * v1.x() + s2 * v2.x() * v2.x() + s3 * v3.x() * v3.x();
  const double cxy = s1 * v1.x() * v1.y() + s2 * v2.x() * v2.y() + s3 * v3.x() * v3.y();
  const double cyy = s1 * v1.y() * v1.y() + s2 * v2.y() * v2.y() + s3 * v3.y() * v3.y();

  // Geometry helpers
  const double r2 = x * x + y * y;
  const double denom_eta = r2 * (r2 + z * z);
  const double sqrt_term = std::sqrt(r2 / (z * z) + 1);

  // Jacobian elements
  const double J00 = -(x * z * z * sqrt_term) / denom_eta;
  const double J01 = -(y * z * z * sqrt_term) / denom_eta;
  const double J10 = -y / r2;
  const double J11 = x / r2;

  // CovEtaPhi = J * CovXY * J^T
  const double cee = J00 * (J00 * cxx + J01 * cxy) + J01 * (J00 * cxy + J01 * cyy);
  const double cpp = J10 * (J10 * cxx + J11 * cxy) + J11 * (J10 * cxy + J11 * cyy);

  return {std::sqrt(std::abs(cee)), std::sqrt(std::abs(cpp))};
}

void GNNInterpretationAlgo::constructNodeFromWindow(
    const edm::MultiSpan<Trackster>& tracksters,
    const std::vector<std::tuple<Vector, unsigned, AlgebraicMatrix55>>& seeding,
    const std::array<TICLLayerTile, 2>& tracksterTiles,
    const std::vector<Vector>& tracksterPropPoints,
    float delta2,
    unsigned trackstersSize,
    std::vector<GraphNode>& graph) {
  const float delta = 0.5f * delta2;

  for (const auto& [seedPos, seedIdx, _] : seeding) {
    const float seedEta = seedPos.Eta();
    const float seedPhi = seedPos.Phi();
    const bool isPositiveZ = (seedEta > 0.0f);

    const TICLLayerTile& tile = tracksterTiles[isPositiveZ];

    const float etaMin = std::max(std::abs(seedEta) - delta, static_cast<float>(TileConstants::minEta));
    const float etaMax = std::min(std::abs(seedEta) + delta, static_cast<float>(TileConstants::maxEta));

    const auto searchBox = tile.searchBoxEtaPhi(etaMin, etaMax, seedPhi - delta, seedPhi + delta);

    GraphNode node;
    node.index = seedIdx;
    node.isTrackster = false;  // this node represents a track

    for (int iEta = searchBox[0]; iEta <= searchBox[1]; ++iEta) {
      for (int iPhi = searchBox[2]; iPhi <= searchBox[3]; ++iPhi) {
        const int globalBin = tile.globalBin(iEta, iPhi % TileConstants::nPhiBins);

        const auto& candidates = tile[globalBin];

        for (const unsigned tsIdx : candidates) {
          if (tsIdx >= trackstersSize)
            continue;

          const float dEta = tracksterPropPoints[tsIdx].Eta() - seedEta;
          const float dPhi = tracksterPropPoints[tsIdx].Phi() - seedPhi;

          const float deltaR2 = dEta * dEta + dPhi * dPhi;
          if (deltaR2 < delta2) {
            GraphEdge edge;
            edge.target_index = tsIdx;
            node.neighbours.push_back(edge);
          }
        }
      }
    }
    graph.emplace_back(std::move(node));
  }
}

std::vector<float> GNNInterpretationAlgo::padFeatures(const std::vector<float>& core_feats,
                                                      size_t track_block_size,
                                                      size_t trackster_block_size,
                                                      bool isTrack) {
  std::vector<float> out;
  out.reserve(track_block_size + trackster_block_size);

  if (isTrack) {
    out.insert(out.end(), core_feats.begin(), core_feats.end());
    out.insert(out.end(), trackster_block_size, 0.f);
  } else {
    out.insert(out.end(), track_block_size, 0.f);
    out.insert(out.end(), core_feats.begin(), core_feats.end());
  }

  return out;
}
void GNNInterpretationAlgo::buildGraphFromNodes(const std::tuple<Vector, AlgebraicMatrix55, int>& TrackInfo,
                                                const reco::Track& track,
                                                const edm::MultiSpan<Trackster>& tracksters,
                                                const std::vector<reco::CaloCluster>& clusters,
                                                const std::vector<GraphNode>& nodeVec,
                                                GraphData& outGraphData) {
  outGraphData = {};  // clear previous data

  std::unordered_map<int, std::vector<float>> track_node_features;
  std::unordered_map<int, std::vector<float>> trackster_node_features;

  const auto& [pos, localErrMatrix, track_idx] = TrackInfo;

  // Track coordinates and covariances
  const float eta = pos.Eta(), phi = pos.Phi();
  const float x = pos.X(), y = pos.Y(), z = pos.Z();

  AlgebraicMatrix22 covMatrixXY;
  covMatrixXY(0, 0) = localErrMatrix(3, 3);
  covMatrixXY(0, 1) = localErrMatrix(3, 4);
  covMatrixXY(1, 0) = localErrMatrix(3, 4);
  covMatrixXY(1, 1) = localErrMatrix(4, 4);

  const double sqrt_term = std::sqrt((x * x + y * y) / (z * z) + 1);
  const double denom_eta = (x * x + y * y) * (x * x + y * y + z * z);
  const double denom_phi = x * x + y * y;

  AlgebraicMatrix22 jacobian;
  jacobian(0, 0) = -(x * z * z * sqrt_term) / denom_eta;
  jacobian(0, 1) = -(y * z * z * sqrt_term) / denom_eta;
  jacobian(1, 0) = -y / denom_phi;
  jacobian(1, 1) = x / denom_phi;

  AlgebraicMatrix22 covMatrixEtaPhi = ROOT::Math::Transpose(jacobian) * covMatrixXY * jacobian;
  const float track_etaErr = std::sqrt(covMatrixEtaPhi(0, 0));
  const float track_phiErr = std::sqrt(covMatrixEtaPhi(1, 1));

  const float track_p = track.p();
  const float track_pt = track.pt();
  const float trackHits = track.recHitsSize();

  std::vector<float> trk_feats = {
      std::abs(eta), phi, track_etaErr, track_phiErr, x, y, std::abs(z), track_p, track_pt, trackHits};
  trk_feats.resize(track_block_size);

  // Lambda to wrap deltaPhi
  auto wrapPhi = [](float dphi) -> float {
    const float pi = M_PI, two_pi = 2.0f * M_PI;
    dphi = std::fmod(dphi + pi, two_pi);
    if (dphi < 0)
      dphi += two_pi;
    return dphi - pi;
  };

  // Fill node features
  for (const auto& node : nodeVec) {
    if (!node.isTrackster && static_cast<int>(node.index) == track_idx) {
      track_node_features[node.index] = padFeatures(trk_feats, track_block_size, trackster_block_size, true);

      for (const auto& edge : node.neighbours) {
        const unsigned ts_idx = edge.target_index;
        if (ts_idx >= tracksters.size())
          continue;

        if (!trackster_node_features.count(ts_idx)) {
          const auto& ts = tracksters[ts_idx];
          auto [errEta, errPhi] = CalculateTrackstersError(ts);

          std::vector<float> ts_feats = {std::abs(ts.barycenter().eta()),
                                         ts.barycenter().phi(),
                                         errEta,
                                         errPhi,
                                         ts.barycenter().x(),
                                         ts.barycenter().y(),
                                         std::abs(ts.barycenter().z()),
                                         ts.raw_energy(),
                                         ts.time(),
                                         ts.timeError(),
                                         ts.raw_em_energy(),
                                         ts.raw_em_pt(),
                                         ts.raw_pt()};
          ts_feats.resize(trackster_block_size);
          trackster_node_features[ts_idx] = padFeatures(ts_feats, track_block_size, trackster_block_size, false);
        }
        outGraphData.edge_index.emplace_back(node.index, ts_idx);
      }
    }
  }

  // Insert nodes
  size_t row_idx = 0;
  for (const auto& [idx, feats] : track_node_features) {
    outGraphData.nodeIndexToRow[{false, idx}] = row_idx++;
    outGraphData.node_features.push_back(feats);
  }
  for (const auto& [idx, feats] : trackster_node_features) {
    outGraphData.nodeIndexToRow[{true, idx}] = row_idx++;
    outGraphData.node_features.push_back(feats);
  }
  outGraphData.num_nodes = outGraphData.node_features.size();

  // Fill edge attributes
  for (const auto& edge : outGraphData.edge_index) {
    const NodeKey src_key{false, edge.first};
    const NodeKey dst_key{true, edge.second};

    if (!outGraphData.nodeIndexToRow.count(src_key) || !outGraphData.nodeIndexToRow.count(dst_key))
      continue;

    const auto& src_feats = outGraphData.node_features[outGraphData.nodeIndexToRow[src_key]];
    const auto& dst_feats = outGraphData.node_features[outGraphData.nodeIndexToRow[dst_key]];

    const int trkster_offset = track_block_size;

    const float trk_eta = src_feats[0], trk_phi = src_feats[1];
    const float ts_eta = dst_feats[trkster_offset], ts_phi = dst_feats[trkster_offset + 1];

    const float delta_eta = trk_eta - ts_eta;
    const float delta_phi = wrapPhi(trk_phi - ts_phi);
    const float deta_sig = delta_eta / std::sqrt(dst_feats[trkster_offset + 2] * dst_feats[trkster_offset + 2] +
                                                 src_feats[2] * src_feats[2] + 1e-8f);
    const float dphi_sig = delta_phi / std::sqrt(dst_feats[trkster_offset + 3] * dst_feats[trkster_offset + 3] +
                                                 src_feats[3] * src_feats[3] + 1e-8f);
    const float deltaR = std::sqrt(delta_eta * delta_eta + delta_phi * delta_phi);

    const float dx = dst_feats[trkster_offset + 4] - src_feats[4];
    const float dy = dst_feats[trkster_offset + 5] - src_feats[5];
    const float dz = dst_feats[trkster_offset + 6] - src_feats[6];
    const float dist3D = std::sqrt(dx * dx + dy * dy + dz * dz);
    const float distXY = std::sqrt(dx * dx + dy * dy);

    const float dE = dst_feats[trkster_offset + 7] - src_feats[7];
    const float E_ratio = dst_feats[trkster_offset + 7] / (src_feats[7] + 1e-8f);

    const auto& ts = tracksters[edge.second];
    const auto& vertices = ts.vertices();
    float min_dist = std::numeric_limits<float>::max();
    float max_dist = 0.f;

    for (const auto& vtx : vertices) {
      const auto& cl = clusters[vtx];
      const float dist = std::sqrt(std::pow(cl.x() - src_feats[4], 2) + std::pow(cl.y() - src_feats[5], 2) +
                                   std::pow(std::abs(cl.z()) - src_feats[6], 2));
      min_dist = std::min(min_dist, dist);
      max_dist = std::max(max_dist, dist);
    }

    outGraphData.edge_attr.push_back(
        {delta_eta, delta_phi, deta_sig, dphi_sig, deltaR, dist3D, distXY, dE, E_ratio, min_dist, max_dist});
  }
}

void GNNInterpretationAlgo::makeCandidates(const Inputs& input,
                                           edm::Handle<MtdHostCollection> inputTiming_h,
                                           std::vector<Trackster>& resultTracksters,
                                           std::vector<int>& resultCandidate) {
  const auto& tracks = *input.tracksHandle;
  const auto& maskTracks = input.maskedTracks;
  const auto& tracksters = input.tracksters;
  const auto& clusters = input.layerClusters;

  const auto bFieldProd = bfield_.product();
  const Propagator& prop = (*propagator_);

  // propagated point collections
  // elements in the propagated points collecions are used
  // to look for potential linkages in the appropriate tiles
  // Track propagation
  using TrackPropInfo = std::tuple<Vector, unsigned, AlgebraicMatrix55>;

  std::vector<TrackPropInfo> tkPropFront;  // propagated to first disk
  std::vector<TrackPropInfo> tkPropInt;    // propagated to interface disk
  tkPropFront.reserve(tracks.size());
  tkPropInt.reserve(tracks.size());

  std::vector<unsigned> candidateTrackIds;
  candidateTrackIds.reserve(tracks.size());

  for (unsigned i = 0; i < tracks.size(); ++i) {
    if (maskTracks[i])
      candidateTrackIds.push_back(i);
  }
  std::sort(candidateTrackIds.begin(), candidateTrackIds.end(), [&](unsigned i, unsigned j) {
    return tracks[i].p() > tracks[j].p();
  });

  for (unsigned trkId : candidateTrackIds) {
    const auto& tk = tracks[trkId];
    const int side = (tk.eta() > 0);

    const auto& fts = trajectoryStateTransform::outerFreeState(tk, bFieldProd);
    // to the HGCal front
    const auto tsosFront = prop.propagate(fts, firstDisk_[side]->surface());
    if (tsosFront.isValid()) {
      tkPropFront.emplace_back(
          Vector(tsosFront.globalPosition().x(), tsosFront.globalPosition().y(), tsosFront.globalPosition().z()),
          trkId,
          tsosFront.localError().matrix());
    }

    // Interface disk
    const auto tsosInt = prop.propagate(fts, interfaceDisk_[side]->surface());
    if (tsosInt.isValid()) {
      tkPropInt.emplace_back(
          Vector(tsosInt.globalPosition().x(), tsosInt.globalPosition().y(), tsosInt.globalPosition().z()),
          trkId,
          tsosInt.localError().matrix());
    }
  }  // Tracks
  tkPropFront.shrink_to_fit();
  tkPropInt.shrink_to_fit();
  candidateTrackIds.shrink_to_fit();
  // Propagate tracksters
  // Record postions of all tracksters propagated to layer 1 and lastLayerEE,
  // to be used later for distance calculation in the link finding stage
  // indexed by trackster index in event collection
  std::array<TICLLayerTile, 2> tsTilesFront = {};
  std::array<TICLLayerTile, 2> tsTilesInt = {};

  std::vector<Vector> tsPropFront, tsPropInt;
  tsPropFront.reserve(tracksters.size());
  tsPropInt.reserve(tracksters.size());

  for (unsigned i = 0; i < tracksters.size(); ++i) {
    const auto& ts = tracksters[i];

    float zFront = hgcons_->waferZ(1, true);
    tsPropFront.emplace_back(propagateTrackster(ts, i, zFront, tsTilesFront));

    float zInt = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();
    tsPropInt.emplace_back(propagateTrackster(ts, i, zInt, tsTilesInt));
  }

  // Step 1: Construct nodes from tracksters and tracks
  std::vector<GraphNode> nodesFront, nodesInt;

  constructNodeFromWindow(
      tracksters, tkPropFront, tsTilesFront, tsPropFront, del_tk_ts_, tracksters.size(), nodesFront);

  constructNodeFromWindow(tracksters, tkPropInt, tsTilesInt, tsPropInt, del_tk_ts_, tracksters.size(), nodesInt);

  std::vector<std::vector<unsigned>> trackToTracksters(tracks.size());
  std::vector<std::vector<std::pair<unsigned, float>>> trackToScores(tracks.size());
  std::vector<bool> tracksterAvailable(tracksters.size(), true);

  auto runInferenceForTrack = [&](unsigned trkId,
                                  const std::vector<TrackPropInfo>& tkProps,
                                  const std::vector<GraphNode>& nodes,
                                  bool useInterfaceModel) {
    auto it =
        std::find_if(tkProps.begin(), tkProps.end(), [&](const auto& info) { return std::get<1>(info) == trkId; });

    if (it == tkProps.end())
      return;

    GraphData graphData;
    buildGraphFromNodes(std::make_tuple(std::get<0>(*it), std::get<2>(*it), trkId),
                        tracks[trkId],
                        tracksters,
                        clusters,
                        nodes,
                        graphData);

    // Prepare ONNX input
    std::vector<std::vector<float>> inputData(3);
    std::vector<std::vector<int64_t>> inputShapes;

    const int64_t nNodes = graphData.node_features.size();
    const int64_t nNodeFeat = nNodes ? graphData.node_features.front().size() : 0;

    for (const auto& feat : graphData.node_features)
      inputData[0].insert(inputData[0].end(), feat.begin(), feat.end());

    inputShapes.push_back({nNodes, nNodeFeat});

    std::vector<float> src_nodes, dst_nodes;
    for (const auto& edge : graphData.edge_index) {
      NodeKey src_key = {false, edge.first};
      NodeKey dst_key = {true, edge.second};
      src_nodes.push_back(graphData.nodeIndexToRow.at(src_key));
      dst_nodes.push_back(graphData.nodeIndexToRow.at(dst_key));
    }
    inputData[1].insert(inputData[1].end(), src_nodes.begin(), src_nodes.end());
    inputData[1].insert(inputData[1].end(), dst_nodes.begin(), dst_nodes.end());
    inputShapes.push_back({2, static_cast<int64_t>(graphData.edge_index.size())});

    const int64_t nEdges = graphData.edge_attr.size();
    const int64_t nEdgeFeat = nEdges ? graphData.edge_attr.front().size() : 0;

    for (const auto& attr : graphData.edge_attr)
      inputData[2].insert(inputData[2].end(), attr.begin(), attr.end());

    inputShapes.push_back({nEdges, nEdgeFeat});

    if (inputData[1].empty())
      return;
    const auto& outputs = useInterfaceModel
                              ? onnxLinkingSessionInterfaceDisk_->run(inputNames_, inputData, inputShapes, output_)
                              : onnxLinkingSessionFirstDisk_->run(inputNames_, inputData, inputShapes, output_);

    const auto& scores = outputs[0];

    for (size_t i = 0; i < graphData.edge_index.size(); ++i) {
      if (scores[i] <= threshold_)
        continue;
      const auto& edge = graphData.edge_index[i];
      const float deltaR = graphData.edge_attr[i][4];
      const float score = std::log(tracks[trkId].pt() / (deltaR + 1e-5f));

      trackToScores[trkId].emplace_back(edge.second, score);
    }
  };

  for (unsigned trkId : candidateTrackIds) {
    runInferenceForTrack(trkId, tkPropFront, nodesFront, false);  //First disk
    runInferenceForTrack(trkId, tkPropInt, nodesInt, true);       //Interface disk
  }
  // Resolve global associations
  std::vector<std::tuple<unsigned, unsigned, float>> allLinks;

  for (unsigned trkId = 0; trkId < trackToScores.size(); ++trkId) {
    for (const auto& [tsId, score] : trackToScores[trkId])
      allLinks.emplace_back(trkId, tsId, score);
  }

  std::sort(
      allLinks.begin(), allLinks.end(), [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
  for (const auto& [trkId, tsId, score] : allLinks) {
    if (tracksterAvailable[tsId]) {
      trackToTracksters[trkId].push_back(tsId);
      tracksterAvailable[tsId] = false;
    }
  }
  // Build output tracksters

  for (unsigned trkId = 0; trkId < trackToTracksters.size(); ++trkId) {
    if (trackToTracksters[trkId].empty())
      continue;

    resultCandidate[trkId] = resultTracksters.size();

    if (trackToTracksters[trkId].size() == 1) {
      resultTracksters.push_back(tracksters[trackToTracksters[trkId][0]]);
    } else {
      Trackster merged;
      merged.mergeTracksters(tracksters, trackToTracksters[trkId]);

      bool hasHadron = false;
      for (auto tsId : trackToTracksters[trkId])
        hasHadron |= tracksters[tsId].isHadronic();
      merged.setIdProbability(hasHadron ? Trackster::ParticleType::charged_hadron : Trackster::ParticleType::electron,
                              1.f);

      resultTracksters.push_back(std::move(merged));
    }
  }

  // Add unlinked tracksters
  for (unsigned i = 0; i < tracksters.size(); ++i) {
    if (tracksterAvailable[i])
      resultTracksters.push_back(tracksters[i]);
  }
}

void GNNInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  desc.add<edm::FileInPath>(
          "onnxTrkLinkingModelFirstDisk",
          edm::FileInPath("RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/FirstDiskPropGNN_v0.onnx"))
      ->setComment("Path to ONNX tracks tracksters linking model at first disk ");
  desc.add<edm::FileInPath>(
          "onnxTrkLinkingModelInterfaceDisk",
          edm::FileInPath("RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/InterfaceDiskPropGNN_v0.onnx"))
      ->setComment("Path to ONNX tracks tracksters linking model at interface disk ");

  desc.add<std::vector<std::string>>("inputNames", {"x", "edge_index", "edge_attr"});
  desc.add<std::vector<std::string>>("output", {"output"});
  desc.add<double>("delta_tk_ts", 0.1);
  desc.add<double>("thr_gnn", 0.5);

  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
