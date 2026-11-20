#ifndef RecoMuon_L3TrackFinder_CompactForest_h
#define RecoMuon_L3TrackFinder_CompactForest_h

// Compact forest of boosted decision trees (XGBoost binary:logistic model), and working
// points in pT bins. Used by MuonIOTracksForestSelector.
//
// Binary file format (little endian):
//   int32 nNodes, int32 nTrees, float32 baseLogit,
//   int8    feat[nNodes]   (-1 = leaf),
//   float32 val[nNodes]    (split threshold, or leaf value),
//   int32   left[nNodes], right[nNodes],
//   int32   roots[nTrees]
// Traversal: at an internal node, go left if x[feat] < val, else go right. Add the value of
// the leaf to the margin. The score is sigmoid(baseLogit + sum of the leaf values).
// A NaN feature fails each "<" comparison and goes right. This is the direction of missing
// values in the model, so the model must not send missing values to the left.
//
// In memory, each tree is stored again depth-first as packed 8-byte nodes (value, right-child
// index and feature). The left child comes directly after its parent. Thus a traversal reads
// fewer cache lines than with four separate arrays (feat, val, left, right). The trees are
// still added in file order, so the scores are bit-identical to a traversal of the file layout.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

namespace muonhp {

  class CompactForest {
  public:
    // Reads and checks the forest for feature vectors of nFeatures entries. Each split feature must be in the
    // vector, each child must come after its parent (thus each traversal stops), and the file size must agree
    // with the header.
    static std::unique_ptr<CompactForest> load(const std::string& path, int nFeatures) {
      std::ifstream in(path, std::ios::binary | std::ios::ate);
      if (!in) {
        throw cms::Exception("CompactForest") << "cannot open compact forest binary " << path;
      }
      const std::streamoff fileSize = in.tellg();
      in.seekg(0);

      int32_t nNodes = 0, nTrees = 0;
      float baseLogit = 0.f;
      in.read(reinterpret_cast<char*>(&nNodes), 4);
      in.read(reinterpret_cast<char*>(&nTrees), 4);
      in.read(reinterpret_cast<char*>(&baseLogit), 4);
      if (!in || nNodes <= 0 || nTrees <= 0) {
        throw cms::Exception("CompactForest") << "invalid header in " << path;
      }
      const std::streamoff expected = 12 + std::streamoff(nNodes) * 13 + std::streamoff(nTrees) * 4;
      if (fileSize != expected) {
        throw cms::Exception("CompactForest") << path << " has " << fileSize << " bytes, the header (" << nNodes
                                              << " nodes, " << nTrees << " trees) implies " << expected;
      }

      auto forest = std::unique_ptr<CompactForest>(new CompactForest());
      forest->baseLogit_ = baseLogit;
      forest->feat_.resize(nNodes);
      forest->val_.resize(nNodes);
      forest->left_.resize(nNodes);
      forest->right_.resize(nNodes);
      forest->roots_.resize(nTrees);
      in.read(reinterpret_cast<char*>(forest->feat_.data()), nNodes);
      in.read(reinterpret_cast<char*>(forest->val_.data()), 4LL * nNodes);
      in.read(reinterpret_cast<char*>(forest->left_.data()), 4LL * nNodes);
      in.read(reinterpret_cast<char*>(forest->right_.data()), 4LL * nNodes);
      in.read(reinterpret_cast<char*>(forest->roots_.data()), 4LL * nTrees);
      if (!in) {
        throw cms::Exception("CompactForest") << "truncated compact forest binary " << path;
      }

      int maxFeature = -1;
      for (int32_t n = 0; n < nNodes; ++n) {
        const int f = forest->feat_[n];
        if (f < 0) {
          continue;
        }
        maxFeature = std::max(maxFeature, f);
        if (f >= nFeatures) {
          throw cms::Exception("CompactForest")
              << path << ": node " << n << " splits on feature " << f << " but the extractor provides " << nFeatures
              << " features (model/feature-set mismatch)";
        }
        if (forest->left_[n] <= n || forest->left_[n] >= nNodes || forest->right_[n] <= n ||
            forest->right_[n] >= nNodes) {
          throw cms::Exception("CompactForest") << path << ": node " << n << " has invalid children";
        }
      }
      for (const int32_t r : forest->roots_) {
        if (r < 0 || r >= nNodes) {
          throw cms::Exception("CompactForest") << path << ": invalid tree root " << r;
        }
      }
      if (nNodes >= (1 << 24) || nFeatures > static_cast<int>(kLeaf)) {
        throw cms::Exception("CompactForest") << path << ": model too large for the packed node layout";
      }
      forest->maxFeature_ = maxFeature;
      forest->relayout();
      return forest;
    }

    // Probability of the positive class (genuine muon track) for n feature vectors, row-major and `stride`
    // floats apart. The loop goes tree by tree over blocks of tracks, so the nodes of a tree stay in cache for
    // all tracks of a block. Each track adds its trees in file order, so the scores are bit-identical to
    // evaluate(x).
    void evaluate(const float* x, size_t n, size_t stride, float* scores) const {
      constexpr size_t kBlock = 16;
      const PackedNode* nodes = nodes_.data();
      float margin[kBlock];
      for (size_t first = 0; first < n; first += kBlock) {
        const size_t m = std::min(kBlock, n - first);
        std::fill(margin, margin + m, baseLogit_);
        for (const uint32_t root : packedRoots_) {
          for (size_t j = 0; j < m; ++j) {
            const float* xj = x + (first + j) * stride;
            uint32_t i = root;
            uint32_t meta = nodes[i].meta;
            while ((meta & 0xffu) != kLeaf) {
              i = (xj[meta & 0xffu] < nodes[i].val) ? i + 1 : (meta >> 8);
              meta = nodes[i].meta;
            }
            margin[j] += nodes[i].val;
          }
        }
        for (size_t j = 0; j < m; ++j) {
          scores[first + j] = std::clamp(1.0f / (1.0f + std::exp(-margin[j])), 0.0f, 1.0f);
        }
      }
    }

    // Probability of the positive class (genuine muon track) for one feature vector.
    float evaluate(const float* x) const {
      const PackedNode* nodes = nodes_.data();
      float margin = baseLogit_;
      for (const uint32_t root : packedRoots_) {
        uint32_t i = root;
        uint32_t meta = nodes[i].meta;
        while ((meta & 0xffu) != kLeaf) {
          i = (x[meta & 0xffu] < nodes[i].val) ? i + 1 : (meta >> 8);
          meta = nodes[i].meta;
        }
        margin += nodes[i].val;
      }
      return std::clamp(1.0f / (1.0f + std::exp(-margin)), 0.0f, 1.0f);
    }

    int nNodes() const { return static_cast<int>(nodes_.size()); }
    int nTrees() const { return static_cast<int>(packedRoots_.size()); }
    int maxFeature() const { return maxFeature_; }
    float baseLogit() const { return baseLogit_; }

  private:
    // val: split threshold or leaf value. meta: right-child index << 8 | feature index (kLeaf for a leaf).
    // The left child is the next node.
    struct PackedNode {
      float val;
      uint32_t meta;
    };
    static constexpr uint32_t kLeaf = 0xffu;

    CompactForest() = default;

    // Writes the subtree of file node n into nodes_ in depth-first (pre-order) sequence. Returns its index.
    uint32_t emit(int32_t n) {
      const uint32_t i = nodes_.size();
      nodes_.push_back({val_[n], kLeaf});
      if (feat_[n] >= 0) {
        emit(left_[n]);  // the left child gets index i + 1
        const uint32_t right = emit(right_[n]);
        nodes_[i].meta = (right << 8) | static_cast<uint32_t>(feat_[n]);
      }
      return i;
    }
    void relayout() {
      nodes_.reserve(feat_.size());
      for (const int32_t r : roots_) {
        packedRoots_.push_back(emit(r));
      }
      feat_ = {};
      val_ = {};
      left_ = {};
      right_ = {};
      roots_ = {};
    }

    float baseLogit_ = 0.f;
    int maxFeature_ = -1;
    std::vector<PackedNode> nodes_;
    std::vector<uint32_t> packedRoots_;
    // File layout, released after relayout()
    std::vector<int8_t> feat_;
    std::vector<float> val_;
    std::vector<int32_t> left_;
    std::vector<int32_t> right_;
    std::vector<int32_t> roots_;
  };

  // Score thresholds in pT bins: thresholds[i] applies to pT in [edges[i], edges[i+1]). The last bin has no
  // upper edge. If edges is empty, one threshold applies to all tracks.
  class BinnedWorkingPoints {
  public:
    BinnedWorkingPoints(double threshold, std::vector<double> edges, std::vector<double> thresholds)
        : threshold_(threshold), edges_(std::move(edges)), thresholds_(std::move(thresholds)) {
      if (edges_.empty()) {
        if (!thresholds_.empty()) {
          throw cms::Exception("Configuration")
              << "decisionThresholds set without ptBinEdges; set both for pT-binned working points or neither";
        }
        return;
      }
      if (thresholds_.size() != edges_.size()) {
        throw cms::Exception("Configuration")
            << thresholds_.size() << " decisionThresholds for " << edges_.size()
            << " ptBinEdges; thresholds[i] applies to pT in [edges[i], edges[i+1]) (last bin open-ended)";
      }
      for (size_t i = 1; i < edges_.size(); ++i) {
        if (edges_[i] <= edges_[i - 1]) {
          throw cms::Exception("Configuration") << "ptBinEdges must be strictly increasing";
        }
      }
    }

    float threshold(double pt) const {
      if (edges_.empty()) {
        return static_cast<float>(threshold_);
      }
      size_t b = 0;
      while (b + 1 < edges_.size() && pt >= edges_[b + 1]) {
        ++b;
      }
      return static_cast<float>(thresholds_[b]);
    }

  private:
    double threshold_;
    std::vector<double> edges_;
    std::vector<double> thresholds_;
  };

}  // namespace muonhp

#endif
