#ifndef __L1Trigger_VertexFinder_VertexFinder_h__
#define __L1Trigger_VertexFinder_VertexFinder_h__

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/VertexFinder/interface/AlgoSettings.h"
#include "L1Trigger/VertexFinder/interface/RecoVertex.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace l1tVertexFinder {

  typedef std::vector<L1Track> FitTrackCollection;
  typedef std::vector<RecoVertex<>> RecoVertexCollection;

  class VertexFinder {
  public:
    /// Constructor and destructor
    VertexFinder(FitTrackCollection& fitTracks, const AlgoSettings& settings) {
      fitTracks_ = fitTracks;
      settings_ = &settings;
    }
    ~VertexFinder() {}

    /// Helper structs/classes
    struct SortTracksByZ0 {
      inline bool operator()(const L1Track track0, const L1Track track1) { return (track0.z0() < track1.z0()); }
    };

    struct SortTracksByPt {
      inline bool operator()(const L1Track track0, const L1Track track1) {
        return (std::abs(track0.pt()) > std::abs(track1.pt()));
      }
    };

    /// Accessors

    /// Number of iterations
    unsigned int IterationsPerTrack() const { return double(iterations_) / double(fitTracks_.size()); }
    /// Storage for tracks out of the L1 Track finder
    unsigned int numInputTracks() const { return fitTracks_.size(); }
    /// Number of iterations
    unsigned int NumIterations() const { return iterations_; }
    /// Number of reconstructed vertices
    unsigned int numVertices() const { return vertices_.size(); }
    /// Reconstructed Primary Vertex
    RecoVertex<> primaryVertex() const {
      if (pv_index_ < vertices_.size())
        return vertices_[pv_index_];
      else {
        edm::LogWarning("VertexFinder") << "PrimaryVertex::No Primary Vertex has been found.";
        return RecoVertex<>();
      }
    }
    /// Reconstructed Primary Vertex Id
    unsigned int primaryVertexId() const { return pv_index_; }
    /// Returns the z positions of the reconstructed primary vertices
    const std::vector<RecoVertex<>>& vertices() const { return vertices_; }
    /// Storage for tracks out of the L1 Track finder
    const FitTrackCollection& fitTracks() const { return fitTracks_; }

    /// Find the primary vertex
    void findPrimaryVertex();
    /// Associate the primary vertex with the real one
    void associatePrimaryVertex(double trueZ0);
    /// Gap Clustering Algorithm
    void GapClustering();
    /// Find maximum distance in two clusters of tracks
    float maxDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Find minimum distance in two clusters of tracks
    float minDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Find average distance in two clusters of tracks
    float meanDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Find distance between centres of two clusters
    float centralDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Simple Merge Algorithm
    void agglomerativeHierarchicalClustering();
    /// Adaptive Vertex Reconstruction algorithm
    void adaptiveVertexReconstruction();
    /// High pT Vertex Algorithm
    void fastHistoLooseAssociation();
    /// Histogramming algorithm
    void fastHisto(const TrackerTopology* tTopo);
    /// Sort vertices in pT
    void SortVerticesInPt() {
      std::sort(vertices_.begin(), vertices_.end(), [](const RecoVertex<>& vertex0, const RecoVertex<>& vertex1) {
        return (vertex0.pt() > vertex1.pt());
      });
    }
    /// Sort vertices in z
    void SortVerticesInZ0() {
      std::sort(vertices_.begin(), vertices_.end(), [](const RecoVertex<>& vertex0, const RecoVertex<>& vertex1) {
        return (vertex0.z0() < vertex1.z0());
      });
    }
    /// Number of iterations
    unsigned int numIterations() const { return iterations_; }
    /// Number of iterations
    unsigned int iterationsPerTrack() const { return double(iterations_) / double(fitTracks_.size()); }

    template <typename ForwardIterator, typename T>
    void strided_iota(ForwardIterator first, ForwardIterator last, T value, T stride) {
      while (first != last) {
        *first++ = value;
        value += stride;
      }
    }

    /// Vertexing algorithms

    /// Adaptive Vertex Reconstruction algorithm
    void AdaptiveVertexReconstruction();
    /// Simple Merge Algorithm
    void AgglomerativeHierarchicalClustering();
    /// Find distance between centres of two clusters
    float CentralDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Compute the vertex parameters
    void computeAndSetVertexParameters(RecoVertex<>& vertex,
                                       const std::vector<float>& bin_centers,
                                       const std::vector<unsigned int>& counts);
    /// DBSCAN algorithm
    void DBSCAN();
    /// TDR histogramming algorithmn
    void FastHistoLooseAssociation();
    /// Histogramming algorithm
    void FastHisto(const TrackerTopology* tTopo);
    /// High pT Vertex Algorithm
    void HPV();
    /// Kmeans Algorithm
    void Kmeans();
    /// Find maximum distance in two clusters of tracks
    float MaxDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Find minimum distance in two clusters of tracks
    float MinDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Find average distance in two clusters of tracks
    float MeanDistance(RecoVertex<> cluster0, RecoVertex<> cluster1);
    /// Principal Vertex Reconstructor algorithm
    void PVR();

  private:
    const AlgoSettings* settings_;
    std::vector<RecoVertex<>> vertices_;
    unsigned int numMatchedVertices_;
    FitTrackCollection fitTracks_;
    unsigned int pv_index_;
    unsigned int iterations_;
  };

}  // end namespace l1tVertexFinder

#endif
