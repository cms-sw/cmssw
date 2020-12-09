#ifndef __L1Trigger_VertexFinder_RecoVertex_h__
#define __L1Trigger_VertexFinder_RecoVertex_h__

#include <set>
#include <vector>

#include "L1Trigger/VertexFinder/interface/L1TrackTruthMatched.h"
#include "L1Trigger/VertexFinder/interface/TP.h"

namespace l1tVertexFinder {

  class RecoVertex {
  public:
    /// Constructor and destructor
    RecoVertex() {
      z0_ = -999.;
      z0width_ = -999.;
      pT_ = -999.;
      highestPt_ = -999.;
      pv_ = false;
      highPt_ = false;
      numHighPtTracks_ = 0;
      tracks_.clear();
    }
    ~RecoVertex() {}

    /// Operators
    RecoVertex& operator+=(const RecoVertex& rhs){
      this->tracks_.insert(std::end(this->tracks_), std::begin(rhs.tracks()), std::end(rhs.tracks()));
      return *this;
    }

    /// Clear track vector
    void clear() { tracks_.clear(); }
    /// Compute vertex parameters
    void computeParameters(unsigned int weightedmean = false, double highPtThreshold = 50., int highPtBehavior = -1);
    /// Contain high-pT track?
    bool hasHighPt() const { return highPt_; }
    /// highest track pT in the vertex
    double highestPt() const { return highestPt_; }
    /// Assign fitted track to this vertex
    void insert(const L1Track* fitTrack) { tracks_.push_back(fitTrack); }
    /// Set primary vertex tag
    void isPrimary(bool is) { pv_ = is; }
    /// Number of high-pT tracks (pT > 10 GeV)
    unsigned int numHighPtTracks() const { return numHighPtTracks_; }
    /// Number of tracks originating from this vertex
    unsigned int numTracks() const { return tracks_.size(); }
    /// True if primary vertex
    bool primaryVertex() const { return pv_; }
    /// Sum of fitted tracks transverse momentum [GeV]
    double pT() const { return pT_; }
    /// Tracking Particles in vertex
    const std::vector<const L1Track*>& tracks() const { return tracks_; }
    /// Set z0 position [cm]
    void setZ(double z) { z0_ = z; }
    /// Vertex z0 position [cm]
    double z0() const { return z0_; }
    /// Vertex z0 width [cm]
    double z0width() const { return z0width_; }

  private:
    double z0_;
    double z0width_;
    double pT_;
    double highestPt_;
    std::vector<const L1Track*> tracks_;
    bool pv_;
    bool highPt_;
    unsigned int numHighPtTracks_;
  };

}  // namespace l1tVertexFinder

#endif
