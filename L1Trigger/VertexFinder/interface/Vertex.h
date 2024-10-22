#ifndef __L1Trigger_VertexFinder_Vertex_h__
#define __L1Trigger_VertexFinder_Vertex_h__

#include <vector>

#include "L1Trigger/VertexFinder/interface/TP.h"

namespace l1tVertexFinder {

  class Vertex {
  public:
    // Fill useful info about tracking particle.
    Vertex() { Vertex(-999.); }

    Vertex(double vz) : vz_(vz) {
      z0_ = -999.;
      z0width_ = -999.;
      pT_ = -999.;
    }

    ~Vertex() {}

    /// Tracking Particles in vertex
    const std::vector<TP>& tracks() const { return tracks_; }
    /// Number of tracks originating from this vertex
    unsigned int numTracks() const { return tracks_.size(); }
    /// Assign TP to this vertex
    void insert(TP& tp) { tracks_.push_back(tp); }
    /// Compute vertex parameters
    void computeParameters();
    /// Sum of fitted tracks transverse momentum [GeV]
    double pT() const { return pT_; }
    /// Vertex z0 position [cm]
    double z0() const { return z0_; }
    /// Vertex z0 width [cm]
    double z0width() const { return z0width_; }
    /// Vertex z position [cm]
    double vz() const { return vz_; }
    /// Reset/initialize all of the member data
    void reset();

  private:
    double vz_;
    double z0_;
    double z0width_;
    double pT_;

    std::vector<TP> tracks_;
  };

}  // end namespace l1tVertexFinder

#endif
