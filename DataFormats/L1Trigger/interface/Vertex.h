#ifndef DataFormats_L1TVertex_Vertex_h
#define DataFormats_L1TVertex_Vertex_h

#include <vector>

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t {

  class Vertex {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> Track_t;

    Vertex() : pt_(0.0), z0_(0.0) {}
    Vertex(float pt, float z0, const std::vector<edm::Ptr<Track_t>>& tracks) : pt_(pt), z0_(z0), tracks_(tracks) {}
    ~Vertex() {}

    float pt() const { return pt_; }
    float z0() const { return z0_; }

    const std::vector<edm::Ptr<Track_t>>& tracks() const { return tracks_; }

  private:
    float pt_;
    float z0_;
    std::vector<edm::Ptr<Track_t>> tracks_;
  };

  typedef std::vector<Vertex> VertexCollection;

}  // namespace l1t

#endif
