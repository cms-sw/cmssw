#include "DataFormats/L1Trigger/interface/Vertex.h"

namespace l1t {

  Vertex::Vertex() : pt_(0.0), z0_(0.0) {}

  Vertex::Vertex(float pt, float z0, const std::vector<edm::Ptr<Track_t>>& tracks)
      : pt_(pt), z0_(z0), tracks_(tracks) {}

  Vertex::~Vertex() {}

  float Vertex::pt() const { return pt_; }

  float Vertex::z0() const { return z0_; }

  const std::vector<edm::Ptr<Vertex::Track_t>>& Vertex::tracks() const { return tracks_; }

}  // namespace l1t
