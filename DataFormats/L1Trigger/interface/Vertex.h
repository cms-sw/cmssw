#ifndef DataFormats_L1TVertex_Vertex_h
#define DataFormats_L1TVertex_Vertex_h

#include <vector>

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t {

  class Vertex;
  typedef std::vector<Vertex> VertexCollection;

  class Vertex {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> Track_t;

    Vertex();
    Vertex(float pt, float z0, const std::vector<edm::Ptr<Track_t>>& tracks);
    ~Vertex();

    float pt() const;
    float z0() const;

    const std::vector<edm::Ptr<Track_t>>& tracks() const;

  private:
    float pt_;
    float z0_;
    std::vector<edm::Ptr<Track_t>> tracks_;
  };

}  // namespace l1t

#endif
