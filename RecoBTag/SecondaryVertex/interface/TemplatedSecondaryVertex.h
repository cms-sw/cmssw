#ifndef RecoBTag_SecondaryVertex_TemplatedSecondaryVertex_h
#define RecoBTag_SecondaryVertex_TemplatedSecondaryVertex_h

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace reco {

  template <class SV>
  class TemplatedSecondaryVertex : public SV {
  public:
    TemplatedSecondaryVertex() {}
    TemplatedSecondaryVertex(const reco::Vertex &pv,
                             const SV &sv,
                             const GlobalVector &direction,
                             bool withPVError = false)
        : SV(sv),
          dist1d_(computeDist1d(pv, sv, direction, withPVError)),
          dist2d_(computeDist2d(pv, sv, direction, withPVError)),
          dist3d_(computeDist3d(pv, sv, direction, withPVError)) {}

    ~TemplatedSecondaryVertex() {}  //NOLINT
    inline Measurement1D dist1d() const { return dist1d_; }
    inline Measurement1D dist2d() const { return dist2d_; }
    inline Measurement1D dist3d() const { return dist3d_; }
    static Measurement1D computeDist1d(const reco::Vertex &pv,
                                       const SV &sv,
                                       const GlobalVector &direction,
                                       bool withPVError);
    static Measurement1D computeDist2d(
        //		const reco::Vertex &pv, const reco::Vertex &sv,
        const reco::Vertex &pv,
        const SV &sv,
        const GlobalVector &direction,
        bool withPVError);
    static Measurement1D computeDist3d(
        //		const reco::Vertex &pv, const reco::Vertex &sv,
        const reco::Vertex &pv,
        const SV &sv,
        const GlobalVector &direction,
        bool withPVError);
    operator reco::Vertex();

  private:
    Measurement1D dist1d_;
    Measurement1D dist2d_;
    Measurement1D dist3d_;
  };

}  // namespace reco

#endif  // RecoBTag_SecondaryVertex_TemplatedSecondaryVertex_h
