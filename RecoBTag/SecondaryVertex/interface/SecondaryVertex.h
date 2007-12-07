#ifndef RecoBTag_SecondaryVertex_SecondaryVertex_h
#define RecoBTag_SecondaryVertex_SecondaryVertex_h

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace reco {

class SecondaryVertex : public reco::Vertex {
    public:
	SecondaryVertex() {}
	SecondaryVertex(const reco::Vertex &pv,
	                const reco::Vertex &sv,
	                const GlobalVector &direction,
	                bool withPVError = false) :
		reco::Vertex(sv),
		dist2d_(computeDist2d(pv, sv, direction, withPVError)),
		dist3d_(computeDist3d(pv, sv, direction, withPVError)) {}
	~SecondaryVertex() {}

	inline Measurement1D dist2d() const { return dist2d_; }
	inline Measurement1D dist3d() const { return dist3d_; }

	static Measurement1D computeDist2d(
		const reco::Vertex &pv, const reco::Vertex &sv,
		const GlobalVector &direction, bool withPVError);
	static Measurement1D computeDist3d(
		const reco::Vertex &pv, const reco::Vertex &sv,
		const GlobalVector &direction, bool withPVError);

    private:
	Measurement1D dist2d_;
	Measurement1D dist3d_;
};

} // namespace reco

#endif // RecoBTag_SecondaryVertex_SecondaryVertex_h
