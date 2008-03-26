#ifndef RecoBTag_SecondaryVertex_VertexFilter_h
#define RecoBTag_SecondaryVertex_VertexFilter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"

namespace reco {

class VertexFilter {
    public:
	VertexFilter(const edm::ParameterSet &params);
	~VertexFilter() {}

	bool operator () (const reco::Vertex &pv, const SecondaryVertex &sv,
	                  const GlobalVector &direction) const;

    private:
	bool		useTrackWeights;
	double		minTrackWeight;
	double		massMax;
	double		fracPV;
	unsigned int	multiplicityMin;

	double		distVal2dMin;
	double		distVal2dMax;
	double		distVal3dMin;
	double		distVal3dMax;

	double		distSig2dMin;
	double		distSig2dMax;
	double		distSig3dMin;
	double		distSig3dMax;

	double		maxDeltaRToJetAxis;
	V0Filter	v0Filter;
};

} // namespace reco

#endif // RecoBTag_SecondaryVertex_VertexFilter_h
