#include <algorithm>
#include <iterator>
#include <set>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTag/SecondaryVertex/interface/VertexFilter.h"

using namespace reco; 

VertexFilter::VertexFilter(const edm::ParameterSet &params) :
	useTrackWeights(params.getParameter<bool>("useTrackWeights")),
	massMax(params.getParameter<double>("massMax")),
	fracPV(params.getParameter<double>("fracPV")),
	multiplicityMin(params.getParameter<unsigned int>("multiplicityMin")),
	distVal2dMin(params.getParameter<double>("distVal2dMin")),
	distVal2dMax(params.getParameter<double>("distVal2dMax")),
	distVal3dMin(params.getParameter<double>("distVal3dMin")),
	distVal3dMax(params.getParameter<double>("distVal3dMax")),
	distSig2dMin(params.getParameter<double>("distSig2dMin")),
	distSig2dMax(params.getParameter<double>("distSig2dMax")),
	distSig3dMin(params.getParameter<double>("distSig3dMin")),
	distSig3dMax(params.getParameter<double>("distSig3dMax")),
	maxDeltaRToJetAxis(params.getParameter<double>("maxDeltaRToJetAxis"))
{
}

static unsigned int computeSharedTracks(const Vertex &pv, const Vertex &sv)
{
	std::set<TrackRef> pvTracks;
	std::copy(pv.tracks_begin(), pv.tracks_end(),
	          std::inserter(pvTracks, pvTracks.begin()));

	unsigned int count = 0;
	for(track_iterator iter = sv.tracks_begin();
	    iter != sv.tracks_end(); iter++)
		count += pvTracks.count(*iter);

	return count;
}

bool VertexFilter::operator () (const Vertex &pv,
                                const SecondaryVertex &sv,
                                const GlobalVector &direction) const
{
	// minimum number of tracks at vertex

	if (sv.tracksSize() < multiplicityMin)
		return false;

	// flight distance limits (value and significance, 2d and 3d)

	if (sv.dist2d().value()        < distVal2dMin ||
	    sv.dist2d().value()        > distVal2dMax ||
	    sv.dist3d().value()        < distVal3dMin ||
	    sv.dist3d().value()        > distVal3dMax ||
	    sv.dist2d().significance() < distSig2dMin ||
	    sv.dist2d().significance() > distSig2dMax ||
	    sv.dist3d().significance() < distSig3dMin ||
	    sv.dist3d().significance() > distSig3dMax)
		return false;

	// compute fourvector sum of tracks as vertex and cut on inv. mass

	TrackKinematics kin(sv);

	double mass = useTrackWeights ? kin.weightedVectorSum().M()
	                              : kin.vectorSum().M();

	if (mass > massMax)
		return false;

	// find shared tracks between PV and SV

	unsigned int sharedTracks = computeSharedTracks(pv, sv);

	if ((double)sharedTracks / sv.tracksSize() > fracPV)
		return false;

	return true;
}
