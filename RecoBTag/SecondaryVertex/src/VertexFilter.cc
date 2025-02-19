#include <functional>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <set>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"
#include "RecoBTag/SecondaryVertex/interface/VertexFilter.h"

using namespace reco; 

VertexFilter::VertexFilter(const edm::ParameterSet &params) :
	useTrackWeights(params.getParameter<bool>("useTrackWeights")),
	minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
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
	maxDeltaRToJetAxis(params.getParameter<double>("maxDeltaRToJetAxis")),
	v0Filter(params.getParameter<edm::ParameterSet>("v0Filter"))
{
}

static unsigned int
computeSharedTracks(const Vertex &pv, const std::vector<TrackRef> &svTracks,
                    double minTrackWeight)
{
	std::set<TrackRef> pvTracks;
	for(std::vector<TrackBaseRef>::const_iterator iter = pv.tracks_begin();
	    iter != pv.tracks_end(); iter++)
		if (pv.trackWeight(*iter) >= minTrackWeight)
			pvTracks.insert(iter->castTo<TrackRef>());

	unsigned int count = 0;
	for(std::vector<TrackRef>::const_iterator iter = svTracks.begin();
	    iter != svTracks.end(); iter++)
		count += pvTracks.count(*iter);

	return count;
}

bool VertexFilter::operator () (const Vertex &pv,
                                const SecondaryVertex &sv,
                                const GlobalVector &direction) const
{
	std::vector<TrackRef> svTracks;
	for(std::vector<TrackBaseRef>::const_iterator iter = sv.tracks_begin();
	    iter != sv.tracks_end(); iter++)
		if (sv.trackWeight(*iter) >= minTrackWeight)
			svTracks.push_back(iter->castTo<TrackRef>());

	// minimum number of tracks at vertex

	if (svTracks.size() < multiplicityMin)
		return false;

	// invalid errors

	if (sv.dist2d().error() < 0 || sv.dist3d().error() < 0)
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

	// SV direction filter

	if (Geom::deltaR(sv.position() - pv.position(),
	                 (maxDeltaRToJetAxis > 0) ? direction : -direction)
						> std::abs(maxDeltaRToJetAxis))
		return false;

	// compute fourvector sum of tracks as vertex and cut on inv. mass

	TrackKinematics kin(sv);

	double mass = useTrackWeights ? kin.weightedVectorSum().M()
	                              : kin.vectorSum().M();

	if (mass > massMax)
		return false;

	// find shared tracks between PV and SV

	if (fracPV < 1.0) {
		unsigned int sharedTracks =
			computeSharedTracks(pv, svTracks, minTrackWeight);
		if ((double)sharedTracks / svTracks.size() > fracPV)
			return false;
	}

	// check for V0 vertex

	if (sv.hasRefittedTracks())
		return v0Filter(sv.refittedTracks());
	else
		return v0Filter(svTracks);
}
