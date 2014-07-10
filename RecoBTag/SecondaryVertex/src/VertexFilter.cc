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

static unsigned int
computeSharedTracks(const Vertex &pv, const std::vector<CandidatePtr> &svTracks,
                    double minTrackWeight)
{
/*      std::set<TrackRef> pvTracks;
        for(std::vector<TrackBaseRef>::const_iterator iter = pv.tracks_begin();
            iter != pv.tracks_end(); iter++)
                if (pv.trackWeight(*iter) >= minTrackWeight)
                        pvTracks.insert(iter->castTo<TrackRef>());

        unsigned int count = 0;
        for(std::vector<TrackRef>::const_iterator iter = svTracks.begin();
            iter != svTracks.end(); iter++)
                count += pvTracks.count(*iter);

        return count;
*/
//FIXME TODO
return 0;
}
