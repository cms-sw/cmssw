#include <functional>
#include <cmath>
#include <map>

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetSignalVertexCompatibilityAlgo.h"

using namespace reco;

// helper
template<typename T>
inline bool
JetSignalVertexCompatibilityAlgo::RefToBaseLess<T>::operator()(
	const edm::RefToBase<T> &r1, const edm::RefToBase<T> &r2) const
{ return r1.id() < r2.id() || (r1.id() == r2.id() && r1.key() < r2.key()); }


JetSignalVertexCompatibilityAlgo::JetSignalVertexCompatibilityAlgo(
					double cut, double temperature) :
	cut(cut), temperature(temperature)
{}

JetSignalVertexCompatibilityAlgo::~JetSignalVertexCompatibilityAlgo()
{}

double JetSignalVertexCompatibilityAlgo::trackVertexCompat(
			const Vertex &vtx, const TransientTrack &track)
{
	GlobalPoint point1 = RecoVertex::convertPos(vtx.position());
	AnalyticalImpactPointExtrapolator extrap(track.field());
	TrajectoryStateOnSurface tsos =
			extrap.extrapolate(track.impactPointState(), point1);

	if (!tsos.isValid())
		return 1.0e6;

	GlobalPoint point2 = tsos.globalPosition();
	ROOT::Math::SVector<double, 3> dir(point1.x() - point2.x(),
	                                   point1.y() - point2.y(),
	                                   point1.z() - point2.z());
	GlobalError cov = RecoVertex::convertError(vtx.covariance()) +
	                  tsos.cartesianError().position();

	return ROOT::Math::Mag2(dir) /
	       std::sqrt(ROOT::Math::Similarity(cov.matrix_new(), dir));
}

const TransientTrack&
JetSignalVertexCompatibilityAlgo::convert(const TrackBaseRef &track) const
{
	TransientTrackMap::iterator pos = trackMap.lower_bound(track);
	if (pos != trackMap.end() && pos->first == track)
		return pos->second;

	// the castTo will only work with regular, i.e. no GsfTracks
	// the interface is not intrinsically polymorph...
	return trackMap.insert(pos,
		std::make_pair(track, trackBuilder->build(
					track.castTo<TrackRef>())))->second;
}

double JetSignalVertexCompatibilityAlgo::activation(double compat) const
{
	return 1. / (std::exp((compat - cut) / temperature) + 1.);
}

std::vector<float>
JetSignalVertexCompatibilityAlgo::compatibility(
				const reco::VertexCollection &vertices,
				const reco::TrackRefVector &tracks) const
{
	std::vector<float> result(vertices.size(), 0.);
	float sum = 0.;

	for(TrackRefVector::const_iterator track = tracks.begin();
	    track != tracks.end(); ++track) {
		const TransientTrack &transientTrack =
					convert(TrackBaseRef(*track));

		for(unsigned int i = 0; i < vertices.size(); i++) {
			double compat =
				trackVertexCompat(vertices[i], transientTrack);
			double contribution =
				activation(compat) * (*track)->pt();

			result[i] += contribution;
			sum += contribution;
		}
	}

	if (sum < 1.0e-9) {
		for(unsigned int i = 0; i < result.size(); i++)
			result[i] = 1.0 / result.size();
	} else {
		for(unsigned int i = 0; i < result.size(); i++)
			result[i] /= sum;
	}

	return result;
}

void JetSignalVertexCompatibilityAlgo::resetEvent(
					const TransientTrackBuilder *builder)
{
	trackMap.clear();
	trackBuilder = builder;
}

