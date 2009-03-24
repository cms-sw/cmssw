#include <memory>
#include <cmath>
#include <map>

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

using namespace reco;

// map helper - for some reason RefToBase lacks operator < (...)
namespace {
	template<typename T>
	struct RefToBaseLess : public std::binary_function<edm::RefToBase<T>,
	                                                   edm::RefToBase<T>,
	                                                   bool> {
		inline bool operator()(const edm::RefToBase<T> &r1,
		                       const edm::RefToBase<T> &r2) const
		{
			return r1.id() < r2.id() ||
			       (r1.id() == r2.id() && r1.key() < r2.key());
		}
	};
}

typedef std::map<TrackBaseRef, TransientTrack,
                 RefToBaseLess<Track> > TransientTrackMap;

static double trackVertexCompat(const Vertex &vtx,
                                const TransientTrack &track)
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

static const TransientTrack &convert(TransientTrackMap &map,
                                     const TransientTrackBuilder &builder,
                                     const TrackBaseRef &track)
{
	TransientTrackMap::iterator pos = map.lower_bound(track);
	if (pos != map.end() && pos->first == track)
		return pos->second;

	// the castTo will only work with regular, i.e. no GsfTracks
	// the interface is not intrinsically polymorph...
	return map.insert(pos,
		std::make_pair(track, builder.build(
					track.castTo<TrackRef>())))->second;
}

namespace reco {

class JetSignalVertexCompatibility : public edm::EDProducer {
    public:
	JetSignalVertexCompatibility(const edm::ParameterSet &params);
	~JetSignalVertexCompatibility();

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	const edm::InputTag	jetTracksAssocLabel;
	const edm::InputTag	primaryVerticesLabel;
	const double		primaryVertexThreshold;
};

JetSignalVertexCompatibility::JetSignalVertexCompatibility(
					const edm::ParameterSet &params) :
	jetTracksAssocLabel(params.getParameter<edm::InputTag>("jetTracksAssoc")),
	primaryVerticesLabel(params.getParameter<edm::InputTag>("primaryVertices")),
	primaryVertexThreshold(params.getParameter<double>("primaryVertexThreshold"))
{
	produces<JetFloatAssociation::Container>();
}

JetSignalVertexCompatibility::~JetSignalVertexCompatibility()
{
}

void JetSignalVertexCompatibility::produce(edm::Event &event,
                                           const edm::EventSetup &es)
{
	edm::ESHandle<TransientTrackBuilder> trackBuilder;
	es.get<TransientTrackRecord>().get("TransientTrackBuilder",
	                                   trackBuilder);

	edm::Handle<JetTracksAssociationCollection> jetTracksAssoc;
	event.getByLabel(jetTracksAssocLabel, jetTracksAssoc);

	edm::Handle<VertexCollection> primaryVertices;
	event.getByLabel(primaryVerticesLabel, primaryVertices);

	std::auto_ptr<JetFloatAssociation::Container> result(
		new JetFloatAssociation::Container(jetTracksAssoc->keyProduct()));

	for(unsigned int i = 0; i < result->size(); i++)
		result->setValue(i, -1.);

	// don't convert tracks into transient tracks twice
	TransientTrackMap trackMap;

	for(JetTracksAssociationCollection::const_iterator iter =
						jetTracksAssoc->begin();
	    iter != jetTracksAssoc->end(); ++iter) {
		const TrackRefVector &tracks = iter->second;

		double sigSum = 0.;
		double bkgSum = 0.;

		for(TrackRefVector::const_iterator track = tracks.begin();
		    track != tracks.end(); ++track) {
			if (!(*track)->quality(TrackBase::tight))
				continue;

			const TransientTrack &transientTrack =
					convert(trackMap, *trackBuilder,
					        TrackBaseRef(*track));

			double bestCompat = 1000.;
			int best = -1;

			for(int i = 0; i < (int)primaryVertices->size(); i++) {
				double compat =
					trackVertexCompat(
						(*primaryVertices)[i],
						transientTrack);
				if (compat < bestCompat) {
					best = i;
					bestCompat = compat;
				}
			}

			if (best < 0 || bestCompat > primaryVertexThreshold)
				continue;

			*(best == 0 ? &sigSum : &bkgSum) += (*track)->pt();
		}

		double sum = sigSum + bkgSum;
		if (sum < 1.0e-9)
			continue;

		(*result)[iter->first] = sigSum / sum;
	}

	event.put(result);
}

} // namespace reco

using reco::JetSignalVertexCompatibility;
DEFINE_FWK_MODULE(JetSignalVertexCompatibility);
