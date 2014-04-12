#ifndef JetSignalVertexCompatibilityAlgo_h
#define JetSignalVertexCompatibilityAlgo_h

#include <functional>
#include <vector>
#include <map>

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

namespace reco {

class JetSignalVertexCompatibilityAlgo {
    public:
	JetSignalVertexCompatibilityAlgo(double cut, double temperature);
	~JetSignalVertexCompatibilityAlgo();

	std::vector<float> compatibility(
				const reco::VertexCollection &vertices,
				const reco::TrackRefVector &tracks) const;

	void resetEvent(const TransientTrackBuilder *trackBuilder);

    private:
	template<typename T>
	struct RefToBaseLess : public std::binary_function<edm::RefToBase<T>,
	                                                   edm::RefToBase<T>,
                                                           bool> {
		bool operator()(const edm::RefToBase<T> &r1,
		                const edm::RefToBase<T> &r2) const;
	};

	typedef std::map<reco::TrackBaseRef, reco::TransientTrack,
	                 RefToBaseLess<reco::Track> > TransientTrackMap;

	const TransientTrack &convert(const reco::TrackBaseRef &track) const;
	double activation(double compat) const;

	static double trackVertexCompat(const reco::Vertex &vtx,
	                                const TransientTrack &track);

	mutable TransientTrackMap	trackMap;
	const TransientTrackBuilder	*trackBuilder;

	const double			cut;
	const double			temperature;
};

} // namespace reco

#endif // JetSignalVertexCompatibilityAlgo_h
