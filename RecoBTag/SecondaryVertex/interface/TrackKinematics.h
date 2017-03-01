#ifndef RecoBTag_SecondaryVertex_TrackKinematics_h
#define RecoBTag_SecondaryVertex_TrackKinematics_h

#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

namespace reco {

class TrackKinematics {
    public:
	TrackKinematics();
	TrackKinematics(const std::vector<reco::Track> &tracks);
	TrackKinematics(const reco::TrackRefVector &tracks);
	TrackKinematics(const std::vector<reco::CandidatePtr> &tracks);
	TrackKinematics(const reco::CandidatePtrVector &tracks);
	TrackKinematics(const reco::Vertex &vertex);
	TrackKinematics(const reco::VertexCompositePtrCandidate &vertex):
 	       n(vertex.numberOfSourceCandidatePtrs()), sumWeights(vertex.numberOfSourceCandidatePtrs()),
	       sum(vertex.p4()),weightedSum(vertex.p4()){}

	~TrackKinematics() {}

	void add(const reco::Track &track, double weight = 1.0);
	void add(const reco::CandidatePtr &track);

	inline
	void add(const reco::TrackRef &track, double weight = 1.0)
	{return add(*track, weight); }

	TrackKinematics &operator += (const TrackKinematics &other);
	inline TrackKinematics operator + (const TrackKinematics &other)
	{ TrackKinematics copy = *this; copy += other; return copy; }

	inline unsigned int numberOfTracks() const { return n; }
	inline double sumOfWeights() const { return sumWeights; }

	inline const math::XYZTLorentzVector &vectorSum() const
	{ return sum; }
	inline const math::XYZTLorentzVector &weightedVectorSum() const
	{ return weightedSum; }

    private:
	unsigned int		n;
	double			sumWeights;
	math::XYZTLorentzVector	sum;
	math::XYZTLorentzVector	weightedSum;
};

} // namespace reco

#endif // RecoBTag_SecondaryVertex_TrackKinematics_h
