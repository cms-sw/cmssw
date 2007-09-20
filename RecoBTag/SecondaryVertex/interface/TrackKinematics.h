#ifndef RecoBTag_SecondaryVertex_TrackKinematics_h
#define RecoBTag_SecondaryVertex_TrackKinematics_h

#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class TrackKinematics {
    public:
	TrackKinematics();
	TrackKinematics(const std::vector<reco::Track> &tracks);
	TrackKinematics(const reco::Vertex &vertex);
	~TrackKinematics() {}

	void add(const reco::Track &track, double weight = 1.0);

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

#endif // RecoBTag_SecondaryVertex_TrackKinematics_h
