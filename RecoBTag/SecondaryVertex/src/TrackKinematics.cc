#include <cmath>
#include <vector>

#include <Math/GenVector/PxPyPzE4D.h>
#include <Math/GenVector/PxPyPzM4D.h>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoBTag/SecondaryVertex/interface/ParticleMasses.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"

using namespace reco;

TrackKinematics::TrackKinematics() :
	n(0), sumWeights(0)
{
}

TrackKinematics::TrackKinematics(const std::vector<Track> &tracks) :
	n(0), sumWeights(0)
{
	for(std::vector<Track>::const_iterator iter = tracks.begin();
	    iter != tracks.end(); iter++)
		add(*iter);
}

TrackKinematics::TrackKinematics(const TrackRefVector &tracks) :
	n(0), sumWeights(0)
{
	for(TrackRefVector::const_iterator iter = tracks.begin();
	    iter != tracks.end(); iter++)
		add(**iter);
}

TrackKinematics::TrackKinematics(const Vertex &vertex) :
	n(0), sumWeights(0)
{
	bool hasRefittedTracks = vertex.hasRefittedTracks();
	for(Vertex::trackRef_iterator iter = vertex.tracks_begin();
	    iter != vertex.tracks_end(); ++iter) {
		if (hasRefittedTracks)
			add(vertex.refittedTrack(*iter),
			    vertex.trackWeight(*iter));
		else
			add(**iter, vertex.trackWeight(*iter));
	}
}

TrackKinematics &TrackKinematics::operator += (const TrackKinematics &other)
{
	n += other.n;
	sumWeights += other.sumWeights;
	sum += other.sum;
	weightedSum += other.weightedSum;

	return *this;
}

void TrackKinematics::add(const Track &track, double weight)
{
	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > vec;

	vec.SetPx(track.px());
	vec.SetPy(track.py());
	vec.SetPz(track.pz());
	vec.SetM(ParticleMasses::piPlus);

	n++;
	sumWeights += weight;
	sum += vec;
	weightedSum += weight * vec;
}
