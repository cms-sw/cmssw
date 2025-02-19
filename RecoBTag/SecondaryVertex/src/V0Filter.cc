#include <cmath>

#include <Math/GenVector/PxPyPzM4D.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoBTag/SecondaryVertex/interface/ParticleMasses.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

using namespace reco; 

V0Filter::V0Filter(const edm::ParameterSet &params) :
	k0sMassWindow(params.getParameter<double>("k0sMassWindow"))
{
}

inline bool
V0Filter::operator () (const reco::Track **tracks, unsigned int n) const
{
	// only check for K0s for now

	if (n != 2)
		return true;

	if (tracks[0]->charge() * tracks[1]->charge() > 0)
		return true;

	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > vec1;
	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > vec2;

	vec1.SetPx(tracks[0]->px());
	vec1.SetPy(tracks[0]->py());
	vec1.SetPz(tracks[0]->pz());
	vec1.SetM(ParticleMasses::piPlus);

	vec2.SetPx(tracks[1]->px());
	vec2.SetPy(tracks[1]->py());
	vec2.SetPz(tracks[1]->pz());
	vec2.SetM(ParticleMasses::piPlus);

	double invariantMass = (vec1 + vec2).M();
	if (std::abs(invariantMass - ParticleMasses::k0) < k0sMassWindow)
		return false;

	return true;
}

bool
V0Filter::operator () (const reco::TrackRef *tracks, unsigned int n) const
{
	std::vector<const reco::Track*> trackPtrs(n);
	for(unsigned int i = 0; i < n; i++)
		trackPtrs[i] = &*tracks[i];

	return (*this)(&trackPtrs[0], n);
}

bool
V0Filter::operator () (const reco::Track *tracks, unsigned int n) const
{
	std::vector<const reco::Track*> trackPtrs(n);
	for(unsigned int i = 0; i < n; i++)
		trackPtrs[i] = &tracks[i];

	return (*this)(&trackPtrs[0], n);
}
