#ifndef RecoBTag_SecondaryVertex_V0Filter_h
#define RecoBTag_SecondaryVertex_V0Filter_h

#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {

class V0Filter {
    public:
	V0Filter(const edm::ParameterSet &params);
	~V0Filter() {}

	bool operator () (const reco::TrackRef *tracks, unsigned int n) const;
	bool operator () (const reco::Track *tracks, unsigned int n) const;
	bool operator () (const std::vector<reco::CandidatePtr> &tracks) const;
	bool operator () (const std::vector<const Track *> &tracks) const;

       
	inline bool
	operator () (const std::vector<reco::TrackRef> &tracks) const
	{ return (*this)(&tracks[0], tracks.size()); }

	inline bool
	operator () (const std::vector<reco::Track> &tracks) const
	{ return (*this)(&tracks[0], tracks.size()); }

	bool
	operator () (const reco::Track * const *tracks, unsigned int n) const;
    private:

	double	k0sMassWindow;
};

} // namespace reco

#endif // RecoBTag_SecondaryVertex_V0Filter_h
