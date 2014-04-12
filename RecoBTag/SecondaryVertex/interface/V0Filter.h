#ifndef RecoBTag_SecondaryVertex_V0Filter_h
#define RecoBTag_SecondaryVertex_V0Filter_h

#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

class V0Filter {
    public:
	V0Filter(const edm::ParameterSet &params);
	~V0Filter() {}

	bool operator () (const reco::TrackRef *tracks, unsigned int n) const;
	bool operator () (const reco::Track *tracks, unsigned int n) const;

	inline bool
	operator () (const std::vector<reco::TrackRef> &tracks) const
	{ return (*this)(&tracks[0], tracks.size()); }

	inline bool
	operator () (const std::vector<reco::Track> &tracks) const
	{ return (*this)(&tracks[0], tracks.size()); }

    private:
	inline bool
	operator () (const reco::Track **tracks, unsigned int n) const;

	double	k0sMassWindow;
};

} // namespace reco

#endif // RecoBTag_SecondaryVertex_V0Filter_h
