#ifndef RecoBTag_SecondaryVertex_TrackSorting_h
#define RecoBTag_SecondaryVertex_TrackSorting_h

#include <string>

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

namespace TrackSorting {
	extern reco::TrackIPTagInfo::SortCriteria
	getCriterium(const std::string &name);
}

#endif // RecoBTag_SecondaryVertex_TrackSorting_h
