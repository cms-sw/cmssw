#ifndef RecoBTag_SecondaryVertex_TrackSorting_h
#define RecoBTag_SecondaryVertex_TrackSorting_h

#include <string>

#include "DataFormats/BTauReco/interface/IPTagInfo.h"

namespace TrackSorting {
	extern reco::btag::SortCriteria
	getCriterium(const std::string &name);
}

#endif // RecoBTag_SecondaryVertex_TrackSorting_h
