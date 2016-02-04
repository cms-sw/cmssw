#ifndef RecoBTag_SecondaryVertex_VertexSorting_h
#define RecoBTag_SecondaryVertex_VertexSorting_h

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"

namespace reco {

class VertexSorting {
    public:
	VertexSorting(const edm::ParameterSet &params);
	~VertexSorting() {}

	std::vector<unsigned int>
	operator () (const std::vector<SecondaryVertex> &svCandidates) const;

    private:
	enum SortCriterium {
		sortDist3dVal = 0,
		sortDist3dErr,
		sortDist3dSig,
		sortDist2dErr,
		sortDist2dSig,
		sortDist2dVal
	};

	static SortCriterium getSortCriterium(const std::string &criterium);

	SortCriterium	sortCriterium;
};

} // namespace reco

#endif // RecoBTag_SecondaryVertex_VertexSorting_h
