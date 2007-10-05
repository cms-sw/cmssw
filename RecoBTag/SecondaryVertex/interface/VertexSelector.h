#ifndef RecoBTag_SecondaryVertex_VertexSelector_h
#define RecoBTag_SecondaryVertex_VertexSelector_h

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"

class VertexSelector {
    public:
	VertexSelector(const edm::ParameterSet &params);
	~VertexSelector() {}

	const SecondaryVertex*
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

#endif // RecoBTag_SecondaryVertex_VertexSelector_h
