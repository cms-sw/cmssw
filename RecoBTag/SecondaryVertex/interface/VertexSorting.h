#ifndef RecoBTag_SecondaryVertex_VertexSorting_h
#define RecoBTag_SecondaryVertex_VertexSorting_h

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/VertexSorting.h"



namespace reco {

template <class SecondaryVertex>
class VertexSorting {
    public:
	VertexSorting(const edm::ParameterSet &params):
		sortCriterium(getSortCriterium(params.getParameter<std::string>("sortCriterium"))){}
	~VertexSorting() {}

	std::vector<unsigned int>
	operator () (const std::vector<SecondaryVertex>&svCandidates) const;

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

template <class SecondaryVertex>
typename VertexSorting<SecondaryVertex>::SortCriterium
VertexSorting<SecondaryVertex>::getSortCriterium(const std::string &criterium)
{
	if (criterium == "dist3dError")
		return sortDist3dErr;
	if (criterium == "dist3dValue")
		return sortDist3dVal;
	if (criterium == "dist3dSignificance")
		return sortDist3dSig;
	if (criterium == "dist2dError")
		return sortDist2dErr;
	if (criterium == "dist2dValue")
		return sortDist2dVal;
	if (criterium == "dist2dSignificance")
		return sortDist2dSig;

	throw cms::Exception("InvalidArgument")
		<< "Vertex sort criterium \"" << criterium << "\" is invalid."
		<< std::endl;
}

// identify most probable SV (closest to interaction point, significance-wise)
// FIXME: identify if this is the best strategy!
template <class SecondaryVertex>
std::vector<unsigned int> VertexSorting<SecondaryVertex>::operator () (
		const std::vector<SecondaryVertex> &svCandidates) const
{
	Measurement1D (SecondaryVertex::*measurementFn)() const = 0;
	switch(sortCriterium) {
	    case sortDist3dErr:
	    case sortDist3dVal:
	    case sortDist3dSig:
		measurementFn = &SecondaryVertex::dist3d;
		break;
	    case sortDist2dErr:
	    case sortDist2dVal:
	    case sortDist2dSig:
		measurementFn = &SecondaryVertex::dist2d;
		break;
	}

	double (Measurement1D::*valueFn)() const = 0;
	switch(sortCriterium) {
	    case sortDist3dErr:
	    case sortDist2dErr:
		valueFn = &Measurement1D::error;
		break;
	    case sortDist3dVal:
	    case sortDist2dVal:
		valueFn = &Measurement1D::value;
		break;
	    case sortDist3dSig:
	    case sortDist2dSig:
		valueFn = &Measurement1D::significance;
		break;
	}

	std::multimap<double, unsigned int> sortedMap;
	unsigned int i = 0;
	for(typename std::vector<SecondaryVertex>::const_iterator iter =
		svCandidates.begin(); iter != svCandidates.end(); iter++) {

		double value = std::abs((((*iter).*measurementFn)().*valueFn)());
		sortedMap.insert(std::make_pair(value, i++));
	}

	std::vector<unsigned int> result;
	for(std::multimap<double, unsigned int>::const_iterator iter =
		sortedMap.begin(); iter != sortedMap.end(); iter++)
		result.push_back(iter->second);

	return result;
}
} // namespace reco

#endif // RecoBTag_SecondaryVertex_VertexSorting_h
