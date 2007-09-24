#include <limits>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/VertexSelector.h"

using namespace reco; 

VertexSelector::SortCriterium
VertexSelector::getSortCriterium(const std::string &criterium)
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

VertexSelector::VertexSelector(const edm::ParameterSet &params) :
	sortCriterium(getSortCriterium(params.getParameter<std::string>("sortCriterium")))
{
}

// identify most probable SV (closest to interaction point)
// FIXME: identify if this is the best strategy!

const SecondaryVertex* VertexSelector::operator () (
		const std::vector<SecondaryVertex> &svCandidates) const
{
	const SecondaryVertex *bestSV = 0;
	double bestValue = std::numeric_limits<double>::max();

	Measurement1D (SecondaryVertex::*measurementFn)() const;
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

	double (Measurement1D::*valueFn)() const;
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

	for(std::vector<SecondaryVertex>::const_iterator iter =
		svCandidates.begin(); iter != svCandidates.end(); iter++) {

		double value = std::abs((((*iter).*measurementFn)().*valueFn)());
		if (value < bestValue) {
			bestValue = value;
			bestSV = &*iter;
		}
	}

	return bestSV;
}
