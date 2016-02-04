#include <string>

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"

using namespace reco;

TrackIPTagInfo::SortCriteria TrackSorting::getCriterium(const std::string &name)
{
	if (name == "sip3dSig")
		return TrackIPTagInfo::IP3DSig;
	if (name == "prob3d")
		return TrackIPTagInfo::Prob3D;
	if (name == "sip2dSig")
		return TrackIPTagInfo::IP2DSig;
	if (name == "prob2d")
		return TrackIPTagInfo::Prob2D;
	if (name == "sip2dVal")
		return TrackIPTagInfo::IP2DValue;

	throw cms::Exception("InvalidArgument")
		<< "Identifier \"" << name << "\" does not represent a valid "
		<< "track sorting criterium." << std::endl;
}
