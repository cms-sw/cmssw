#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"

using namespace reco; 

TrackSelector::TrackSelector(const edm::ParameterSet &params) :
	sip2dValMin(params.getParameter<double>("sip2dValMin")),
	sip2dValMax(params.getParameter<double>("sip2dValMax")),
	sip2dSigMin(params.getParameter<double>("sip2dSigMin")),
	sip2dSigMax(params.getParameter<double>("sip2dSigMax")),
	sip3dValMin(params.getParameter<double>("sip3dValMin")),
	sip3dValMax(params.getParameter<double>("sip3dValMax")),
	sip3dSigMin(params.getParameter<double>("sip3dSigMin")),
	sip3dSigMax(params.getParameter<double>("sip3dSigMax"))
{
}

bool
TrackSelector::operator () (const Track &track,
                            const TrackIPTagInfo::TrackIPData &ipData) const
{
	return ipData.ip2d.value()        >= sip2dValMin &&
	       ipData.ip2d.value()        <= sip2dValMax &&
	       ipData.ip2d.significance() >= sip2dSigMin &&
	       ipData.ip2d.significance() <= sip2dSigMax &&
	       ipData.ip3d.value()        >= sip3dValMin &&
	       ipData.ip3d.value()        <= sip3dValMax &&
	       ipData.ip3d.significance() >= sip3dSigMin &&
	       ipData.ip3d.significance() <= sip3dSigMax;
}
