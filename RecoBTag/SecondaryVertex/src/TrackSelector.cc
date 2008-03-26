#include <Math/GenVector/VectorUtil.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"

using namespace reco; 
using namespace ROOT::Math;

TrackSelector::TrackSelector(const edm::ParameterSet &params) :
	minPixelHits(params.getParameter<unsigned int>("pixelHitsMin")),
	minTotalHits(params.getParameter<unsigned int>("totalHitsMin")),
	minPt(params.getParameter<double>("ptMin")),
	maxNormChi2(params.getParameter<double>("normChi2Max")),
	maxJetDeltaR(params.getParameter<double>("jetDeltaRMax")),
	sip2dValMin(params.getParameter<double>("sip2dValMin")),
	sip2dValMax(params.getParameter<double>("sip2dValMax")),
	sip2dSigMin(params.getParameter<double>("sip2dSigMin")),
	sip2dSigMax(params.getParameter<double>("sip2dSigMax")),
	sip3dValMin(params.getParameter<double>("sip3dValMin")),
	sip3dValMax(params.getParameter<double>("sip3dValMax")),
	sip3dSigMin(params.getParameter<double>("sip3dSigMin")),
	sip3dSigMax(params.getParameter<double>("sip3dSigMax"))
{
	std::string qualityClass =
			params.getParameter<std::string>("qualityClass");
	if (qualityClass == "any" || qualityClass == "Any" ||
	    qualityClass == "ANY" || qualityClass == "") {
		selectQuality = false;
		quality = reco::TrackBase::undefQuality;
	} else {
		selectQuality = true;
		quality = reco::TrackBase::qualityByName(qualityClass);
	}
}

bool
TrackSelector::operator () (const Track &track,
                            const TrackIPTagInfo::TrackIPData &ipData,
                            const Jet &jet) const
{
	return (!selectQuality || track.quality(quality)) &&
	       (minPixelHits <= 0 ||
	        track.hitPattern().numberOfValidPixelHits() >= minPixelHits) &&
	       (minTotalHits <= 0 ||
	        track.hitPattern().numberOfValidHits() >= minPixelHits) &&
	       track.pt() >= minPt &&
	       track.normalizedChi2() < maxNormChi2 &&
	       VectorUtil::DeltaR(jet.momentum(), track.momentum()) < maxJetDeltaR &&
	       ipData.ip2d.value()        >= sip2dValMin &&
	       ipData.ip2d.value()        <= sip2dValMax &&
	       ipData.ip2d.significance() >= sip2dSigMin &&
	       ipData.ip2d.significance() <= sip2dSigMax &&
	       ipData.ip3d.value()        >= sip3dValMin &&
	       ipData.ip3d.value()        <= sip3dValMax &&
	       ipData.ip3d.significance() >= sip3dSigMin &&
	       ipData.ip3d.significance() <= sip3dSigMax;
}
