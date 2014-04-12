#ifndef RecoBTag_SecondaryVertex_TrackSelector_h
#define RecoBTag_SecondaryVertex_TrackSelector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

namespace reco {

class TrackSelector {
    public:
	TrackSelector(const edm::ParameterSet &params);
	~TrackSelector() {}

	bool operator() (const reco::Track &track,
	                 const reco::TrackIPTagInfo::TrackIPData &ipData,
	                 const reco::Jet &jet,
	                 const GlobalPoint &pv) const;

    private:
	bool				selectQuality;
	reco::TrackBase::TrackQuality	quality;
	unsigned int			minPixelHits;
	unsigned int			minTotalHits;
	double				minPt;
	double				maxNormChi2;
	double				maxJetDeltaR;
	double				maxDistToAxis;
	double				maxDecayLen;
	double				sip2dValMin;
	double				sip2dValMax;
	double				sip2dSigMin;
	double				sip2dSigMax;
	double				sip3dValMin;
	double				sip3dValMax;
	double				sip3dSigMin;
	double				sip3dSigMax;
	bool                            useVariableJTA_;
	reco::TrackIPTagInfo::variableJTAParameters varJTApars;
};

} // namespace reco

#endif // RecoBTag_SecondaryVertex_TrackSelector_h
