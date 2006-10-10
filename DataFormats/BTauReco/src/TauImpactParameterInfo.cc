#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/RefVector.h"

using namespace edm;
using namespace reco;
using namespace std;

double reco::TauImpactParameterInfo::discriminator (double ip_min,double ip_max,double sip_min,bool use_sign, bool use3D) const {

	double discriminator = isolatedTaus->discriminator();

	const TrackRef leadingTrack = isolatedTaus->leadingSignalTrack(0.4,1.);

	if(! leadingTrack.isNull()){
	  const TrackData* ipData = getTrackData(leadingTrack);
	  Measurement1D ip = ipData->transverseIp;
	  if(use3D) ip = ipData->ip3D;
	  if( ip.value() < ip_min ||
	      ip.value() > ip_max ||
	      ip.significance() < sip_min ){
		discriminator = 0;
	  }
	}
	return discriminator;
}
double reco::TauImpactParameterInfo::discriminator() const {
        //default discriminator: returns the value of the discriminator of the jet tag
	return jetTag->discriminator();
}

const reco::TauImpactParameterInfo::TrackData* TauImpactParameterInfo::getTrackData(reco::TrackRef trackRef) const {

        reco::TauImpactParameterInfo::TrackDataAssociation::const_iterator iter
//	map<TrackRef,TrackData>::const_iterator iter
	    = trackDataMap.find(trackRef);

        if (iter != trackDataMap.end()) return &(iter->val);
//	if (iter != trackDataMap.end()) return &(iter->second);

	return 0; // if track not found return 0
}

void reco::TauImpactParameterInfo::storeTrackData(reco::TrackRef trackRef,
                  const reco::TauImpactParameterInfo::TrackData& trackData) {

	trackDataMap.insert(trackRef, trackData);
//	trackDataMap[trackRef] = trackData;
}

void reco::TauImpactParameterInfo::setJetTag(const JetTagRef myRef) {
	jetTag = myRef;
}

const JetTagRef & reco::TauImpactParameterInfo::getJetTag() const {
	return jetTag;
}


void reco::TauImpactParameterInfo::setIsolatedTauTag(const IsolatedTauTagInfoRef isolationRef){
	isolatedTaus = isolationRef;
}

const IsolatedTauTagInfoRef& reco::TauImpactParameterInfo::getIsolatedTauTag() const {
        return isolatedTaus;
}

