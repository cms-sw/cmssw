#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace edm;
using namespace reco;
using namespace std;

float reco::TauImpactParameterInfo::discriminator(
    double ip_min, double ip_max, double sip_min, bool use_sign, bool use3D) const {
  double discriminator = isolatedTaus->discriminator();

  const TrackRef leadingTrack = isolatedTaus->leadingSignalTrack(0.4, 1.);

  if (!leadingTrack.isNull()) {
    const TauImpactParameterTrackData* ipData = getTrackData(leadingTrack);
    Measurement1D ip = ipData->transverseIp;
    if (use3D)
      ip = ipData->ip3D;

    if (ip.value() < ip_min || ip.value() > ip_max || ip.significance() < sip_min) {
      discriminator = 0;
    }
  }
  return discriminator;
}
float reco::TauImpactParameterInfo::discriminator() const {
  //default discriminator: returns the value of the discriminator of the jet tag
  return isolatedTaus->discriminator();
}

const reco::TauImpactParameterTrackData* TauImpactParameterInfo::getTrackData(const reco::TrackRef& trackRef) const {
  reco::TrackTauImpactParameterAssociationCollection::const_iterator iter = trackDataMap.find(trackRef);

  if (iter != trackDataMap.end())
    return &(iter->val);

  return nullptr;  // if track not found return 0
}

void reco::TauImpactParameterInfo::storeTrackData(const reco::TrackRef& trackRef,
                                                  const reco::TauImpactParameterTrackData& trackData) {
  trackDataMap.insert(trackRef, trackData);
}

void reco::TauImpactParameterInfo::setIsolatedTauTag(const IsolatedTauTagInfoRef& isolationRef) {
  isolatedTaus = isolationRef;
}

const IsolatedTauTagInfoRef& reco::TauImpactParameterInfo::getIsolatedTauTag() const { return isolatedTaus; }
