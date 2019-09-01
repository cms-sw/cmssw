#ifndef EgammaTools_ConversionInfo_h
#define EgammaTools_ConversionInfo_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/Point3D.h"

struct ConversionInfo {
  const double dist;
  const double dcot;
  const double radiusOfConversion;
  const math::XYZPoint pointOfConversion;
  // if the partner track is found in the  GSF track collection,
  // this is a ref to the GSF partner track
  const reco::TrackRef conversionPartnerCtfTk;
  // if the partner track is found in the  CTF track collection,
  // this is a ref to the CTF partner track
  const reco::GsfTrackRef conversionPartnerGsfTk;
  const int deltaMissingHits;
  const int flag;

  // flag 0: Partner track found in the CTF collection using the electron's CTF track
  // flag 1: Partner track found in the CTF collection using the electron's GSF track
  // flag 2: Partner track found in the GSF collection using the electron's CTF track
  // flag 3: Partner track found in the GSF collection using the electron's GSF track
};

#endif
