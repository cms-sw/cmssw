#ifndef RecoEgamma_EgammaTools_ConversionFinder_h
#define RecoEgamma_EgammaTools_ConversionFinder_h
/** \class reco::ConversionFinder ConversionFinder.h RecoEgamma/EgammaTools/interface/ConversionFinder.h
  *  
  * Conversion finding and rejection code 
  * Returns true or false given a GsfElectron and track collection
  * Uses simple geometric methods to determine whether or not the 
  * electron did indeed come from a conversion
  * \author Puneeth Kalavase, University Of California, Santa Barbara
  *
  * \version $Id: ConversionFinder.h,v 1.2 2009/10/19 11:21:33 kalavase Exp $
  *
  */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/VectorUtil.h"

class ConversionFinder {
 public:
  ConversionFinder();
  ~ConversionFinder();
  //bField has to be supplied in Tesla
  
  static reco::TrackRef getConversionPartnerTrack(const reco::GsfElectron& gsfElectron, 
						  const edm::Handle<reco::TrackCollection>& track_h,
						  const float bFieldAtOrigin,
						  const float maxAbsDist = 0.02,
						  const float maxAbsDCot = 0.02,
						  const float minFracSharedHits = 0.45);

  //very simple function that calls getConversionPartnerTrack
  //if such a conversion partner track is found, will return true, else false
  static bool isElFromConversion(const reco::GsfElectron& gsfElectron, 
				 const edm::Handle<reco::TrackCollection>& track_h,
				 const float bFieldAtOrigin,
				 const float maxAbsDist = 0.02,
				 const float maxAbsDCot = 0.02,
				 const float minFracSharedHits = 0.45);

  
  static std::pair<double, double> getConversionInfo(math::XYZTLorentzVector trk1_p4, 
						     int trk1_q, float trk1_d0, 
						     math::XYZTLorentzVector trk2_p4,
						     int trk2_q, float trk2_d0,
						     float bFieldAtOrigin);
  
};
#endif
