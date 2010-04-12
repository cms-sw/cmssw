#ifndef EgammaTools_ConversionFinder_h
#define EgammaTools_ConversionFinder_h
/** \class reco::ConversionFinder ConversionFinder.h RecoEgamma/EgammaTools/interface/ConversionFinder.h
  *  
  * Conversion finding and rejection code 
  * Uses simple geometric methods to determine whether or not the 
  * electron did indeed come from a conversion
  * \author Puneeth Kalavase, University Of California, Santa Barbara
  *
  * \version $Id: ConversionFinder.h,v 1.4 2009/11/18 20:40:25 kalavase Exp $
  *
  */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Math/VectorUtil.h"
#include "ConversionInfo.h"


/* 
   Class Looks for oppositely charged track in the 
   track collection with the minimum delta cot(theta) between the track
   and the electron's CTF track (if it doesn't exist, we use the electron's
   GSF track). Calculate the dist, dcot, point of conversion and the 
   radius of conversion for this pair and fill the ConversionInfo
*/

class ConversionFinder {    
 public:
  ConversionFinder();
  ~ConversionFinder();
  //bField has to be supplied in Tesla
  ConversionInfo getConversionInfo(const reco::GsfElectron& gsfElectron, 
				   const edm::Handle<reco::TrackCollection>& track_h, 
				   const double bFieldAtOrigin,
				   const double minFracSharedHits = 0.45);
  /*
    cuts tuned for high pt ( pt > 20 GeV )electrons
    fnc must be called after getConversionInfo is called
  */
  bool isFromConversion(double maxAbsDist = 0.2, double maxAbsDcot = 0.02);
  const reco::Track* getElectronTrack(const reco::GsfElectron& electron, const float minFracSharedHits = 0.45);
 private:
  ConversionInfo convInfo_;
};
#endif
