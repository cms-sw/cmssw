#ifndef EgammaTools_ConversionFinder_h
#define EgammaTools_ConversionFinder_h
/** \class reco::ConversionFinder ConversionFinder.h RecoEgamma/EgammaTools/interface/ConversionFinder.h
  *
  * Conversion finding and rejection code
  * Uses simple geometric methods to determine whether or not the
  * electron did indeed come from a conversion
  * \author Puneeth Kalavase, University Of California, Santa Barbara
  *
  *
  */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
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

namespace egammaTools {
  //bField has to be supplied in Tesla

  //returns a vector of Conversion Infos.
  //you have to decide which is the best
  std::vector<ConversionInfo> getConversionInfos(const reco::GsfElectronCore&,
                                                 const edm::Handle<reco::TrackCollection>& ctftracks_h,
                                                 const edm::Handle<reco::GsfTrackCollection>& gsftracks_h,
                                                 const double bFieldAtOrigin,
                                                 const double minFracSharedHits = 0.45);

  //retruns the "best" Conversion Info after calling getConversionInfos
  ConversionInfo getConversionInfo(const reco::GsfElectronCore&,
                                   const edm::Handle<reco::TrackCollection>& ctftracks_h,
                                   const edm::Handle<reco::GsfTrackCollection>& gsftracks_h,
                                   const double bFieldAtOrigin,
                                   const double minFracSharedHits = 0.45);

  //retruns the "best" Conversion Info after calling getConversionInfos
  ConversionInfo getConversionInfo(const reco::GsfElectron& gsfElectron,
                                   const edm::Handle<reco::TrackCollection>& ctftracks_h,
                                   const edm::Handle<reco::GsfTrackCollection>& gsftracks_h,
                                   const double bFieldAtOrigin,
                                   const double minFracSharedHits = 0.45);

  //takes in a vector of candidate conversion partners
  //and arbitrates between them returning the one with the
  //smallest R=sqrt(dist*dist + dcot*dcot)
  ConversionInfo arbitrateConversionPartnersbyR(const std::vector<ConversionInfo>& v_convCandidates);

  //places different cuts on dist, dcot, delmissing hits and arbitration based on R = sqrt(dist*dist + dcot*dcot)
  ConversionInfo findBestConversionMatch(const std::vector<ConversionInfo>& v_convCandidates);

  //for backwards compatibility. Does not use the GSF track collection. This function will be
  //deprecated soon
  ConversionInfo getConversionInfo(const reco::GsfElectron& gsfElectron,
                                   const edm::Handle<reco::TrackCollection>& track_h,
                                   const double bFieldAtOrigin,
                                   const double minFracSharedHits = 0.45);

  // DEPRECATED
  bool isFromConversion(const ConversionInfo&, double maxAbsDist = 0.02, double maxAbsDcot = 0.02);

}  // namespace egammaTools
#endif
