#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

// HEP library include files
//#include <Math/GenVector/VectorUtil.h>

// CMSSW include files
#include "DataFormats/JetReco/interface/Jet.h"

#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"

using namespace reco;

//
//-------------------------------------------------------------------------------
//

const reco::TrackRefVector IsolatedTauTagInfo::tracksInCone(const math::XYZVector& coneAxis, double coneSize, int coneType, double ptTrackMin, int& error) const 
{
  return tracksInCone(coneAxis, coneSize, coneType, ptTrackMin, 0., 1.e6, error);
}

const reco::TrackRefVector IsolatedTauTagInfo::tracksInCone(const math::XYZVector& coneAxis, double coneSize, int coneType, double ptTrackMin, 
							    double zPrimaryVertex, double dzTrackMax, int& error) const
{
//--- reset error flag
//   
//    error codes : 1 = undefined cone type
//
  error = 0;

  reco::TrackRefVector matchingTracks;
  switch ( coneType ) {
  case kEtaPhiCone :
    matchingTracks = coneIsolationAlgorithmEtaPhi_.operator()(coneAxis, coneSize, selectedTracks_, metricEtaPhi_);
    break;
  case kOpeningAngleCone :
    matchingTracks = coneIsolationAlgorithmAngle_(coneAxis, coneSize, selectedTracks_, metricAngle_);
    break;
  default:
    error = 1;
    return reco::TrackRefVector();
  }

  reco::TrackRefVector selectedTracks;
  for ( reco::TrackRefVector::const_iterator track = matchingTracks.begin();
	track != matchingTracks.end(); ++track ) {
    if ( (*track)->pt() > ptTrackMin ) {
      selectedTracks.push_back(*track);
    }
  }

  return selectedTracks;
}

//
//-------------------------------------------------------------------------------
//

const reco::TrackRef IsolatedTauTagInfo::leadingSignalTrack(double matchingConeSize, int matchingConeType,
							    double ptTrackMin, int& error) const 
{
  const Jet& jetRef = (*jet()); 
  math::XYZVector jetAxis(jetRef.px(), jetRef.py(), jetRef.pz());

  return leadingSignalTrack(jetAxis, matchingConeSize, matchingConeType, ptTrackMin, error);
}

const reco::TrackRef IsolatedTauTagInfo::leadingSignalTrack(const math::XYZVector& jetAxis, double matchingConeSize, int matchingConeType,
							    double ptTrackMin, int& error) const 
{
//--- reset error flag
//   
//    error codes : 1 = undefined cone type
//
  error = 0;

  const reco::TrackRefVector matchingConeTracks = tracksInCone(jetAxis, matchingConeSize, matchingConeType, ptTrackMin, error);
 
//--- check error code;
//    return NULL reference in case of errors
  if ( error != 0 ) {
    error = 1;
    return reco::TrackRef();
  }
  
  reco::TrackRef leadingTrack;
  double leadingTrackPt = 0.;
  for ( reco::TrackRefVector::const_iterator track = matchingConeTracks.begin();
	track != matchingConeTracks.end(); ++track ) {
    if ( (*track)->pt() > ptTrackMin    &&
	 (*track)->pt() > leadingTrackPt ) {
      leadingTrack = (*track);
      leadingTrackPt = leadingTrack->pt();
    }
  }
  
  return leadingTrack;
}

//
//-------------------------------------------------------------------------------
//

double IsolatedTauTagInfo::discriminator(const math::XYZVector& jetAxis, 
					 double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, double dzOtherTrackMax, 
					 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
					 unsigned int numTracksIsolationRingMax, int& error) const 
{
//--- reset error flag
//   
//    error codes : 1 = undefined matching cone type
//                  2 = undefined signal cone type
//                  3 = undefined isolation cone type
//                  4 = computation of fixed area isolation cone size failed
//
  error = 0;

//--- get leading (i.e. highest Pt) track within jet;
//    the leading track defines the axis of signal and isolation cones
//    and the primary event vertex
  const reco::TrackRef leadingTrack = leadingSignalTrack(jetAxis, matchingConeSize, matchingConeType, ptLeadingTrackMin, error);

//--- check error code;
//    return zero in case of errors
  if ( error != 0 ) {
    error = 1;
    return 0.;
  }

//--- return zero in case no leading track is selected
  if ( !leadingTrack ) return 0.;
  
  math::XYZVector coneAxis = leadingTrack->momentum();
  double zPrimaryVertex = leadingTrack->dz();

//--- select subset of tracks compatible with originating 
//    from the same primary event vertex as the leading track
  reco::TrackRefVector sameVertexTracks;
  for ( reco::TrackRefVector::const_iterator track = selectedTracks_.begin();
	track != selectedTracks_.end(); ++track ) {
    if ( fabs((*track)->dz() - zPrimaryVertex) < dzOtherTrackMax ) {
      sameVertexTracks.push_back(*track);
    }
  }

//--- compute number of selected tracks within signal cone
  reco::TrackRefVector signalConeTracks;
  switch ( signalConeType ) {
  case kEtaPhiCone :
    signalConeTracks = coneIsolationAlgorithmEtaPhi_(coneAxis, signalConeSize, sameVertexTracks, metricEtaPhi_);
    break;
  case kOpeningAngleCone :
    signalConeTracks = coneIsolationAlgorithmAngle_(coneAxis, signalConeSize, sameVertexTracks, metricAngle_);
    break;
  default :
    error = 2;
    return 0.;
  }

//--- compute number of selected tracks within isolation cone
//    (tracks within the signal cone are not excluded a priori from being in the isolation cone)
  reco::TrackRefVector isolationConeTracks;
  switch ( isolationConeType ) {
  case kEtaPhiCone :
    isolationConeTracks = coneIsolationAlgorithmEtaPhi_(coneAxis, isolationConeSize, sameVertexTracks, metricEtaPhi_);
    break;
  case kOpeningAngleCone :
    isolationConeTracks = coneIsolationAlgorithmAngle_(coneAxis, isolationConeSize, sameVertexTracks, metricAngle_);
    break;
  case kFixedAreaIsolationCone :
    {
      double isolationConeArea = isolationConeSize;
      int errorFlag = 0;
      const double etaMaxTrackingAcceptance = 2.5; // maximum pseudo-rapidity at which charged particle can be reconstructed in SiStrip + Pixel detectors
      FixedAreaIsolationCone fixedAreaIsolationCone;
      fixedAreaIsolationCone.setAcceptanceLimit(etaMaxTrackingAcceptance);
      double isolationConeOpeningAngle = fixedAreaIsolationCone(coneAxis.theta(), coneAxis.phi(), signalConeSize, isolationConeArea, errorFlag);
      if ( errorFlag != 0 ) {
	error = 4;
	return 0.;
      }
      isolationConeTracks = coneIsolationAlgorithmAngle_(coneAxis, isolationConeOpeningAngle, sameVertexTracks, metricAngle_);
    }
    break;
  default :
    error = 3;
    return 0.;
  }

//--- if difference in the number of tracks within isolation - signal cone
//    is equal to or smaller (higher) than "numTracksIsolationRingMax" parameter given as function argument
//    the jet is selected (rejected) as tau-jet 
  if ( (signalConeTracks.size() - isolationConeTracks.size()) <= numTracksIsolationRingMax ) 
    return 1.;
  else 
    return 0.;
}

double IsolatedTauTagInfo::discriminator(const math::XYZVector& jetAxis, 
					 double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, 
					 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
					 unsigned int numTracksIsolationRingMax, int& error) const 
{
  return discriminator(jetAxis, matchingConeSize, matchingConeType, ptLeadingTrackMin, ptOtherTracksMin, 1.e6, 
		       signalConeSize, signalConeType, isolationConeSize, isolationConeType, 
		       numTracksIsolationRingMax, error);
}

double IsolatedTauTagInfo::discriminator(double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, double dzOtherTrackMax, 
					 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
					 unsigned int numTracksIsolationRingMax, int& error) const 
{
  const Jet& jetRef = (*jet()); 
  math::XYZVector jetAxis(jetRef.px(), jetRef.py(), jetRef.pz());

  return discriminator(jetAxis, matchingConeSize, matchingConeType, ptLeadingTrackMin, ptOtherTracksMin, dzOtherTrackMax, 
		       signalConeSize, signalConeType, isolationConeSize, isolationConeType, 
		       numTracksIsolationRingMax, error);
}

double IsolatedTauTagInfo::discriminator(double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, 
					 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
					 unsigned int numTracksIsolationRingMax, int& error) const 
{
  return discriminator(matchingConeSize, matchingConeType, ptLeadingTrackMin, ptOtherTracksMin, 1.e6, 
		       signalConeSize, signalConeType, isolationConeSize, isolationConeType, 
		       numTracksIsolationRingMax, error);
}
