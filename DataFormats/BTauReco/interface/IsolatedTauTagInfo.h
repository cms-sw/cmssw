#ifndef DataFormats_BTauReco_IsolatedTauTagInfo_h
#define DataFormats_BTauReco_IsolatedTauTagInfo_h
//
// \class IsolatedTauTagInfo
// \short Extended object for the Tau Isolation algorithm.
// contains the result and the methods used in the ConeIsolation Algorithm, to create the 
// object to be made persistent on RECO
//
// \author: Simone Gennai, based on ORCA class by S. Gennai and F. Moortgat;
//          extended to different cone types and energy dependent cone sizes by Christian Veelken, UC Davis, on 05/18/2007
//

// CMSSW include files
#include "PhysicsTools/IsolationUtils/interface/TauConeIsolationAlgo.h"

#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "PhysicsTools/Utilities/interface/Angle.h"

#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco { 
  class IsolatedTauTagInfo : public JTATagInfo 
  {
   public:

    enum generalConeTypes { kEtaPhiCone, kOpeningAngleCone, kFixedAreaIsolationCone }; // "kFixedAreaIsolationCone" valid for isolation cone only

    // default constructor
    IsolatedTauTagInfo() {}
    
    IsolatedTauTagInfo(reco::TrackRefVector tracks, const reco::JetTracksAssociationRef & jtaRef)
      : JTATagInfo(jtaRef) 
    {    
      for ( reco::TrackRefVector::iterator track = tracks.begin();
	    track != tracks.end(); ++track ) {
	selectedTracks_.push_back(*track);
      }
    }
    
    virtual IsolatedTauTagInfo* clone() const { return new IsolatedTauTagInfo( *this ); }
    
    // destructor
    virtual ~IsolatedTauTagInfo() {}
    
    // get all tracks from the jetTag (without any track selection applied)
    const reco::TrackRefVector allTracks() const { return tracks(); }
    
    // get list of selected tracks passed to IsolatedTauTagInfo constructor
    const reco::TrackRefVector& selectedTracks() const { return selectedTracks_; }
    
    // get leading (i.e. highest Pt track)
    const reco::TrackRef leadingSignalTrack(double matchingConeSize, int matchingConeType, double ptTrackMin, int& error) const;
    const reco::TrackRef leadingSignalTrack(const math::XYZVector& jetAxis, double matchingConeSize, int matchingConeType, double ptTrackMin, int& error) const;
  
    // get list of selected tracks within cone given as function argument;
    // tracks within cone are required to originate from the same primary event vertex as the leading track
    const reco::TrackRefVector tracksInCone(const math::XYZVector& coneAxis, double coneSize, int coneType, double ptTrackMin, 
					    double zPrimaryVertex, double dzTrackMax, int& error) const;
    // no requirement on origin of tracks 
    const reco::TrackRefVector tracksInCone(const math::XYZVector& coneAxis, double coneSize, int coneType, double ptTrackMin, int& error) const;
    
    // functions that allow to recompute discriminator value "on-the-fly"
    //
    //  return value: 0 = jet fails tau-jet selection
    //                1 = jet passes tau-jet selection
    //
    // axes of signal and isolation cone given as function argument,
    // tracks within signal and isolation cones are required to originate from the same primary event vertex as the leading track
    double discriminator(const math::XYZVector& coneAxes, 
			 double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, double dzOtherTrackMax, 
			 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
			 unsigned int numTracksIsolationRingMax, int& error) const;
    // axes of signal and isolation cone given as function argument,
    // no requirement on origin of tracks within signal and isolation cones
    double discriminator(const math::XYZVector& coneAxes, 
			 double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, 
			 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
			 unsigned int numTracksIsolationRingMax, int& error) const;
    // axes of signal and isolation cone taken as jet-axis;
    // tracks within signal and isolation cones are required to originate from the same primary event vertex as the leading track
    double discriminator(double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, double dzOtherTrackMax, 
			 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
			 unsigned int numTracksIsolationRingMax, int& error) const;
    // axes of signal and isolation cone taken as jet-axis;
    // no requirement on origin of tracks within signal and isolation cones
    double discriminator(double matchingConeSize, int matchingConeType, double ptLeadingTrackMin, double ptOtherTracksMin, 
			 double signalConeSize, int signalConeType, double isolationConeSize, int isolationConeType, 
			 unsigned int numTracksIsolationRingMax, int& error) const;

    // dummy implementation for backwards compatibility
    // (to be removed in the future...)
    double discriminator() const { return -1; }
    
  private:
//--- list of tracks within jet
//    (track selection criteria applied)
    reco::TrackRefVector selectedTracks_;

//--- template objects 
//    for eta-phi and three-dimensional angle metrics
    TauConeIsolationAlgo<math::XYZVector, reco::TrackCollection, DeltaR<math::XYZVector> > coneIsolationAlgorithmEtaPhi_;
    DeltaR<math::XYZVector> metricEtaPhi_;
  
    TauConeIsolationAlgo<math::XYZVector, reco::TrackCollection, Angle<math::XYZVector> > coneIsolationAlgorithmAngle_; 
    Angle<math::XYZVector> metricAngle_;
  };
}

#endif
