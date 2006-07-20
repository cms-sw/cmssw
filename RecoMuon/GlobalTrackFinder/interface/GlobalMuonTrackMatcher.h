#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2006/07/16 01:48:54 $
 *  $Revision: 1.4 $
 *  \author Chang Liu  - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Handle.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

class TrajectoryStateOnSurface;
class MuonUpdatorAtVertex;
class MagneticField;
//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonTrackMatcher {
public:
  /// constructor
  GlobalMuonTrackMatcher(double chi2, const MagneticField*);

  /// destructor
  virtual ~GlobalMuonTrackMatcher() {};

  /// choose one that with smallest chi2
  std::pair<bool, reco::TrackRef> matchOne(const reco::TrackRef&, const edm::Handle<reco::TrackCollection>&) const;

  /// choose all that has chi2 less than MaxChi2
  std::vector<reco::TrackRef> match(const reco::TrackRef&, const edm::Handle<reco::TrackCollection>&) const;

  /// choose all that has chi2 less than MaxChi2
  std::vector<reco::TrackRef> match(const reco::TrackRef&, const std::vector<reco::TrackRef>&) const;

  /// check if two tracks are match
  std::pair<bool,double> match(const reco::Track&, const reco::Track&) const; 

  /// check if two TSOS are match
  std::pair<bool,double> match(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;

private:
  double theMaxChi2;
  double theMinP;
  double theMinPt;
  GlobalPoint theVertexPos;
  GlobalError theVertexErr;
  MuonUpdatorAtVertex* theUpdator;
  const MagneticField* theField;

};

#endif
