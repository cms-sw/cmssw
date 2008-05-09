#ifndef RecoMuon_GlobalTrackingTools_GlobalMuonTrackMatcher_H
#define RecoMuon_GlobalTrackingTools_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher 
 *
 * Match a standalone muon track with the most compatible tracker tracks
 * in a TrackCollection.  GlobalMuonTrackMatcher is used during global
 * muon reconstruction to check the compatability of a tracker track
 * with a standalone muon track.  The compatability is determined based
 * on a chi2 comparison of the local parameters of the two corresponding
 * TrajectoryStates on the surface of the innermost muon measurement.
 * If the comparison of local parameters fails to yield any
 * matches, then it makes a comparison of the TSOS global positon.  If
 * there is still no match, then it matches the standalone muon with the
 * tracker track that is closest in eta-phi space.
 *
 *
 *  $Date: 2008/03/21 01:16:23 $
 *  $Revision: 1.6 $
 *
 *  \author Chang Liu           Purdue University
 *  \author Adam Everett        Purdue University
 *  \author Norbert Neumeister  Purdue University
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"

class TrajectoryStateOnSurface;
class MuonServiceProxy;
class Trajectory;

namespace edm {class ParameterSet;}

//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonTrackMatcher {

  public:

    typedef std::pair<const Trajectory*,reco::TrackRef> TrackCand;

    /// constructor
    GlobalMuonTrackMatcher(const edm::ParameterSet&,
                           const MuonServiceProxy*);

    /// destructor
    virtual ~GlobalMuonTrackMatcher();
    
    /// check if two tracks are compatible (less than Chi2Cut, DeltaDCut, DeltaRCut)
    bool match(const TrackCand& sta, 
               const TrackCand& track) const;
    
    /// check if two tracks are compatible
    /// matchOption: 0 = chi2, 1 = distance, 2 = deltaR
    /// surfaceOption: 0 = outermost tracker surface, 1 = innermost muon system surface 
    double match(const TrackCand& sta, 
                 const TrackCand& track,
                 int matchOption = 0,
                 int surfaceOption = 1) const;

    /// choose all tracks with a matching-chi2 less than Chi2Cut
    std::vector<TrackCand> match(const TrackCand& sta, 
                                 const std::vector<TrackCand>& tracks) const;

    /// choose the one tracker track which best matches a muon track
    std::vector<TrackCand>::const_iterator matchOne(const TrackCand& sta,
                                                    const std::vector<TrackCand>& tracks) const;

  private:

    double match_R_IP(const TrackCand&, const TrackCand&) const;
    double match_D(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
    double match_d(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
    double match_Rmom(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
    double match_Rpos(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
    double match_Chi2(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
    double match_dist(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;

    std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface> convertToTSOSTk(const TrackCand&, const TrackCand&) const;
    std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface> convertToTSOSMuHit(const TrackCand&, const TrackCand&) const;
    std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface> convertToTSOSTkHit(const TrackCand&, const TrackCand&) const;
    bool samePlane(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;

  private:
    
    double theMinP;
    double theMinPt;
    double theMaxChi2;
    double theDeltaD;
    double theDeltaR;

    const MuonServiceProxy* theService;
    std::string theOutPropagatorName;

};

#endif
