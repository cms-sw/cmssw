#ifndef RecoMuon_GlobalTrackingTools_GlobalMuonTrackMatcher_H
#define RecoMuon_GlobalTrackingTools_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher 
 *
 * Match a standalone muon track with the most compatible tracker tracks
 * in a TrackCollection.  GlobalMuonTrackMatcher is used during global
 * muon reconstruction to check the compatability of a tracker track
 * with a standalone muon track.  The compatability is determined 
 * on a chi2 comparison, of the local parameters of the two corresponding
 * TrajectoryStates on the surface of the innermost muon measurement for
 * momentum below a threshold and above this, with the position and direction 
 * parameters on the mentioned surface. 
 * If the comparison of local parameters fails to yield any
 * matches, then it makes a comparison of the TSOS local direction.
 *
 *
 *  $Date: 2010/05/17 09:44:29 $
 *  $Revision: 1.12 $
 *
 *  \author Edwin Antillon      Purdue University
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
    bool matchTight(const TrackCand& sta, 
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
    double thePt_threshold1;
    double thePt_threshold2;
    double theEta_threshold;
    double theChi2_1;
    double theChi2_2;
    double theChi2_3;
    double theLocChi2;
    double theDeltaD_1;
    double theDeltaD_2;
    double theDeltaD_3;
    double theDeltaR_1;
    double theDeltaR_2;
    double theDeltaR_3;
    double theQual_1;
    double theQual_2;
    double theQual_3;

    const MuonServiceProxy* theService;
    std::string theOutPropagatorName;

};

#endif
