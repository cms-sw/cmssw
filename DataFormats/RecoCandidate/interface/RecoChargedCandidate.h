#ifndef RecoCandidate_RecoChargedCandidate_h
#define RecoCandidate_RecoChargedCandidate_h
/** \class reco::RecoChargedCandidate
 *
 * Reco Candidates with a Track component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoChargedCandidate.h,v 1.4 2006/08/03 14:54:34 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

namespace reco {

  class RecoChargedCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoChargedCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoChargedCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) :
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~RecoChargedCandidate();
    /// returns a clone of the candidate
    virtual RecoChargedCandidate * clone() const;
    /// set reference to track
    void setTrack( const reco::TrackRef & r ) { track_ = r; }
    /// reference to a track
    virtual reco::TrackRef track() const;

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a track
    reco::TrackRef track_;
  };
  
}

#endif
