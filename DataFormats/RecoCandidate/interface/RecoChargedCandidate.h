#ifndef RecoCandidate_RecoChargedCandidate_h
#define RecoCandidate_RecoChargedCandidate_h
/** \class reco::RecoChargedCandidate
 *
 * Reco Candidates with a Track component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoChargedCandidate.h,v 1.8 2007/10/22 09:38:13 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoChargedCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoChargedCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoChargedCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			  int pdgId = 0, int status = 0 ) :
      RecoCandidate( q, p4, vtx, pdgId, status ) { }
    /// constructor from values
    RecoChargedCandidate( Charge q , const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			  int pdgId = 0, int status = 0 ) :
      RecoCandidate( q, p4, vtx, pdgId, status ) { }
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
