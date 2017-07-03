#ifndef RecoCandidate_RecoChargedCandidate_h
#define RecoCandidate_RecoChargedCandidate_h
/** \class reco::RecoChargedCandidate
 *
 * Reco Candidates with a Track component
 *
 * \author Luca Lista, INFN
 *
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
    ~RecoChargedCandidate() override;
    /// returns a clone of the candidate
    RecoChargedCandidate * clone() const override;
    /// set reference to track
    void setTrack( const reco::TrackRef & r ) { track_ = r; }
    /// reference to a track
    reco::TrackRef track() const override;

  private:
    /// check overlap with another candidate
    bool overlap( const Candidate & ) const override;
    /// reference to a track
    reco::TrackRef track_;
  };
  
}

#endif
