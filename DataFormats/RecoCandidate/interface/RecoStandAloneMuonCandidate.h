#ifndef RecoCandidate_RecoStandAloneMuonCandidate_h
#define RecoCandidate_RecoStandAloneMuonCandidate_h
/** \class reco::RecoStandAloneMuonCandidate
 *
 * Reco Candidates with a Track component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoStandAloneMuonCandidate.h,v 1.1 2007/12/14 12:38:59 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoStandAloneMuonCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoStandAloneMuonCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoStandAloneMuonCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			  int pdgId = 0, int status = 0 ) :
      RecoCandidate( q, p4, vtx, pdgId, status ) { }
    /// constructor from values
    RecoStandAloneMuonCandidate( Charge q , const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			  int pdgId = 0, int status = 0 ) :
      RecoCandidate( q, p4, vtx, pdgId, status ) { }
    /// destructor
    virtual ~RecoStandAloneMuonCandidate();
    /// returns a clone of the candidate
    virtual RecoStandAloneMuonCandidate * clone() const;
    /// set reference to track
    void setTrack( const reco::TrackRef & r ) { standAloneMuonTrack_ = r; }
    /// reference to a track
    virtual reco::TrackRef standAloneMuon() const;

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a track
    reco::TrackRef standAloneMuonTrack_;
  };
  
}

#endif
