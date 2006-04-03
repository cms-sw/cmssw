#ifndef RecoCandidate_RecoMuonCandidate_h
#define RecoCandidate_RecoMuonCandidate_h
/** \class reco::RecoMuonCandidate
 *
 * Reco Candidates with a Muon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoMuonCandidate.h,v 1.2 2006/03/01 16:31:47 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoMuonCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoMuonCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoMuonCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~RecoMuonCandidate();
    /// returns a clone of the candidate
    virtual RecoMuonCandidate * clone() const;
    /// set the muon component
    void setMuon( const reco::MuonRef & r ) { muon_ = r; }

  private:
    /// referente to a track
    virtual reco::TrackRef track() const;
    /// reference to a muon
    virtual reco::MuonRef muon() const;
    /// reference to a muon
    /// reference to a stand-alone muon Track
    virtual reco::TrackRef standAloneMuon() const;
    reco::MuonRef muon_;
  };
  
}

#endif
