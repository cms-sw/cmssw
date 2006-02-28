#ifndef RecoCandidate_RecoMuonCandidate_h
#define RecoCandidate_RecoMuonCandidate_h
// $Id: RecoMuonCandidate.h,v 1.5 2006/02/23 16:52:38 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

namespace reco {

  class RecoMuonCandidate : public RecoChargedCandidate {
  public:
    RecoMuonCandidate() : RecoChargedCandidate() { }
    RecoMuonCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoChargedCandidate( q, p4, vtx ) { }
    virtual ~RecoMuonCandidate();
    virtual RecoMuonCandidate * clone() const;
    void setMuon( const reco::MuonRef & r ) { muon_ = r; }

  private:
    virtual reco::TrackRef track() const;
    virtual reco::MuonRef muon() const;
    reco::MuonRef muon_;
  };
  
}

#endif
