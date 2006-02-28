#ifndef RecoCandidate_RecoChargedCandidate_h
#define RecoCandidate_RecoChargedCandidate_h
// $Id: RecoChargedCandidate.h,v 1.4 2006/02/21 10:37:35 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoChargedCandidate : public RecoCandidate {
  public:
    RecoChargedCandidate() : RecoCandidate() { }
    RecoChargedCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) :
      RecoCandidate( q, p4, vtx ) { }
    virtual ~RecoChargedCandidate();
    virtual RecoChargedCandidate * clone() const;
    void setTrack( const reco::TrackRef & r ) { track_ = r; }

  private:
    virtual reco::TrackRef track() const;
    reco::TrackRef track_;
  };
  
}

#endif
