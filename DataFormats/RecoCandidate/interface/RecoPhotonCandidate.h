#ifndef RecoCandidate_RecoPhotonCandidate_h
#define RecoCandidate_RecoPhotonCandidate_h
// $Id: RecoPhotonCandidate.h,v 1.4 2006/02/21 10:37:35 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoPhotonCandidate : public RecoCandidate {
  public:
    RecoPhotonCandidate() : RecoCandidate() { }
    RecoPhotonCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    virtual ~RecoPhotonCandidate();
    virtual RecoPhotonCandidate * clone() const;
    void setPhoton( const reco::PhotonRef & r ) { photon_ = r; }

  private:
    virtual reco::PhotonRef photon() const;
    virtual reco::SuperClusterRef superCluster() const;
    reco::PhotonRef photon_;
  };
  
}

#endif
