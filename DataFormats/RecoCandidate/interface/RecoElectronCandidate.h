#ifndef RecoCandidate_RecoElectronCandidate_h
#define RecoCandidate_RecoElectronCandidate_h
// $Id: RecoElectronCandidate.h,v 1.4 2006/02/21 10:37:35 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoElectronCandidate : public RecoCandidate {
  public:
    RecoElectronCandidate() : RecoCandidate() { }
    RecoElectronCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    virtual ~RecoElectronCandidate();
    virtual RecoElectronCandidate * clone() const;
    void setElectron( const reco::ElectronRef & r ) { electron_ = r; }

  private:
    virtual reco::ElectronRef electron() const;
    virtual reco::TrackRef track() const;
    virtual reco::SuperClusterRef superCluster() const;
    reco::ElectronRef electron_;
  };
  
}

#endif
