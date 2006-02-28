#ifndef RecoCandidate_RecoCandidate_h
#define RecoCandidate_RecoCandidate_h
// $Id: RecoCandidate.h,v 1.9 2006/02/21 10:37:35 llista Exp $
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EGammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EGammaReco/interface/ElectronFwd.h"
#include "DataFormats/EGammaReco/interface/PhotonFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

namespace reco {

  class RecoCandidate : public LeafCandidate {
  public:
    RecoCandidate() : LeafCandidate() { }
    RecoCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      LeafCandidate( q, p4, vtx ) { }
    virtual ~RecoCandidate();

    typedef edm::Ref<CaloTowerCollection> CaloTowerRef;

  private:
    virtual reco::TrackRef track() const;
    virtual reco::MuonRef muon() const;
    virtual reco::SuperClusterRef superCluster() const;
    virtual reco::ElectronRef electron() const;
    virtual reco::PhotonRef photon() const;
    virtual CaloTowerRef caloTower() const;
    template<typename T> friend struct component; 
    bool overlap( const Candidate & ) const;
  };

  template<>
  struct component<reco::Track> {
    typedef reco::TrackRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->track();
    }
  };
    
  template<>
  struct  component<reco::Muon> {
    typedef reco::MuonRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->muon();
    }
  };
  
  template<>
  struct  component<reco::Electron> {
    typedef reco::ElectronRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->electron();
    }
  };

  template<>
  struct  component<reco::SuperCluster> {
    typedef reco::SuperClusterRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->superCluster();
    }
  };

  template<>
  struct  component<reco::Photon> {
    typedef reco::PhotonRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->photon();
    }
  };
  
  template<>
  struct  component<CaloTower> {
    typedef RecoCandidate::CaloTowerRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->caloTower();
    }
  };
  
}

#endif
