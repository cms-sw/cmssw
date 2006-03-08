#ifndef RecoCandidate_RecoCandidate_h
#define RecoCandidate_RecoCandidate_h
/** \class reco::RecoCandidate
 *
 * base class for all Reco Candidates
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoCandidate.h,v 1.2 2006/03/01 16:31:47 llista Exp $
 *
 */
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
    /// default constructor
    RecoCandidate() : LeafCandidate() { }
    /// constructor from values
    RecoCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      LeafCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~RecoCandidate();
    /// reference to CaloTower. 
    /// Should better be defined in CaloTower package
    /// defined here for the moment
    typedef edm::Ref<CaloTowerCollection> CaloTowerRef;

  private:
    /// reference to a Track
    virtual reco::TrackRef track() const;
    /// reference to a Muon
    virtual reco::MuonRef muon() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// reference to an Electron
    virtual reco::ElectronRef electron() const;
    /// reference to a Photon
    virtual reco::PhotonRef photon() const;
    /// reference to a CaloTowe
    virtual CaloTowerRef caloTower() const;
    template<typename T> friend struct component; 
    /// check overlap with another candidate
    bool overlap( const Candidate & ) const;
  };

  /// get Track component 
  template<>
  struct component<reco::Track> {
    typedef reco::TrackRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->track();
    }
  };
    
  /// get Muon component 
  template<>
  struct  component<reco::Muon> {
    typedef reco::MuonRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->muon();
    }
  };
  
  /// get Electron component 
  template<>
  struct  component<reco::Electron> {
    typedef reco::ElectronRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->electron();
    }
  };

  /// get SuperCluster component 
  template<>
  struct  component<reco::SuperCluster> {
    typedef reco::SuperClusterRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->superCluster();
    }
  };

  /// get Photon component 
  template<>
  struct  component<reco::Photon> {
    typedef reco::PhotonRef Ref;
    static Ref get( const Candidate & c ) {
      const RecoCandidate * dc = dynamic_cast<const RecoCandidate *>( & c );
      if ( dc == 0 ) return Ref();
      return dc->photon();
    }
  };
  
  /// get CaloTower component 
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
