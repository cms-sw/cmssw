#ifndef RecoCandidate_RecoCandidate_h
#define RecoCandidate_RecoCandidate_h
/** \class reco::RecoCandidate
 *
 * base class for all Reco Candidates
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoCandidate.h,v 1.4 2006/03/08 12:57:08 llista Exp $
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
    /// reference to a stand-alone muon Track
    virtual reco::TrackRef standAloneMuon() const;
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

  /// stand alone muon component tag
  struct StandAloneMuonTag {
  };

  /// get default Track component
  GET_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, DefaultComponentTag, track );
  /// get default Muon component
  GET_CANDIDATE_COMPONENT( RecoCandidate, MuonRef, DefaultComponentTag, muon );
  /// get stand-alone muon Track component
  GET_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, StandAloneMuonTag, standAloneMuon );
  /// get default Electron component
  GET_CANDIDATE_COMPONENT( RecoCandidate, ElectronRef, DefaultComponentTag, electron );
  /// get default SuperCluster component
  GET_CANDIDATE_COMPONENT( RecoCandidate, SuperClusterRef, DefaultComponentTag, superCluster );
  /// get default Photon component
  GET_CANDIDATE_COMPONENT( RecoCandidate, PhotonRef, DefaultComponentTag, photon );
  /// get default CaloTower component
  GET_CANDIDATE_COMPONENT( RecoCandidate, RecoCandidate::CaloTowerRef, DefaultComponentTag, caloTower );
  
}

#endif
