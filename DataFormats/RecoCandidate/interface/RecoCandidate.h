#ifndef RecoCandidate_RecoCandidate_h
#define RecoCandidate_RecoCandidate_h
/** \class reco::RecoCandidate
 *
 * base class for all Reco Candidates
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoCandidate.h,v 1.10 2006/05/02 09:48:46 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

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
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const = 0;
    /// reference to a Track
    virtual reco::TrackRef track() const;
    /// reference to a stand-alone muon Track
    virtual reco::TrackRef standAloneMuon() const;
    /// reference to a stand-alone muon Track
    virtual reco::TrackRef combinedMuon() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// reference to a CaloTower
    virtual CaloTowerRef caloTower() const;

  protected:
    /// check if two components overlap
    template<typename R>
    bool checkOverlap( const R & r1, const R & r2 ) const {
      return( ! r1.isNull() && ! r2.isNull() && r1 == r2 );
    }

  private:
    template<typename T> friend struct component; 
  };

  /// stand alone muon component tag
  struct StandAloneMuonTag {
  };

  struct CombinedMuonTag {
  };

  /// get default Track component
  GET_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, DefaultComponentTag, track );
  /// get stand-alone muon Track component
  GET_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, StandAloneMuonTag, standAloneMuon );
  /// get combined muon Track component
  GET_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, CombinedMuonTag, combinedMuon );
  /// get default SuperCluster component
  GET_CANDIDATE_COMPONENT( RecoCandidate, SuperClusterRef, DefaultComponentTag, superCluster );
  /// get default CaloTower component
  GET_CANDIDATE_COMPONENT( RecoCandidate, CaloTowerRef, DefaultComponentTag, caloTower );
  
}

#endif
