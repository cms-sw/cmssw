#ifndef RecoCandidate_RecoCandidate_h
#define RecoCandidate_RecoCandidate_h
/** \class reco::RecoCandidate
 *
 * base class for all Reco Candidates
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

namespace reco {

  class RecoCandidate : public LeafCandidate {
  public:
    /// default constructor
    RecoCandidate() : LeafCandidate() { }
    /// constructor from values
    template<typename P4>
    RecoCandidate( Charge q, const P4 & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0 ) : 
      LeafCandidate( q, p4, vtx, pdgId, status ) { }
    /// destructor
    virtual ~RecoCandidate();
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const = 0;
    /// returns a clone of the Candidate object                                           
    virtual RecoCandidate * clone() const ;

    /// reference to a Track
    virtual reco::TrackRef track() const;
    /// reference to one of multiple Tracks
    virtual reco::TrackRef track( size_t ) const;
    /// number of multiple Tracks
    virtual size_t numberOfTracks() const;
    /// reference to a GsfTrack
    virtual reco::GsfTrackRef gsfTrack() const;
    /// reference to a stand-alone muon Track
    virtual reco::TrackRef standAloneMuon() const;
    /// reference to a stand-alone muon Track
    virtual reco::TrackRef combinedMuon() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// reference to a CaloTower
    virtual CaloTowerRef caloTower() const;
    /// best track pointer
    virtual const Track * bestTrack() const;
    /// best track RefToBase
    virtual TrackBaseRef bestTrackRef() const;
    /// track type
    enum TrackType { noTrackType, recoTrackType, gsfTrackType };
    ///track type
    virtual TrackType bestTrackType() const;
    /// uncertainty on dz 
    virtual float dzError() const; 
    /// uncertainty on dxy
    virtual float dxyError() const; 


  protected:
    /// check if two components overlap
    template<typename R>
    bool checkOverlap( const R & r1, const R & r2 ) const {
      return( ! r1.isNull() && ! r2.isNull() && r1 == r2 );
    }

  private:
    template<typename, typename, typename> friend struct component; 
  };

  /// stand alone muon component tag
  struct StandAloneMuonTag { };
  /// conbined muon component tag
  struct CombinedMuonTag { };

  /// get default Track component
  GET_DEFAULT_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, track );
  /// get multuple tracks
  GET_DEFAULT_CANDIDATE_MULTIPLECOMPONENTS( RecoCandidate, TrackRef, track, numberOfTracks );
  /// get default GsfTrack component
  GET_DEFAULT_CANDIDATE_COMPONENT( RecoCandidate, GsfTrackRef, gsfTrack );
  /// get stand-alone muon Track component
  GET_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, standAloneMuon, StandAloneMuonTag );
  /// get combined muon Track component
  GET_CANDIDATE_COMPONENT( RecoCandidate, TrackRef, combinedMuon, CombinedMuonTag );
  /// get default SuperCluster component
  GET_DEFAULT_CANDIDATE_COMPONENT( RecoCandidate, SuperClusterRef, superCluster );
  /// get default CaloTower component
  GET_DEFAULT_CANDIDATE_COMPONENT( RecoCandidate, CaloTowerRef, caloTower );
  /// get default CaloTower component
  GET_DEFAULT_CANDIDATE_COMPONENT( RecoCandidate, const Track *, bestTrack );
  /// get default CaloTower component
  GET_DEFAULT_CANDIDATE_COMPONENT( RecoCandidate, RecoCandidate::TrackType, bestTrackType );
  
}

#endif
