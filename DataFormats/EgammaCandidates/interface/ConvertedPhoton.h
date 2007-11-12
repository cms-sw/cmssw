#ifndef EgammaCandidates_ConvertedPhoton_h
#define EgammaCandidates_ConvertedPhoton_h
/** \class reco::ConvertedPhoton ConvertedPhoton.h DataFormats/EgammaCandidates/interface/ConvertedPhoton.h
 *
 * Reco Candidates with an ConvertedPhoton component
 *
 * \author N.Marinelli  University of Notre Dame, US
 *
 * \version $Id: ConvertedPhoton.h,v 1.5 2007/03/16 13:59:37 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ConvertedPhotonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

namespace reco {
    class ConvertedPhoton : public RecoCandidate {
  public:
    /// default constructor
    ConvertedPhoton() : RecoCandidate() { }

   ConvertedPhoton( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ), const Point & convVtx = Point( 0, 0, 0 ) ):  RecoCandidate( q, p4, vtx, 22 * q ), theConversionVertex_(convVtx) { }

    /// destructor
    virtual ~ConvertedPhoton();
    /// returns a clone of the candidate
    ConvertedPhoton * clone() const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster() const ;
    /// set refrence to ConvertedPhoton component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }


    // set reference to a pair of Tracks
     void setTrackPairRef( const  std::vector<reco::TrackRef>  & pair ) { tracks_ = pair; }
    // reference to a vector of tracks
    const  std::vector<reco::TrackRef>& tracks() const ; 
   /// reference to one of multiple Tracks: implements the method inherited from RecoCandidate
     reco::TrackRef track( size_t ) const;
    // returns the position of the conversion vertex
     const Point & convVertexPosition() const { return theConversionVertex_ ; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
   

    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    //    reco::TrackRefVector  tracks_;
    std::vector<reco::TrackRef>  tracks_;
    reco::Particle::Point theConversionVertex_;

  };
  
}

#endif
