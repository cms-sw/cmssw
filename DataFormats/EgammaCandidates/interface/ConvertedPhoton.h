#ifndef EgammaCandidates_ConvertedPhoton_h
#define EgammaCandidates_ConvertedPhoton_h
/** \class reco::ConvertedPhoton ConvertedPhoton.h DataFormats/EgammaCandidates/interface/ConvertedPhoton.h
 *
 * Reco Candidates with an ConvertedPhoton component
 *
 * \author N.Marinelli  University of Notre Dame, US
 *
 * \version $Id: ConvertedPhoton.h,v 1.4 2007/01/26 16:24:39 nancy Exp $
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

   ConvertedPhoton( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ), const Point & convVtx = Point( 0, 0, 0 ) ):  RecoCandidate( q, p4, vtx ), theConversionVertex_(convVtx) { }

    /// destructor
    virtual ~ConvertedPhoton();
    /// returns a clone of the candidate
    ConvertedPhoton * clone() const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster() const ;
    /// set refrence to ConvertedPhoton component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }


    // set reference to a pair of Tracks
    //    void setTrackPairRef( const reco::TrackRefVector & pair ) { tracks_ = pair; }
    void setTrackPairRef( const  std::vector<reco::TrackRef>  & pair ) { tracks_ = pair; }
    // reference to a vector of tracks
    // reco::TrackRefVector tracks() const ;  
    std::vector<reco::TrackRef> tracks() const ; 
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
