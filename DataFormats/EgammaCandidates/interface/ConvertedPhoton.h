#ifndef EgammaCandidates_ConvertedPhoton_h
#define EgammaCandidates_ConvertedPhoton_h
/** \class reco::ConvertedPhoton ConvertedPhoton.h DataFormats/EgammaCandidates/interface/ConvertedPhoton.h
 *
 * Reco Candidates with an ConvertedPhoton component
 *
 * \author N.Marinelli  University of Notre Dame, US
 *
 * \version $Id: ConvertedPhoton.h,v 1.1 2006/06/22 12:41:33 nancy Exp $
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
    /// constructor from values
    ConvertedPhoton( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    // 
    ConvertedPhoton( const reco::SuperClusterRef scl,  const reco::TrackRefVector trkRefs , Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ):   RecoCandidate( q, p4, vtx ), superCluster_(scl), tracks_(trkRefs)  { }
    //

    ConvertedPhoton( const reco::SuperClusterRef scl,  Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ):   RecoCandidate( q, p4, vtx ), superCluster_(scl) { }

   
    /// destructor
    virtual ~ConvertedPhoton();
    /// returns a clone of the candidate
    ConvertedPhoton * clone() const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster() const ;
    /// set refrence to ConvertedPhoton component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    // reference to a vector of tracks
    reco::TrackRefVector tracks() const ;  

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    reco::TrackRefVector  tracks_;
  };
  
}

#endif
