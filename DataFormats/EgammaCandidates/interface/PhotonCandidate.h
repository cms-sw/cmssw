#ifndef EgammaCandidates_PhotonCandidate_h
#define EgammaCandidates_PhotonCandidate_h
/** \class reco::PhotonCandidate PhotonCandidate.h DataFormats/EgammaCandidates/interface/PhotonCandidate.h
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: PhotonCandidate.h,v 1.2 2006/04/26 07:56:19 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCandidateFwd.h"

namespace reco {

  class PhotonCandidate : public RecoCandidate {
  public:
    /// default constructor
    PhotonCandidate() : RecoCandidate() { }
    /// constructor from values
    PhotonCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~PhotonCandidate();
    /// returns a clone of the candidate
    virtual PhotonCandidate * clone() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// set refrence to Photon component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
  };
  
}

#endif
