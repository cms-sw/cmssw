#ifndef RecoCandidate_RecoPhotonCandidate_h
#define RecoCandidate_RecoPhotonCandidate_h
/** \class reco::RecoPhotonCandidate
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoPhotonCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoPhotonCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoPhotonCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~RecoPhotonCandidate();
    /// returns a clone of the candidate
    virtual RecoPhotonCandidate * clone() const;
    /// set refrenec to Photon component
    void setPhoton( const reco::PhotonRef & r ) { photon_ = r; }

  private:
    /// refrence to a Photon
    virtual reco::PhotonRef photon() const;
    /// reference to a SuperCluste
    virtual reco::SuperClusterRef superCluster() const;
    /// reference to a Photon
    reco::PhotonRef photon_;
  };
  
}

#endif
