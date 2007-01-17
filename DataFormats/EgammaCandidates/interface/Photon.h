#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Photon.h,v 1.2 2006/06/16 15:01:16 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

namespace reco {

  class Photon : public RecoCandidate {
  public:
    /// default constructor
    Photon() : RecoCandidate() { }
    /// constructor from values
    Photon( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~Photon();
    /// returns a clone of the candidate
    virtual Photon * clone() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// set refrence to Photon component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// PDG identifier
    virtual int pdgId() const { return 22; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
  };
  
}

#endif
