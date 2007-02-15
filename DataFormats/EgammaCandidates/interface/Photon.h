#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Photon.h,v 1.4 2007/01/31 17:11:08 futyand Exp $
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
    Photon( Charge q, const LorentzVector & p4, double r9, double r19, double e5x5, 
	    const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ), r9_(r9), r19_(r19), e5x5_(e5x5) {}
    /// destructor
    virtual ~Photon();
    /// returns a clone of the candidate
    virtual Photon * clone() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// set refrence to Photon component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// set primary event vertex used to define photon direction
    void setVertex(const Point & vertex);
    /// position in ECAL
    math::XYZPoint caloPosition() const;
    /// ratio of E(3x3)/ESC
    double r9() const { return r9_; }
    /// ratio of Emax/E(3x3)
    double r19() const { return r19_; }
    /// 5x5 energy
    double e5x5() const { return e5x5_; }
    /// PDG identifier
    virtual int pdgId() const { return 22; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    double r9_;
    double r19_;
    double e5x5_;
  };
  
}

#endif
