#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Photon.h,v 1.11 2007/03/16 13:59:37 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class Photon : public RecoCandidate {
  public:
    /// default constructor
    Photon() : RecoCandidate() { }
    /// constructor from values
    Photon( Charge q, const LorentzVector & p4, Point unconvPos,
	    double r9, double r19, double e5x5, bool hasPixelSeed=false,
	    const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx, 22 ), unconvPosition_( unconvPos ), 
      r9_( r9 ), r19_( r19 ), e5x5_( e5x5 ), pixelSeed_( hasPixelSeed ) { }
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
    /// position of seed BasicCluster for shower depth of unconverted photon
    math::XYZPoint unconvertedPosition() const { return unconvPosition_; }
    /// ratio of E(3x3)/ESC
    double r9() const { return r9_; }
    /// ratio of Emax/E(3x3)
    double r19() const { return r19_; }
    /// 5x5 energy
    double e5x5() const { return e5x5_; }
    /// Whether or not the SuperCluster has a matched pixel seed
    bool hasPixelSeed() const { return pixelSeed_; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    math::XYZPoint unconvPosition_;
    double r9_;
    double r19_;
    double e5x5_;
    bool pixelSeed_;
  };
  
}

#endif
