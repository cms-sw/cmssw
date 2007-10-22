#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Photon.h,v 1.13 2007/10/06 20:05:46 futyand Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h" 

namespace reco {

  class Photon : public RecoCandidate {
  public:
    /// default constructor
    Photon() : RecoCandidate() { }
    /// constructor from values
    Photon( Charge q, const LorentzVector & p4, Point unconvPos,
	    const SuperClusterRef scl, const ClusterShapeRef shp,
	    bool hasPixelSeed=false, const Point & vtx = Point( 0, 0, 0 ) );
    /// destructor
    virtual ~Photon();
    /// returns a clone of the candidate
    virtual Photon * clone() const;
    /// reference to SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// reference to ClusterShape for seed BasicCluster of SuperCluster
    virtual reco::ClusterShapeRef seedClusterShape() const;
    /// set reference to SuperCluster
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// set reference to ClusterShape
    void setClusterShapeRef( const reco::ClusterShapeRef & r ) { seedClusterShape_ = r; }
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
    /// position of seed BasicCluster for shower depth of unconverted photon
    math::XYZPoint unconvPosition_;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    /// reference to ClusterShape for seed BasicCluster of SuperCluster
    reco::ClusterShapeRef seedClusterShape_;
    double r9_;
    double r19_;
    double e5x5_;
    bool pixelSeed_;
  };
  
}

#endif
