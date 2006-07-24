#ifndef DataFormats_ParticleFlowReco_PFCluster_h
#define DataFormats_ParticleFlowReco_PFCluster_h

#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

#include <iostream>
#include <vector>

namespace reco {

  class PFCluster {
  public:

    /// type definition
    enum Type {
      TYPE_TOPOLOGICAL = 1, 
      TYPE_PF = 2 
    };

    /// energy weighting for position calculation
    enum PosCalc {
      POSCALC_LIN,
      POSCALC_LOG
    };

    typedef ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > REPPoint;
  
    PFCluster();
  
    PFCluster(unsigned id, int type);

    PFCluster(const PFCluster& other);
   
    /// add a given fraction of the rechit
    void AddRecHit( const reco::PFRecHit& rechit, double fraction);

    /// updates cluster info from rechit
    void CalculatePosition( int algo, double p1 = 0, bool depcor = true);

    /// vector of rechit fractions
    const std::vector< reco::PFRecHitFraction >& GetRecHitFractions() const 
      { return rechits_; }
  
    /// cluster id
    unsigned      GetId() const {return id_;}
  
    /// cluster type
    int           GetType() const {return type_;}

    /// cluster layer, see PFClusterLayer.h
    int           GetLayer() const {return layer_;}          

    /// cluster energy
    double        GetEnergy() const {return energy_;}

    /// cluster position: cartesian
    const math::XYZPoint& GetPositionXYZ() const {return posxyz_;}

    /// cluster position: rho, eta, phi
    const REPPoint&       GetPositionREP() const {return posrep_;}

    /// set parameters for depth correction
    static void SetDepthCorParameters( int    mode,
				       double a, 
				       double b, 
				       double ap, 
				       double bp ) 
      {
	depthCorMode_ = mode;
	depthCorA_  = a;
	depthCorB_  = b;
	depthCorAp_ = ap;
	depthCorBp_ = bp;
      }

  
    friend    std::ostream& operator<<(std::ostream& out, 
				       const PFCluster& cluster);

  private:
  
    /// vector of rechit fractions
    std::vector< reco::PFRecHitFraction >  rechits_;

    /// cluster id
    unsigned      id_;

    /// cluster type
    int           type_;

    /// cluster layer, see PFClusterLayer.h
    int           layer_;          

    /// cluster energy
    double        energy_;

    /// cluster position: cartesian
    math::XYZPoint      posxyz_;

    /// cluster position: rho, eta, phi
    REPPoint            posrep_;

    /// keep track of the mode (lin or log E weighting) for position calculation
    int                 posCalcMode_;
  
    /// keep track of the parameter for position calculation
    double              posCalcP1_;

    /// keep track of whether depth correction was required or not
    bool                posCalcDepthCor_;

    /// mode for depth correction (e/gamma or hadron)
    static int          depthCorMode_;

    /// A parameter for depth correction
    static double       depthCorA_;

    /// B parameter for depth correction
    static double       depthCorB_;

    /// A parameter for depth correction (under preshower)
    static double       depthCorAp_;

    /// B parameter for depth correction (under preshower)
    static double       depthCorBp_;
  };

}

#endif
