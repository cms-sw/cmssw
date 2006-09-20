#ifndef DataFormats_ParticleFlowReco_PFCluster_h
#define DataFormats_ParticleFlowReco_PFCluster_h

#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

#include <iostream>
#include <vector>

namespace reco {

  /**\class PFCluster
     \brief Particle flow cluster, see clustering algorithm in PFClusterAlgo
          
     \author Colin Bernet
     \date   July 2006
  */
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
  
    /// default constructor
    PFCluster();
  
    /// constructor
    PFCluster(unsigned id, int type);

    /// copy constructor
    PFCluster(const PFCluster& other);

    /// destructor
    ~PFCluster();

    /// resets clusters parameters
    void reset();
   
    /// add a given fraction of the rechit
    void addRecHit( const reco::PFRecHit& rechit, double fraction);

    /// updates cluster info from rechit
    void calculatePosition( int algo, double p1 = 0, bool depcor = true);

    /// vector of rechit fractions
    const std::vector< reco::PFRecHitFraction >& recHitFractions() const 
      { return rechits_; }

    /// set cluster id
    void          setId(unsigned id) {id_ = id;} 
  
    /// cluster id
    unsigned      id() const {return id_;}
  
    /// cluster type
    int           type() const {return type_;}

    /// cluster layer, see PFClusterLayer.h
    int           layer() const {return layer_;}          

    /// cluster energy
    double        energy() const {return energy_;}

    /// cluster position: cartesian
    const math::XYZPoint& positionXYZ() const {return posxyz_;}

    /// cluster position: rho, eta, phi
    const REPPoint&       positionREP() const {return posrep_;}

    /// calculates posrep_ once and for all
    void calculatePositionREP() {
      posrep_.SetCoordinates( posxyz_.Rho(), posxyz_.Eta(), posxyz_.Phi() ); 
    }

    /// set parameters for depth correction
    static void setDepthCorParameters( int    mode,
				       double a, 
				       double b, 
				       double ap, 
				       double bp ) {
      depthCorMode_ = mode;
      depthCorA_  = a;
      depthCorB_  = b;
      depthCorAp_ = ap;
      depthCorBp_ = bp;
    }

    static double getDepthCorrection(double energy, bool isBelowPS = false,
				     bool isHadron = false);

    void         setColor(int color) {color_ = color;}

    int          color() const {return color_;}
  
    PFCluster& operator+=(const PFCluster&);
    PFCluster& operator=(const PFCluster&);


    friend    std::ostream& operator<<(std::ostream& out, 
				       const PFCluster& cluster);
    /// counter
    static unsigned     instanceCounter_;

  private:
  
    /// vector of rechit fractions (transient)
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

    /// cluster position: rho, eta, phi (transient)
    REPPoint            posrep_;

    /// keep track of the mode (lin or log E weighting) for position calculation
    int                 posCalcMode_;
  
    /// keep track of the parameter for position calculation
    double              posCalcP1_;

    /// keep track of whether depth correction was required or not
    bool                posCalcDepthCor_;

    /// color (transient)
    int                 color_;

    // the following parameters should maybe be moved to PFClusterAlgo

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
