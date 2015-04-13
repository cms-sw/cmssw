#ifndef ParticleFlowReco_PFRecHit_h
#define ParticleFlowReco_PFRecHit_h
/** 
 */
#include <vector>
#include <map>
#include <iostream>

#include "DataFormats/Math/interface/Point3D.h"
#include "Rtypes.h" 
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

//C decide what is the default rechit index. 
//C maybe 0 ? -> compression 
//C then the position is index-1. 
//C provide a helper class to access the rechit. 

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/Common/interface/RefToBase.h"


namespace reco {

  /**\class PFRecHit
     \brief Particle flow rechit (rechit + geometry and topology information). See clustering algorithm in PFClusterAlgo
          
     \author Colin Bernet
     \date   July 2006

     Feb 2014 [Michalis: 8 years later!Modifying the class to be able to generalize the neighbours for 3D calorimeters ]
  */
  class PFRecHit {

  public:

    // Next typedef uses double in ROOT 6 rather than Double32_t due to a bug in ROOT 5,
    // which otherwise would make ROOT5 files unreadable in ROOT6.  This does not increase
    // the size on disk, because due to the bug, double was actually stored on disk in ROOT 5.
    typedef ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> > REPPoint;

    typedef std::vector<REPPoint> REPPointVector;

    enum {
      NONE=0
    };
    /// default constructor. Sets energy and position to zero
    PFRecHit();

    /// constructor from values
    PFRecHit(unsigned detId,
             PFLayer::Layer layer,
             double energy, 
             const math::XYZPoint& posxyz, 
             const math::XYZVector& axisxyz, 
             const std::vector< math::XYZPoint >& cornersxyz);

    PFRecHit(unsigned detId,
             PFLayer::Layer layer,
             double energy, 
             double posx, double posy, double posz, 
             double axisx, double axisy, double axisz);    

    /// copy
    PFRecHit(const PFRecHit& other);

    /// destructor
    virtual ~PFRecHit();

    void setEnergy( double energy) { energy_ = energy; }

    void calculatePositionREP();

    void addNeighbour(short x,short y, short z,const PFRecHitRef&);
    const PFRecHitRef getNeighbour(short x,short y, short z);
    void setTime( double time) { time_ = time; }
    void setDepth( int depth) { depth_ = depth; }
    void clearNeighbours() {
      neighbours_.clear();
    }

    const PFRecHitRefVector& neighbours4() const {
      return neighbours4_;
    }
    const PFRecHitRefVector& neighbours8() const {
      return neighbours8_;
    }

    const PFRecHitRefVector& neighbours() const {
      return neighbours_;
    }

    const std::vector<unsigned short>& neighbourInfos() {
      return neighbourInfos_;
    }


    void      setNWCorner( double posx, double posy, double posz );
    void      setSWCorner( double posx, double posy, double posz );
    void      setSECorner( double posx, double posy, double posz );
    void      setNECorner( double posx, double posy, double posz );

    /// rechit detId
    unsigned detId() const {return detId_;}

    /// rechit layer
    PFLayer::Layer layer() const { return layer_; }

    /// rechit energy
    double energy() const { return energy_; }


    /// timing for cleaned hits
    double time() const { return time_; }

    /// depth for segemntation
    int  depth() const { return depth_; }

    /// rechit momentum transverse to the beam, squared.
    double pt2() const { return energy_ * energy_ *
			   ( position_.X()*position_.X() + 
			     position_.Y()*position_.Y() ) / 
			   ( position_.X()*position_.X() +
			     position_.Y()*position_.Y() + 
			     position_.Z()*position_.Z()) ; }


    /// rechit cell centre x, y, z
    const math::XYZPoint& position() const { return position_; }

    const REPPoint& positionREP() const { return positionrep_; }


    /// rechit cell axis x, y, z
    const math::XYZVector& getAxisXYZ() const { return axisxyz_; }    

    /// rechit corners
    const std::vector< math::XYZPoint >& getCornersXYZ() const 
      { return cornersxyz_; }    

    const std::vector<REPPoint>& getCornersREP() const { return cornersrep_; }

    void size(double& deta, double& dphi) const;

    /// comparison >= operator
    bool operator>=(const PFRecHit& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator> (const PFRecHit& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator<=(const PFRecHit& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison < operator
    bool operator< (const PFRecHit& rhs) const { return (energy_< rhs.energy_); }

    friend std::ostream& operator<<(std::ostream& out, 
                                    const reco::PFRecHit& hit);

    const edm::RefToBase<CaloRecHit>& originalRecHit() const {
      return originalRecHit_;
    }

    template<typename T> 
    void setOriginalRecHit(const T& rh) {
      originalRecHit_ = edm::RefToBase<CaloRecHit>(rh);
    }

  private:

    // original rechit
    edm::RefToBase<CaloRecHit> originalRecHit_;

    ///C cell detid - should be detid or index in collection ?
    unsigned            detId_;             

    /// rechit layer
    PFLayer::Layer      layer_;

    /// rechit energy 
    double              energy_;

    /// time
    double              time_;


    /// depth
    int      depth_;

    /// rechit cell centre: x, y, z
    math::XYZPoint      position_;

    /// rechit cell centre: rho, eta, phi (transient)
    REPPoint positionrep_;

    /// rechit cell axisxyz
    math::XYZVector     axisxyz_;

    /// rechit cell corners
    std::vector< math::XYZPoint > cornersxyz_;

    /// rechit cell corners rho/eta/phi
    std::vector< REPPoint > cornersrep_;
  
    /// indices to existing neighbours (1 common side)
    PFRecHitRefVector   neighbours_;
    std::vector< unsigned short >   neighbourInfos_;

    //Caching the neighbours4/8 per request of Lindsey
    PFRecHitRefVector   neighbours4_;
    PFRecHitRefVector   neighbours8_;


    /// number of corners
    static const unsigned    nCorners_;

    /// set position of one of the corners
    void      setCorner( unsigned i, double posx, double posy, double posz );
  };
  
}

#endif
