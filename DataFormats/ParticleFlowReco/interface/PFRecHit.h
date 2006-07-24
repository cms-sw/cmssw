#ifndef ParticleFlowReco_PFRecHit_h
#define ParticleFlowReco_PFRecHit_h
/** 
 */
#include <vector>
#include <map>
#include <iostream>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Math/GenVector/PositionVector3D.h"

namespace reco {

  class PFRecHit {

  public:
    typedef ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > REPPoint;

    /// default constructor. Sets energy and position to zero
    PFRecHit();

    /// constructor from values
    PFRecHit(unsigned detId,
	     int layer,
	     double energy, 
	     const math::XYZPoint& posxyz, 
	     const math::XYZVector& axisxyz, 
	     const std::vector< math::XYZPoint >& cornersxyz);

    PFRecHit(unsigned detId,
	     int layer,
	     double energy, 
	     double posx, double posy, double posz, 
	     double axisx, double axisy, double axisz);    

    /// copy
    PFRecHit(const PFRecHit& other);

    /// destructor
    virtual ~PFRecHit();

    void      SetNeighbours( const std::vector<PFRecHit*>& neighbours );
    
    void      FindPtrsToNeighbours( const std::map<unsigned,  reco::PFRecHit* >& allhits );

    void      SetNWCorner( double posx, double posy, double posz );
    void      SetSWCorner( double posx, double posy, double posz );
    void      SetSECorner( double posx, double posy, double posz );
    void      SetNECorner( double posx, double posy, double posz );

    /// rechit detId
    unsigned GetDetId() const {return detId_;}

    /// rechit layer
    int GetLayer() const { return layer_; }

    /// rechit energy
    double GetEnergy() const { return energy_; }

    /// is seed ? (-1:unknown, 0:no, 1 yes)
    int  IsSeed() const { return isSeed_; }
    
    /// set seed status
    void YouAreSeed(int seedstate=1) {isSeed_ = seedstate;} 

    /// rechit cell centre x, y, z
    const math::XYZPoint& GetPositionXYZ() const { return posxyz_; }

    /// rechit cell centre rho, eta, phi
    const REPPoint& GetPositionREP();

    /// rechit cell axis x, y, z
    const math::XYZVector& GetAxisXYZ() const { return axisxyz_; }    

    /// rechit corners
    const std::vector< math::XYZPoint >& GetCornersXYZ() const 
      { return cornersxyz_; }    

/*     const std::vector< PFRecHit* >& GetNeighbours() const  */
/*       {return neighbours_;}   */
    
    const std::vector< PFRecHit* >& GetNeighbours4() const 
      {return neighbours4_;}  

    const std::vector< PFRecHit* >& GetNeighbours8() const 
      {return neighbours8_;}  

    const std::vector< unsigned >& GetNeighboursIds4() const 
      {return neighboursIds4_;}  

    const std::vector< unsigned >& GetNeighboursIds8() const 
      {return neighboursIds8_;}  

    /// comparison >= operator
    bool operator>=(const PFRecHit& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator> (const PFRecHit& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator<=(const PFRecHit& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison < operator
    bool operator< (const PFRecHit& rhs) const { return (energy_< rhs.energy_); }

    friend std::ostream& operator<<(std::ostream& out, const reco::PFRecHit& hit);

  private:

    /// cell detid
    unsigned            detId_;             

    /// rechit layer
    int                 layer_;

    /// rechit energy
    double              energy_;

    /// is this a seed ? (-1:unknown, 0:no, 1 yes) (transient)
    int                 isSeed_;
 
    /// rechit cell centre: x, y, z
    math::XYZPoint      posxyz_;

    /// rechit cell centre: rho, eta, phi (transient)
    REPPoint            posrep_;

    /// rechit cell axisxyz
    math::XYZVector     axisxyz_;

    /// rechit cell corners
    std::vector< math::XYZPoint > cornersxyz_;

    /// id's of neighbours
    std::vector<unsigned>    neighboursIds4_;

    /// id's of neighbours
    std::vector<unsigned>    neighboursIds8_;

    /// pointers to neighbours (if null: no neighbour here) (transient)
/*     std::vector<PFRecHit*>   neighbours_; */
  
    /// pointers to existing neighbours (1 common side) (transient)
    std::vector<PFRecHit*>   neighbours4_;

    /// pointers to existing neighbours (1 common side or diagonal) (transient)
    std::vector<PFRecHit*>   neighbours8_;

    /// number of neighbours
    static const unsigned    nNeighbours_;
    
    /// number of corners
    static const unsigned    nCorners_;

    void      SetCorner( unsigned i, double posx, double posy, double posz );
  };
  
}

#endif
