#ifndef ParticleFlowReco_PFRecHit_h
#define ParticleFlowReco_PFRecHit_h
/** 
 */
#include <vector>
#include <map>
#include <iostream>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"


namespace reco {

  /**\class PFRecHit
     \brief Particle flow rechit (rechit + geometry and topology information). See clustering algorithm in PFClusterAlgo
          
     \author Colin Bernet
     \date   July 2006

     Feb 2014 [Michalis: 8 years later!Modifying the class to be able to generalize the neighbours for 3D calorimeters ]
  */
  class PFRecHit {

  public:
    using PositionType = GlobalPoint::BasicVectorType;
    using REPPoint = RhoEtaPhi;
    using RepCorners = CaloCellGeometry::RepCorners;
    using REPPointVector = RepCorners;
    using CornersVec = CaloCellGeometry::CornersVec;
    
    enum {
      NONE=0
    };
    /// default constructor. Sets energy and position to zero
    PFRecHit(){}

    PFRecHit(CaloCellGeometry const * caloCell, unsigned int detId,
             PFLayer::Layer layer,
             float energy) :
        caloCell_(caloCell),  detId_(detId),
        layer_(layer), energy_(energy){}


    
    /// copy
    PFRecHit(const PFRecHit& other) = default;
    PFRecHit(PFRecHit&& other) = default;
    PFRecHit & operator=(const PFRecHit& other) = default;
    PFRecHit & operator=(PFRecHit&& other) = default;


    /// destructor
    ~PFRecHit()=default;

    void setEnergy( float energy) { energy_ = energy; }


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


    /// calo cell
    CaloCellGeometry const & caloCell() const { return  *caloCell_; }
    bool hasCaloCell() const { return caloCell_; }
    
    /// rechit detId
    unsigned detId() const {return detId_;}

    /// rechit layer
    PFLayer::Layer layer() const { return layer_; }

    /// rechit energy
    float energy() const { return energy_; }


    /// timing for cleaned hits
    float time() const { return time_; }

    /// depth for segemntation
    int  depth() const { return depth_; }

    /// rechit momentum transverse to the beam, squared.
    double pt2() const { return energy_ * energy_ *
	( position().perp2()/ position().mag2());}


    /// rechit cell centre x, y, z
    PositionType const & position() const { return caloCell().getPosition().basicVector(); }
    
    RhoEtaPhi const &  positionREP() const { return caloCell().repPos(); }

    /// rechit corners
    CornersVec const & getCornersXYZ() const { return caloCell().getCorners(); }    

    RepCorners const & getCornersREP() const { return caloCell().getCornersREP();}
 
 
    /// comparison >= operator
    bool operator>=(const PFRecHit& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator> (const PFRecHit& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator<=(const PFRecHit& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison < operator
    bool operator< (const PFRecHit& rhs) const { return (energy_< rhs.energy_); }

 
  private:

    /// cell geometry
    CaloCellGeometry const * caloCell_=nullptr;
 
    ///cell detid
    unsigned  int        detId_=0;             

    /// rechit layer
    PFLayer::Layer      layer_=PFLayer::NONE;

    /// rechit energy 
    float              energy_=0;

    /// time
    float              time_=-1;

    /// depth
    int      depth_=0;

  
    /// indices to existing neighbours (1 common side)
    PFRecHitRefVector   neighbours_;
    std::vector< unsigned short >   neighbourInfos_;

    //Caching the neighbours4/8 per request of Lindsey
    PFRecHitRefVector   neighbours4_;
    PFRecHitRefVector   neighbours8_;
  };

}
std::ostream& operator<<(std::ostream& out, const reco::PFRecHit& hit);

#endif
