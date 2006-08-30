#ifndef DataFormats_ParticleFlowReco_PFTrack_h
#define DataFormats_ParticleFlowReco_PFTrack_h

#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include <iostream>
#include <vector>

namespace reco {

  /**\class PFTrack
     \brief reconstructed track for particle flow
     
     \todo   colin: the structure for trajectory points is very inefficient. 
     \author Renaud Bruneliere
     \date   July 2006
  */
  class PFTrack
  {
  public:

    PFTrack();
  
    PFTrack(double charge);

    PFTrack(const PFTrack& other);
   
    /// add a trajectory measurement
    void addPoint(const reco::PFTrajectoryPoint& trajPt);

    /// set a trajectory point
    void setPoint(unsigned int index,
		  const reco::PFTrajectoryPoint& measurement)
      { trajectoryPoints_[index] = measurement; }

    /// calculate posrep_ once and for all for each point
    void calculatePositionREP();

    /// get electric charge
    double charge() const { return charge_; }

    /// get number of trajectory points
    unsigned int nTrajectoryPoints() const 
      { return trajectoryPoints_.size(); }
    
    /// get number of trajectory measurements in tracker
    unsigned int nTrajectoryMeasurements() const 
      { return (indexOutermost_ ? indexOutermost_ - indexInnermost_ + 1 : 0); }

    /// vector of trajectory points
    const std::vector< reco::PFTrajectoryPoint >& trajectoryPoints() const 
      { return trajectoryPoints_; }
    
    /// get a trajectory point
    const reco::PFTrajectoryPoint& trajectoryPoint(unsigned index) const 
      { return trajectoryPoints_[index]; }

    /// get an extrapolated point
    const reco::PFTrajectoryPoint& extrapolatedPoint(unsigned layerid) const; 

    /// iterator on innermost tracker measurement
    std::vector< reco::PFTrajectoryPoint >::const_iterator innermostMeasurement() const
      { return trajectoryPoints_.begin() + indexInnermost_; }
    
    /// iterator on outermost tracker measurement
    std::vector< reco::PFTrajectoryPoint >::const_iterator outermostMeasurement() const
      { return trajectoryPoints_.begin() + indexOutermost_; }
  
    void         setColor(int color) {color_ = color;}

    int          color() const { return color_; }    

    friend  std::ostream& operator<<(std::ostream& out, 
				     const PFTrack& track);

  private:

    /// maximal number of tracking layers
    static const unsigned int nMaxTrackingLayers_;
  
    /// charge
    double charge_;

    /// vector of trajectory points
    std::vector< reco::PFTrajectoryPoint > trajectoryPoints_;

    /// index innermost tracker measurement
    unsigned int indexInnermost_;

    /// index outermost tracker measurement
    unsigned int indexOutermost_;

    /// color (transient)
    int  color_;

  };

}

#endif
