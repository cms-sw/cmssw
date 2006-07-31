#ifndef DataFormats_ParticleFlowReco_PFRecTrack_h
#define DataFormats_ParticleFlowReco_PFRecTrack_h

#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include <iostream>
#include <vector>

namespace reco {

  /**\class PFRecTrack
     \brief reconstructed track for particle flow
     
     \todo   colin: the structure for trajectory points is very inefficient. 
     \author Renaud Bruneliere
     \date   July 2006
  */
  class PFRecTrack
  {
  public:
    
    /// different types of fitting algorithms
    enum AlgoType_t {
      Unknown = 0,
      KF = 1, // Kalman filter 
      GSF = 2 // Gaussian sum filter
    };

    PFRecTrack();
  
    PFRecTrack(double charge, AlgoType_t algoType);

    PFRecTrack(const PFRecTrack& other);
   
    /// add a trajectory measurement
    void addMeasurement(const reco::PFTrajectoryPoint& measurement);

    /// set a trajectory point
    void setPoint(unsigned int index,
		  const reco::PFTrajectoryPoint& measurement)
    { trajectoryPoints_[index] = measurement; }

    /// calculate posrep_ once and for all for each point
    void CalculatePositionREP();

    /// get electric charge
    double getCharge() const    { return charge_; }

    /// get type of algorithm
    unsigned int getAlgoType() const    { return algoType_; }

    /// get number of trajectory points
    unsigned int getNTrajectoryPoints() const 
      { return trajectoryPoints_.size(); }
    
    /// get number of trajectory measurements in tracker
    unsigned int getNTrajectoryMeasurements() const 
      { return (indexOutermost_ ? indexOutermost_ - indexInnermost_ + 1 : 0); }

    /// vector of trajectory points
    const std::vector< reco::PFTrajectoryPoint >& getTrajectoryPoints() const 
      { return trajectoryPoints_; }
    
    /// get a trajectory point
    const reco::PFTrajectoryPoint& getTrajectoryPoint(unsigned index) const 
      { return trajectoryPoints_[index]; }

    /// get extrapolated point
    const reco::PFTrajectoryPoint& getExtrapolatedPoint(unsigned layerid) const; 

    /// iterator on innermost tracker measurement
    std::vector< reco::PFTrajectoryPoint >::const_iterator getInnermostMeasurement() const
      { return trajectoryPoints_.begin() + indexInnermost_; }
    
    /// iterator on outermost tracker measurement
    std::vector< reco::PFTrajectoryPoint >::const_iterator getOutermostMeasurement() const
      { return trajectoryPoints_.begin() + indexOutermost_; }
  
    bool isPropagated() { return (!doPropagation_); }

    void setPropagation(bool doPropagation) { doPropagation_ = doPropagation; }

    void         setColor(int color) {color_ = color;}

    int          getColor() const {return color_;}    

    friend  std::ostream& operator<<(std::ostream& out, 
				     const PFRecTrack& track);

  private:
  
    /// charge
    double charge_;

    /// type of fitting algorithm used to reconstruct the track
    AlgoType_t algoType_;

    /// vector of trajectory points
    std::vector< reco::PFTrajectoryPoint > trajectoryPoints_;

    /// index innermost tracker measurement
    unsigned int indexInnermost_;

    /// index outermost tracker measurement
    unsigned int indexOutermost_;

    /// propagate trajectory to extra positions (transient)
    bool doPropagation_;

    /// color
    int  color_;

  };

}

#endif
