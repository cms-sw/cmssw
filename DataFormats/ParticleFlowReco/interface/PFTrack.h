#ifndef DataFormats_ParticleFlowReco_PFTrack_h
#define DataFormats_ParticleFlowReco_PFTrack_h

#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include <iostream>
#include <vector>

namespace reco {

  /**\class PFTrack
     \brief  Base class for particle flow input reconstructed tracks
     and simulated particles.
     
     A PFTrack contains a vector of PFTrajectoryPoint objects. 
     These points are stored in a vector to benefit from the 
     random access. One must take care of storing the points in 
     the right order, and it might even be necessary to insert dummy points.

     For a PFRecTrack, the ordering of the points is the following:
     
     - point 1: Closest approach

     - point 2: Beam Pipe

     - point 3 to n: Trajectory measurements (from the tracker layers)

     - point n+1: Preshower Layer 1, or dummy point if 
     not in the preshower zone
     
     - point n+2: Preshower Layer 2, or dummy point if 
     not in the preshower zone

     - point n+3: ECAL Entrance

     - point n+4: ECAL expected position of the shower maximum, 
     assuming the track is an electron track.
     
     - point n+5: HCAL Entrance

     For a PFSimParticle, the ordering of the points is the following.
     
     - If the particle decays before ECAL:
     - point 1: start point
     - point 2: end point 

     - If the particle does not decay before ECAL:
     - point 1: start point
     - point 2: PS1 or dummy
     - point 3: PS2 or dummy
     - point 4: ECAL entrance
     - point 5: HCAL entrance
     

     \todo Note that some points are missing, and should be added: shower max,
     intersection with the tracker layers maybe. 
     
     PFRecTracks and PFSimParticles are created in the PFTrackProducer module. 
     \todo   Make this class abstract ? 
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
      /// \todo throw an exception if the number of points is too large
      void addPoint(const reco::PFTrajectoryPoint& trajPt);

      /// set a trajectory point
      void setPoint(unsigned int index,
                    const reco::PFTrajectoryPoint& measurement)
        { trajectoryPoints_[index] = measurement; }

      /// calculate posrep_ once and for all for each point
      /// \todo where is posrep? profile and see if it's necessary.
      void calculatePositionREP();

      /// \return electric charge
      double charge() const { return charge_; }

      /// \return number of trajectory points
      unsigned int nTrajectoryPoints() const 
        { return trajectoryPoints_.size(); }
    
      /// \return number of trajectory measurements in tracker
      unsigned int nTrajectoryMeasurements() const 
        { return (indexOutermost_ ? indexOutermost_ - indexInnermost_ + 1 : 0); }

      /// \return vector of trajectory points
      const std::vector< reco::PFTrajectoryPoint >& trajectoryPoints() const 
        { return trajectoryPoints_; }
    
      /// \return a trajectory point
      const reco::PFTrajectoryPoint& trajectoryPoint(unsigned index) const 
        { return trajectoryPoints_[index]; }

      /// \return an extrapolated point 
      /// \todo throw an exception in case of invalid point.
      const reco::PFTrajectoryPoint& extrapolatedPoint(unsigned layerid) const; 

      /// iterator on innermost tracker measurement
      std::vector< reco::PFTrajectoryPoint >::const_iterator 
        innermostMeasurement() const
        { return trajectoryPoints_.begin() + indexInnermost_; }
    
      /// iterator on outermost tracker measurement
      std::vector< reco::PFTrajectoryPoint >::const_iterator 
        outermostMeasurement() const
        { return trajectoryPoints_.begin() + indexOutermost_; }
    

      void         setColor(int color) {color_ = color;}
    
      int          color() const { return color_; }    

    protected:

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
    std::ostream& operator<<(std::ostream& out, 
                             const PFTrack& track);

}

#endif
