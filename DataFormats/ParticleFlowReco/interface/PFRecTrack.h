#ifndef DataFormats_ParticleFlowReco_PFRecTrack_h
#define DataFormats_ParticleFlowReco_PFRecTrack_h

#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"

#include <iostream>

namespace reco {

  /**\class PFRecTrack
     \brief reconstructed track for particle flow
     
     \todo   colin: the structure for trajectory points is very inefficient. 
     \author Renaud Bruneliere
     \date   July 2006
  */
  class PFRecTrack : public PFTrack {

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

    /// get type of algorithm
    unsigned int algoType() const { return algoType_; }

    friend  std::ostream& operator<<(std::ostream& out, 
				     const PFRecTrack& track);

  private:

    /// type of fitting algorithm used to reconstruct the track
    AlgoType_t algoType_;
  };

}

#endif
