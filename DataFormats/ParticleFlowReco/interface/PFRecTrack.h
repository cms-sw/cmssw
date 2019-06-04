#ifndef DataFormats_ParticleFlowReco_PFRecTrack_h
#define DataFormats_ParticleFlowReco_PFRecTrack_h

#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"
/* #include "DataFormats/Common/interface/RefToBase.h" */
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <iostream>

namespace reco {

  /**\class PFRecTrack
     \brief reconstructed track used as an input to particle flow    

     Additional information w/r to PFTrack: 
     - algorithm used to reconstruct the track
     - track ID, soon to be replaced by a RefToBase to the corresponding Track

     \author Renaud Bruneliere, Michele Pioppi
     \date   July 2006
  */
  class PFRecTrack : public PFTrack {
  public:
    /// different types of fitting algorithms
    enum AlgoType_t {
      Unknown = 0,
      KF = 1,  // Kalman filter
      GSF = 2,
      KF_ELCAND = 3  // Gaussian sum filter
    };

    PFRecTrack();
    ~PFRecTrack(){};
    PFRecTrack(double charge, AlgoType_t algoType, int trackId, const reco::TrackRef& trackref);

    PFRecTrack(double charge, AlgoType_t algoType);

    /*     PFRecTrack(const PFRecTrack& other); */

    /// \return type of algorithm
    unsigned int algoType() const { return algoType_; }

    /// \return id
    int trackId() const { return trackId_; }

    /// \return reference to corresponding track
    const reco::TrackRef& trackRef() const { return trackRef_; }

    /// \set the significance of the signed transverse impact parameter
    void setSTIP(float STIP) { STIP_ = STIP; }

    /// \return the significance of the signed transverse impact parameter
    const float STIP() const { return STIP_; }

  private:
    /// type of fitting algorithm used to reconstruct the track
    AlgoType_t algoType_;

    /// track id
    int trackId_;

    /// reference to corresponding track
    reco::TrackRef trackRef_;
    float STIP_;
  };

  std::ostream& operator<<(std::ostream& out, const PFRecTrack& track);

}  // namespace reco

#endif
