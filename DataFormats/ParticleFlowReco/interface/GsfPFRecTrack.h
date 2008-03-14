#ifndef DataFormats_ParticleFlowReco_GsfPFRecTrack_h
#define DataFormats_ParticleFlowReco_GsfPFRecTrack_h

#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
/* #include "DataFormats/Common/interface/RefToBase.h" */
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBrem.h"
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
  class GsfPFRecTrack : public PFRecTrack
  {

  public:
    GsfPFRecTrack(){};
    GsfPFRecTrack(double charge,
               AlgoType_t algoType,
               int trackId,
               const reco::GsfTrackRef& gtrackref,
               const edm::Ref<std::vector<PFRecTrack> >& kfpfrectrackref);

  

    /// \return reference to corresponding gsftrack
    const reco::GsfTrackRef& 
      gsfTrackRef() const {return gsfTrackRef_;}
    
    /// \return reference to corresponding KF PFRecTrack  (only for GSF PFRecTrack)
    const   edm::Ref<std::vector<PFRecTrack> >&
      kfPFRecTrackRef() const  {return kfPFRecTrackRef_;} 
    /// add a Bremsstrahlung photon
    void addBrem( const reco::PFBrem& brem);

    /// \return the vector of PFBrem
    const std::vector<reco::PFBrem>& PFRecBrem()const {return pfBremVec_;}

    /// \return id
    int trackId() const {return trackId_;}

  private:
    /// reference to corresponding gsf track
    reco::GsfTrackRef     gsfTrackRef_;

    ///ref to the corresponfing PfRecTrack with KF algo (only for PFRecTrack built from GSF track)
    reco::PFRecTrackRef kfPFRecTrackRef_;

    /// vector of PFBrem (empty for KF tracks)
    std::vector<reco::PFBrem> pfBremVec_;


    
    /// track id
    int trackId_;
  };


}

#endif
