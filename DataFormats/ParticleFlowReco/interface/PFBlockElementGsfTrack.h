#ifndef __PFBlockElementGsfTrack__
#define __PFBlockElementGsfTrack__

#include <iostream>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  /// \brief Track Element.
  ///
  /// this class contains a reference to a PFRecTrack
  class PFBlockElementGsfTrack : public PFBlockElement {
  public:
    PFBlockElementGsfTrack() {}

    PFBlockElementGsfTrack(const GsfPFRecTrackRef& gsfref,
                           const math::XYZTLorentzVector& Pin,
                           const math::XYZTLorentzVector& Pout);

    PFBlockElement* clone() const override { return new PFBlockElementGsfTrack(*this); }

    void Dump(std::ostream& out = std::cout, const char* tab = " ") const override;

    /// \return tracktype
    bool trackType(TrackType trType) const override { return (trackType_ >> trType) & 1; }

    /// \set the trackType
    void setTrackType(TrackType trType, bool value) override {
      if (value)
        trackType_ = trackType_ | (1 << trType);
      else
        trackType_ = trackType_ ^ (1 << trType);
    }

    bool isSecondary() const override { return trackType(T_FROM_GAMMACONV); }

    /// \return reference to the corresponding PFGsfRecTrack
    const GsfPFRecTrackRef& GsftrackRefPF() const { return GsftrackRefPF_; }

    /// \return reference to the corresponding GsfTrack
    const reco::GsfTrackRef& GsftrackRef() const { return GsftrackRef_; }

    /// \return position at ECAL entrance
    const math::XYZPointF& positionAtECALEntrance() const { return positionAtECALEntrance_; }

    const GsfPFRecTrack& GsftrackPF() const { return *GsftrackRefPF_; }

    const math::XYZTLorentzVector& Pin() const { return Pin_; }
    const math::XYZTLorentzVector& Pout() const { return Pout_; }

  private:
    /// reference to the corresponding GSF track (transient)
    GsfPFRecTrackRef GsftrackRefPF_;

    /// reference to the corresponding GSF track
    reco::GsfTrackRef GsftrackRef_;

    /// The CorrespondingKFTrackRef is needeed.
    math::XYZTLorentzVector Pin_;
    math::XYZTLorentzVector Pout_;
    unsigned int trackType_;

    /// position at ECAL entrance
    math::XYZPointF positionAtECALEntrance_;
  };
}  // namespace reco

#endif
