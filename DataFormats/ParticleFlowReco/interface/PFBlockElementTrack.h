#ifndef __PFBlockElementTrack__
#define __PFBlockElementTrack__

#include <iostream>

#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  /// \brief Track Element.
  ///
  /// this class contains a reference to a PFRecTrack
  class PFBlockElementTrack : public PFBlockElement {
  public:
    PFBlockElementTrack() {}

    PFBlockElementTrack(const PFRecTrackRef& ref);

    PFBlockElement* clone() const override { return new PFBlockElementTrack(*this); }

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

    /// set position at ECAL entrance
    void setPositionAtECALEntrance(float x, float y, float z) { positionAtECALEntrance_.SetCoordinates(x, y, z); }

    /// \return position at ECAL entrance
    const math::XYZPointF& positionAtECALEntrance() const { return positionAtECALEntrance_; }

    /// \return reference to the corresponding PFRecTrack
    /// please do not use this function after the block production stage!
    const PFRecTrackRef& trackRefPF() const override { return trackRefPF_; }

    /// \return reference to the corresponding Track
    const reco::TrackRef& trackRef() const override { return trackRef_; }

    /// check if the track is secondary
    bool isSecondary() const override {
      return trackType(T_FROM_DISP) || trackType(T_FROM_GAMMACONV) || trackType(T_FROM_V0);
    }

    bool isPrimary() const override { return trackType(T_TO_DISP); }

    bool isLinkedToDisplacedVertex() const override { return isSecondary() || isPrimary(); }

    /// \return the displaced vertex associated
    const PFDisplacedTrackerVertexRef& displacedVertexRef(TrackType trType) const override {
      if (trType == T_TO_DISP)
        return displacedVertexDaughterRef_;
      else if (trType == T_FROM_DISP)
        return displacedVertexMotherRef_;
      else
        return nullPFDispVertex_;
    }

    /// \set the ref to the displaced vertex interaction
    void setDisplacedVertexRef(const PFDisplacedTrackerVertexRef& niref, TrackType trType) override {
      if (trType == T_TO_DISP) {
        displacedVertexDaughterRef_ = niref;
        setTrackType(trType, true);
      } else if (trType == T_FROM_DISP) {
        displacedVertexMotherRef_ = niref;
        setTrackType(trType, true);
      }
    }

    /// \return reference to the corresponding Muon
    const reco::MuonRef& muonRef() const override { return muonRef_; }

    /// \set reference to the Muon
    void setMuonRef(const MuonRef& muref) override {
      muonRef_ = muref;
      setTrackType(MUON, true);
    }

    /// \return ref to original recoConversion
    const ConversionRefVector& convRefs() const override { return convRefs_; }

    /// \set the ref to  gamma conversion
    void setConversionRef(const ConversionRef& convRef, TrackType trType) override {
      convRefs_.push_back(convRef);
      setTrackType(trType, true);
    }

    /// \return ref to original V0
    const VertexCompositeCandidateRef& V0Ref() const override { return v0Ref_; }

    /// \set the ref to  V0
    void setV0Ref(const VertexCompositeCandidateRef& V0Ref, TrackType trType) override {
      v0Ref_ = V0Ref;
      setTrackType(trType, true);
    }

  private:
    /// reference to the corresponding track (transient)
    PFRecTrackRef trackRefPF_;

    /// reference to the corresponding track
    reco::TrackRef trackRef_;

    unsigned int trackType_;

    /// position at ECAL entrance
    math::XYZPointF positionAtECALEntrance_;

    /// reference to the corresponding pf displaced vertex where this track was created
    PFDisplacedTrackerVertexRef displacedVertexMotherRef_;

    /// reference to the corresponding pf displaced vertex which this track was created
    PFDisplacedTrackerVertexRef displacedVertexDaughterRef_;

    /// reference to the corresponding muon
    reco::MuonRef muonRef_;

    /// reference to reco conversion
    ConversionRefVector convRefs_;

    /// reference to V0
    VertexCompositeCandidateRef v0Ref_;
  };
}  // namespace reco

#endif
