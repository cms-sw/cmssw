#ifndef __PFBlockElement__
#define __PFBlockElement__

#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFMultilinksTC.h"  // Glowinski & Gouzevitch

#include <iostream>

namespace reco {
  class PFBlockElementCluster;
  class PFBlockElementTrack;

  /// \brief Abstract base class for a PFBlock element (track, cluster...)
  ///
  /// this class contains a PFRecTrackRef of a
  /// PFClusterRef, depending on the type of the element
  class PFBlockElement {
  public:
    /// possible types for the element
    /// do not modify this enum if you don't know what you're doing!!!
    enum Type {
      NONE = 0,
      TRACK = 1,
      PS1 = 2,
      PS2 = 3,
      ECAL = 4,
      HCAL = 5,
      GSF = 6,
      BREM = 7,
      HFEM = 8,
      HFHAD = 9,
      SC = 10,
      HO = 11,
      HGCAL = 12,
      kNBETypes = 13
    };

    enum TrackType { DEFAULT = 0, T_FROM_DISP, T_TO_DISP, T_FROM_GAMMACONV, MUON, T_FROM_V0 };

    /// standard constructor
    PFBlockElement(Type type = NONE)
        : type_(type), locked_(false), index_(static_cast<unsigned>(-1)), time_(0.f), timeError_(-1.f) {}

    /// destructor
    virtual ~PFBlockElement() {}

    /// print the object inside the element
    virtual void Dump(std::ostream& out = std::cout, const char* tab = " ") const;

    /// necessary to have the edm::OwnVector<PFBlockElement> working
    virtual PFBlockElement* clone() const = 0;

    /// lock element
    void lock() { locked_ = true; }

    /// unlock element
    void unLock() { locked_ = false; }

    /// \return type
    Type type() const { return type_; }

    /// \return tracktype
    virtual bool trackType(TrackType trType) const { return false; }

    /// \set the trackType
    virtual void setTrackType(TrackType trType, bool value) {
      std::cout << "Error in PFBlockElement::setTrackType : this base class method is not implemented" << std::endl;
    }

    /// locked ?
    bool locked() const { return locked_; }

    /// set index
    void setIndex(unsigned index) { index_ = index; }

    /// \return index
    unsigned index() const { return index_; }

    virtual const reco::TrackRef& trackRef() const { return nullTrack_; }
    virtual const PFRecTrackRef& trackRefPF() const { return nullPFRecTrack_; }
    virtual const PFClusterRef& clusterRef() const { return nullPFCluster_; }
    virtual const PFDisplacedTrackerVertexRef& displacedVertexRef(TrackType trType) const { return nullPFDispVertex_; }
    virtual const ConversionRefVector& convRefs() const { return nullConv_; }
    virtual const MuonRef& muonRef() const { return nullMuon_; }
    virtual const VertexCompositeCandidateRef& V0Ref() const { return nullVertex_; }
    virtual void setDisplacedVertexRef(const PFDisplacedTrackerVertexRef& niref, TrackType trType) {
      std::cout << "Error in PFBlockElement::setDisplacedVertexRef : this base class method is not implemented"
                << std::endl;
    }
    virtual void setConversionRef(const ConversionRef& convRef, TrackType trType) {
      std::cout << "Error in PFBlockElement::setConversionRef : this base class method is not implemented" << std::endl;
    }
    virtual void setMuonRef(const MuonRef& muref) {
      std::cout << "Error in PFBlockElement::setMuonRef : this base class method is not implemented" << std::endl;
    }
    virtual void setV0Ref(const VertexCompositeCandidateRef& v0ref, TrackType trType) {
      std::cout << "Error in PFBlockElement::setV0Ref : this base class method is not implemented" << std::endl;
    }

    virtual bool isSecondary() const { return false; }
    virtual bool isPrimary() const { return false; }
    virtual bool isLinkedToDisplacedVertex() const { return false; }

    // Glowinski & Gouzevitch
    void setMultilinks(const PFMultiLinksTC& ml, Type type) { multilinks_[type] = ml; }
    void setIsValidMultilinks(bool isVal, Type type) { multilinks_[type].isValid = isVal; }

    bool isMultilinksValide(Type type) const {
      const auto& it = multilinks_.find(type);
      if (it != multilinks_.end())
        return it->second.isValid;
      else
        return false;  // no multilinks_ for the specified type
    }
    const PFMultilinksType& getMultilinks(Type type) const { return multilinks_.at(type).linkedPFObjects; }
    // ! Glowinski & Gouzevitch

    /// do we have a valid time information
    bool isTimeValid() const { return timeError_ >= 0.f; }
    /// \return the timing
    float time() const { return time_; }
    /// \return the timing uncertainty
    float timeError() const { return timeError_; }
    /// \set the timing information
    void setTime(float time, float timeError = 0.f) {
      time_ = time;
      timeError_ = timeError;
    }

  protected:
    /// type, see PFBlockElementType
    /// \todo replace by a char ?
    Type type_;

    /// locked flag.
    /// \todo can probably be transient. Could be replaced by a
    /// "remaining energy". IS THIS STILL USED ?
    bool locked_;

    /// index in block vector
    unsigned index_;

    // Glowinski & Gouzevitch
    // PFMultiLinks for each different link target type
    std::map<reco::PFBlockElement::Type, PFMultiLinksTC> multilinks_;
    // ! Glowinski & Gouzevitch

    /// timing information (valid if timeError_ >= 0)
    float time_;
    /// timing information uncertainty (<0 if timing not available)
    float timeError_;

    const static reco::TrackRef nullTrack_;
    const static PFRecTrackRef nullPFRecTrack_;
    const static PFClusterRef nullPFCluster_;
    const static PFDisplacedTrackerVertexRef nullPFDispVertex_;
    const static ConversionRefVector nullConv_;
    const static MuonRef nullMuon_;
    const static VertexCompositeCandidateRef nullVertex_;
  };

  std::ostream& operator<<(std::ostream& out, const PFBlockElement& element);

}  // namespace reco
#endif
