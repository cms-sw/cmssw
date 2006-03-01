#ifndef TrackReco_TrackExtension_h
#define TrackReco_TrackExtension_h
/** \class reco::TrackExtension
 *
 * Template providing standard extension of a Track.
 * This class is ment to provide track information
 * available in the RECO, not the AOD.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 *
 */
#include "DataFormats/TrackReco/interface/RecHitFwd.h"

namespace reco {

  template<typename ExtraRef>
  class TrackExtension {
  public:
    /// default constructor
    TrackExtension() { }
    /// set reference to "extra" object
    void setExtra( const ExtraRef & ref ) { extra_ = ref; }
    /// reference to "extra" object
    const ExtraRef & extra() const { return extra_; }
    /// return true if the outermost point is valid
    bool outerOk() const { return extra_->outerOk(); }
    /// position of the outermost point
    const math::XYZPoint & outerPosition()  const { return extra_->outerPosition(); }
    /// momentum vector at the outermost point
    const math::XYZVector & outerMomentum() const { return extra_->outerMomentum(); }
    /// first iterator to RecHits
    recHit_iterator recHitsBegin() const { return extra_->recHitsBegin(); }
    /// last iterator to RecHits
    recHit_iterator recHitsEnd()   const { return extra_->recHitsEnd(); }
    /// number of RecHits
    size_t recHitsSize() const { return extra_->recHitsSize(); }
    /// x coordinate of momentum vector at the outermost point
    double outerPx()     const { return extra_->outerPx(); }
    /// y coordinate of momentum vector at the outermost point
    double outerPy()     const { return extra_->outerPy(); }
    /// z coordinate of momentum vector at the outermost point
    double outerPz()     const { return extra_->outerPz(); }
    /// x coordinate of the outermost point
    double outerX()      const { return extra_->outerX(); }
    /// y coordinate of the outermost point
    double outerY()      const { return extra_->outerY(); }
    /// z coordinate of the outermost point
    double outerZ()      const { return extra_->outerZ(); }
    /// magnitude of momentum vector at the outermost point
    double outerP()      const { return extra_->outerP(); }
    /// transverse momentum at the outermost point
    double outerPt()     const { return extra_->outerPt(); }
    /// azimuthal angle of the  momentum vector at the outermost point
    double outerPhi()    const { return extra_->outerPhi(); }
    /// pseudorapidity of the  momentum vector at the outermost point
    double outerEta()    const { return extra_->outerEta(); }
    /// polar angle of the  momentum vector at the outermost point
    double outerTheta()  const { return extra_->outerTheta(); }    
    /// polar radius of the outermost point
    double outerRadius() const { return extra_->outerRadius(); }

  protected:
    /// reference to "extra" extension
    ExtraRef extra_;
  };

}

#endif
