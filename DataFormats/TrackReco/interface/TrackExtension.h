#ifndef TrackReco_TrackExtension_h
#define TrackReco_TrackExtension_h
// $Id: Track.h,v 1.10 2006/02/20 14:42:00 llista Exp $
#include "DataFormats/TrackReco/interface/RecHitFwd.h"

namespace reco {

  template<typename ExtraRef>
  class TrackExtension {
  public:
    TrackExtension() { }

    void setExtra( const ExtraRef & ref ) { extra_ = ref; }
    const ExtraRef & extra() const { return extra_; }

    bool outerOk() const { return extra_->outerOk(); }
    const math::XYZPoint & outerPosition()  const { return extra_->outerPosition(); }
    const math::XYZVector & outerMomentum() const { return extra_->outerMomentum(); }
    recHit_iterator recHitsBegin() const { return extra_->recHitsBegin(); }
    recHit_iterator recHitsEnd()   const { return extra_->recHitsEnd(); }
    size_t recHitsSize() const { return extra_->recHitsSize(); }
    double outerPx()     const { return extra_->outerPx(); }
    double outerPy()     const { return extra_->outerPy(); }
    double outerPz()     const { return extra_->outerPz(); }
    double outerX()      const { return extra_->outerX(); }
    double outerY()      const { return extra_->outerY(); }
    double outerZ()      const { return extra_->outerZ(); }
    double outerP()      const { return extra_->outerP(); }
    double outerPt()     const { return extra_->outerPt(); }
    double outerPhi()    const { return extra_->outerPhi(); }
    double outerEta()    const { return extra_->outerEta(); }
    double outerTheta()  const { return extra_->outerTheta(); }    
    double outerRadius() const { return extra_->outerRadius(); }

  protected:
    ExtraRef extra_;
  };

}

#endif
