#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEGenericBase_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEGenericBase_H

#include "PixelCPEBase.h"

class PixelCPEGenericBase : public PixelCPEBase {
public:
  PixelCPEGenericBase(edm::ParameterSet const& conf,
                      const MagneticField* mag,
                      const TrackerGeometry& geom,
                      const TrackerTopology& ttopo,
                      const SiPixelLorentzAngle* lorentzAngle,
                      const SiPixelGenErrorDBObject* genErrorDBObject,
                      const SiPixelLorentzAngle* lorentzAngleWidth = nullptr)
      : PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, nullptr, lorentzAngleWidth, 0){};
};

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEGenericBase_H