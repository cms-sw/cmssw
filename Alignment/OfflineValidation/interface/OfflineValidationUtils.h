#ifndef Alignment_OfflineValidation_Utils_h
#define Alignment_OfflineValidation_Utils_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

namespace alignment {
  namespace offlineValidationUtils {
    //*************************************************************
    static bool isHit2D(const TrackingRecHit &hit, const TrackerGeometry *geom, const SiPixelPI::phase &thePhase)
    //*************************************************************
    {
      // helper functionals
      auto isPinPS = [&](DetId detId) {
        // Select only P-hits from the OT barrel
        return (geom->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP);
      };

      auto isPixel = [&](DetId detId) {
        auto subId = detId.subdetId();
        return (subId == PixelSubdetector::PixelBarrel || subId == PixelSubdetector::PixelEndcap);
      };

      bool countStereoHitAs2D_ = true;
      // we count SiStrip stereo modules as 2D if selected via countStereoHitAs2D_
      // (since they provide theta information)
      if (!hit.isValid() ||
          (hit.dimension() < 2 && !countStereoHitAs2D_ && !dynamic_cast<const SiStripRecHit1D *>(&hit))) {
        return false;  // real RecHit1D - but SiStripRecHit1D depends on countStereoHitAs2D_
      } else {
        const DetId detId(hit.geographicalId());
        if (detId.det() == DetId::Tracker) {
          if (isPixel(detId)) {
            return true;  // pixel is always 2D
          } else {
            if (thePhase == SiPixelPI::phase::two) {
              // if the hit is phase-2
              if (isPinPS(detId))
                return true;
              else
                return false;
            } else {
              // should be SiStrip now
              const SiStripDetId stripId(detId);
              if (stripId.stereo())
                return countStereoHitAs2D_;  // stereo modules
              else if (dynamic_cast<const SiStripRecHit1D *>(&hit) || dynamic_cast<const SiStripRecHit2D *>(&hit))
                return false;  // rphi modules hit
              //the following two are not used any more since ages...
              else if (dynamic_cast<const SiStripMatchedRecHit2D *>(&hit))
                return true;  // matched is 2D
              else if (dynamic_cast<const ProjectedSiStripRecHit2D *>(&hit)) {
                const ProjectedSiStripRecHit2D *pH = static_cast<const ProjectedSiStripRecHit2D *>(&hit);
                return (countStereoHitAs2D_ && isHit2D(pH->originalHit(), geom, thePhase));  // depends on original...
              } else {
                edm::LogError("UnknownType") << "alignment::offlineValidationUtils::isHit2D"
                                             << "Tracker hit not in pixel, neither SiStripRecHit[12]D nor "
                                             << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
                return false;
              }
            }
          }
        } else {  // not tracker??
          edm::LogWarning("DetectorMismatch") << "alignment::offlineValidationUtils::isHit2D"
                                              << "Hit not in tracker with 'official' dimension >=2.";
          return true;  // dimension() >= 2 so accept that...
        }
      }
      // never reached...
    }
  }  // namespace offlineValidationUtils
}  // namespace alignment

#endif  // Alignment_OfflineValidation_Utils_h
