#include "HitPairGeneratorFromLayerPairForPhotonConversion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace GeomDetEnumerators;
using namespace std;
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

// #define mydebug_Seed

typedef PixelRecoRange<float> Range;

namespace {
  template <class T>
  inline T sqr(T t) {
    return t * t;
  }
}  // namespace

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HitPairGeneratorFromLayerPairForPhotonConversion::HitPairGeneratorFromLayerPairForPhotonConversion(
    unsigned int inner, unsigned int outer, LayerCacheType* layerCache, unsigned int nSize, unsigned int max)
    : theLayerCache(*layerCache), theOuterLayer(outer), theInnerLayer(inner), theMaxElement(max) {}

void HitPairGeneratorFromLayerPairForPhotonConversion::hitPairs(const ConversionRegion& convRegion,
                                                                const TrackingRegion& region,
                                                                OrderedHitPairs& result,
                                                                const Layers& layers,
                                                                const edm::Event& event,
                                                                const edm::EventSetup& es) {
  auto oldSize = result.size();
  auto maxNum = result.size() + theMaxElement;

#ifdef mydebug_Seed
  ss.str("");
#endif

  typedef OrderedHitPair::InnerRecHit InnerHit;
  typedef OrderedHitPair::OuterRecHit OuterHit;
  typedef RecHitsSortedInPhi::Hit Hit;

  Layer innerLayerObj = layers[theInnerLayer];
  Layer outerLayerObj = layers[theOuterLayer];

#ifdef mydebug_Seed
  ss << "In " << innerLayerObj.name() << " Out " << outerLayerObj.name() << std::endl;
#endif

  if (!checkBoundaries(*innerLayerObj.detLayer(), convRegion, 40., 60.))
    return;  //FIXME, the maxSearchR(Z) are not optimized
  if (!checkBoundaries(*outerLayerObj.detLayer(), convRegion, 50., 60.))
    return;  //FIXME, the maxSearchR(Z) are not optimized

  /*get hit sorted in phi for each layer: NB: doesn't apply any region cut*/
  const RecHitsSortedInPhi& innerHitsMap = theLayerCache(innerLayerObj, region, es);
  if (innerHitsMap.empty())
    return;

  const RecHitsSortedInPhi& outerHitsMap = theLayerCache(outerLayerObj, region, es);
  if (outerHitsMap.empty())
    return;
  /*----------------*/

  /*This object will check the compatibility of the his in phi among the two layers. */
  //InnerDeltaPhi deltaPhi(innerLayerObj.detLayer(), region, es);

  static const float nSigmaRZ = std::sqrt(12.f);
  //  static const float nSigmaPhi = 3.f;
  vector<RecHitsSortedInPhi::Hit> innerHits, outerHits;
  float outerPhimin, outerPhimax;
  float innerPhimin, innerPhimax;

  /*Getting only the Hits in the outer layer that are compatible with the conversion region*/
  if (!getPhiRange(outerPhimin, outerPhimax, *outerLayerObj.detLayer(), convRegion, es))
    return;
  outerHitsMap.hits(outerPhimin, outerPhimax, outerHits);

#ifdef mydebug_Seed
  ss << "\tophimin, ophimax " << outerPhimin << " " << outerPhimax << std::endl;
#endif

  /* loop on outer hits*/
  for (vector<RecHitsSortedInPhi::Hit>::const_iterator oh = outerHits.begin(); oh != outerHits.end(); ++oh) {
    RecHitsSortedInPhi::Hit ohit = (*oh);
#ifdef mydebug_Seed
    GlobalPoint oPos = ohit->globalPosition();

    ss << "\toPos " << oPos << " r " << oPos.perp() << " phi " << oPos.phi() << " cotTheta " << oPos.z() / oPos.perp()
       << std::endl;
#endif

    /*Check the compatibility of the ohit with the eta of the seeding track*/
    if (checkRZCompatibilityWithSeedTrack(ohit, *outerLayerObj.detLayer(), convRegion))
      continue;

    /*  
    //Do I need this? it uses a compatibility that probably I wouldn't 
    //Removing for the time being

    PixelRecoRange<float> phiRange = deltaPhi( oPos.perp(), oPos.phi(), oPos.z(), nSigmaPhi*(ohit->errorGlobalRPhi()));    
    if (phiRange.empty()) continue;
    */

    std::unique_ptr<const HitRZCompatibility> checkRZ = region.checkRZ(innerLayerObj.detLayer(), ohit, es);
    if (!checkRZ) {
#ifdef mydebug_Seed
      ss << "*******\nNo valid checkRZ\n*******" << std::endl;
#endif
      continue;
    }

    /*Get only the inner hits compatible with the conversion region*/
    innerHits.clear();
    if (!getPhiRange(innerPhimin, innerPhimax, *innerLayerObj.detLayer(), convRegion, es))
      continue;
    innerHitsMap.hits(innerPhimin, innerPhimax, innerHits);

#ifdef mydebug_Seed
    ss << "\tiphimin, iphimax " << innerPhimin << " " << innerPhimax << std::endl;
#endif

    /*Loop on inner hits*/
    for (vector<RecHitsSortedInPhi::Hit>::const_iterator ih = innerHits.begin(), ieh = innerHits.end(); ih < ieh;
         ++ih) {
      GlobalPoint innPos = (*ih)->globalPosition();

#ifdef mydebug_Seed
      ss << "\tinnPos " << innPos << " r " << innPos.perp() << " phi " << innPos.phi() << " cotTheta "
         << innPos.z() / innPos.perp() << std::endl;
#endif

      /*Check the compatibility of the ohit with the eta of the seeding track*/
      if (checkRZCompatibilityWithSeedTrack(*ih, *innerLayerObj.detLayer(), convRegion))
        continue;

      float r_reduced = std::sqrt(sqr(innPos.x() - region.origin().x()) + sqr(innPos.y() - region.origin().y()));
      Range allowed;
      Range hitRZ;
      if (innerLayerObj.detLayer()->location() == barrel) {
        allowed = checkRZ->range(r_reduced);
        float zErr = nSigmaRZ * (*ih)->errorGlobalZ();
        hitRZ = Range(innPos.z() - zErr, innPos.z() + zErr);
      } else {
        allowed = checkRZ->range(innPos.z());
        float rErr = nSigmaRZ * (*ih)->errorGlobalR();
        hitRZ = Range(r_reduced - rErr, r_reduced + rErr);
      }
      Range crossRange = allowed.intersection(hitRZ);

#ifdef mydebug_Seed
      ss << "\n\t\t allowed Range " << allowed.min() << " \t, " << allowed.max() << "\n\t\t hitRz   Range "
         << hitRZ.min() << " \t, " << hitRZ.max() << "\n\t\t Cross   Range " << crossRange.min() << " \t, "
         << crossRange.max() << "\n\t\t the seed track has origin " << convRegion.convPoint() << " \t cotTheta "
         << convRegion.cotTheta() << std::endl;
#endif

      if (!crossRange.empty()) {
#ifdef mydebug_Seed
        ss << "\n\t\t !!!!ACCEPTED!!! \n\n";
#endif
        if (theMaxElement != 0 && result.size() >= maxNum) {
          result.resize(oldSize);
#ifdef mydebug_Seed
          edm::LogError("TooManySeeds") << "number of pairs exceed maximum, no pairs produced";
          std::cout << ss.str();
#endif
          return;
        }
        result.push_back(OrderedHitPair(*ih, ohit));
      }
    }
  }

#ifdef mydebug_Seed
  std::cout << ss.str();
#endif
}

float HitPairGeneratorFromLayerPairForPhotonConversion::getLayerRadius(const DetLayer& layer) {
  if (layer.location() == GeomDetEnumerators::barrel) {
    const BarrelDetLayer& bl = static_cast<const BarrelDetLayer&>(layer);
    float rLayer = bl.specificSurface().radius();

    // the maximal delta phi will be for the innermost hits
    float theThickness = layer.surface().bounds().thickness();
    return rLayer + 0.5f * theThickness;
  }

  //Fixme
  return 0;
}

float HitPairGeneratorFromLayerPairForPhotonConversion::getLayerZ(const DetLayer& layer) {
  if (layer.location() == GeomDetEnumerators::endcap) {
    float layerZ = layer.position().z();
    float theThickness = layer.surface().bounds().thickness();
    float layerZmax = layerZ > 0 ? layerZ + 0.5f * theThickness : layerZ - 0.5f * theThickness;
    return layerZmax;
  } else {
    //Fixme
    return 0;
  }
}
bool HitPairGeneratorFromLayerPairForPhotonConversion::checkBoundaries(const DetLayer& layer,
                                                                       const ConversionRegion& convRegion,
                                                                       float maxSearchR,
                                                                       float maxSearchZ) {
  if (layer.location() == GeomDetEnumerators::barrel) {
    float minZEndCap = 130;
    if (fabs(convRegion.convPoint().z()) > minZEndCap) {
#ifdef mydebug_Seed
      ss << "\tthe conversion seems to be in the endcap. Zconv " << convRegion.convPoint().z() << std::endl;
      std::cout << ss.str();
#endif
      return false;
    }

    float R = getLayerRadius(layer);

    if (convRegion.convPoint().perp() > R) {
#ifdef mydebug_Seed
      ss << "\tthis layer is before the conversion : R layer " << R << " [ Rconv " << convRegion.convPoint().perp()
         << " Zconv " << convRegion.convPoint().z() << std::endl;
      std::cout << ss.str();
#endif
      return false;
    }

    if (R - convRegion.convPoint().perp() > maxSearchR) {
#ifdef mydebug_Seed
      ss << "\tthis layer is far from the conversion more than cut " << maxSearchR << " cm. R layer " << R
         << " [ Rconv " << convRegion.convPoint().perp() << " Zconv " << convRegion.convPoint().z() << std::endl;
      std::cout << ss.str();
#endif
      return false;
    }

  } else if (layer.location() == GeomDetEnumerators::endcap) {
    float Z = getLayerZ(layer);
    if ((convRegion.convPoint().z() > 0 && convRegion.convPoint().z() > Z) ||
        (convRegion.convPoint().z() < 0 && convRegion.convPoint().z() < Z)) {
#ifdef mydebug_Seed
      ss << "\tthis layer is before the conversion : Z layer " << Z << " [ Rconv " << convRegion.convPoint().perp()
         << " Zconv " << convRegion.convPoint().z() << std::endl;
      std::cout << ss.str();
#endif
      return false;
    }

    if (fabs(Z - convRegion.convPoint().z()) > maxSearchZ) {
#ifdef mydebug_Seed
      ss << "\tthis layer is far from the conversion more than cut " << maxSearchZ << " cm. Z layer " << Z
         << " [ Rconv " << convRegion.convPoint().perp() << " Zconv " << convRegion.convPoint().z() << std::endl;
      std::cout << ss.str();
#endif
      return false;
    }
  }
  return true;
}

bool HitPairGeneratorFromLayerPairForPhotonConversion::getPhiRange(float& Phimin,
                                                                   float& Phimax,
                                                                   const DetLayer& layer,
                                                                   const ConversionRegion& convRegion,
                                                                   const edm::EventSetup& es) {
  if (layer.location() == GeomDetEnumerators::barrel) {
    return getPhiRange(Phimin, Phimax, getLayerRadius(layer), convRegion, es);
  } else if (layer.location() == GeomDetEnumerators::endcap) {
    float Z = getLayerZ(layer);
    float R = Z / convRegion.cotTheta();
    return getPhiRange(Phimin, Phimax, R, convRegion, es);  //FIXME
  }
  return false;
}

bool HitPairGeneratorFromLayerPairForPhotonConversion::getPhiRange(
    float& Phimin, float& Phimax, const float& layerR, const ConversionRegion& convRegion, const edm::EventSetup& es) {
  Phimin = reco::deltaPhi(convRegion.convPoint().phi(), 0.);

  float dphi;
  float ptmin = 0.1;
  float DeltaL = layerR - convRegion.convPoint().perp();

  if (DeltaL < 0) {
    Phimin = 0;
    Phimax = 0;
    return false;
  }

  float theRCurvatureMin = PixelRecoUtilities::bendingRadius(ptmin, es);

  if (theRCurvatureMin < DeltaL)
    dphi = atan(DeltaL / layerR);
  else
    dphi = atan(theRCurvatureMin / layerR * (1 - sqrt(1 - sqr(DeltaL / theRCurvatureMin))));

  if (convRegion.charge() > 0) {
    Phimax = Phimin;
    Phimin = Phimax - dphi;
  } else {
    Phimax = Phimin + dphi;
  }

  //std::cout << dphi << " " << Phimin << " " << Phimax << " " << layerR << " " << DeltaL  << " " << convRegion.convPoint().phi() << " " << convRegion.convPoint().perp()<< std::endl;
  return true;
}

bool HitPairGeneratorFromLayerPairForPhotonConversion::checkRZCompatibilityWithSeedTrack(
    const RecHitsSortedInPhi::Hit& hit, const DetLayer& layer, const ConversionRegion& convRegion) {
  static const float nSigmaRZ = std::sqrt(12.f);
  Range hitCotTheta;

  double sigmaCotTheta = convRegion.errTheta() *
                         (1 + convRegion.cotTheta() * convRegion.cotTheta());  //Error Propagation from sigma theta.
  Range allowedCotTheta(convRegion.cotTheta() - nSigmaRZ * sigmaCotTheta,
                        convRegion.cotTheta() + nSigmaRZ * sigmaCotTheta);

  double dz = hit->globalPosition().z() - convRegion.pvtxPoint().z();
  double r_reduced = std::sqrt(sqr(hit->globalPosition().x() - convRegion.pvtxPoint().x()) +
                               sqr(hit->globalPosition().y() - convRegion.pvtxPoint().y()));

  if (layer.location() == GeomDetEnumerators::barrel) {
    float zErr = nSigmaRZ * hit->errorGlobalZ();
    hitCotTheta = Range(getCot(dz - zErr, r_reduced), getCot(dz + zErr, r_reduced));
  } else {
    float rErr = nSigmaRZ * hit->errorGlobalR();
    if (dz > 0)
      hitCotTheta = Range(getCot(dz, r_reduced + rErr), getCot(dz, r_reduced - rErr));
    else
      hitCotTheta = Range(getCot(dz, r_reduced - rErr), getCot(dz, r_reduced + rErr));
  }

  Range crossRange = allowedCotTheta.intersection(hitCotTheta);

#ifdef mydebug_Seed
  ss << "\n\t\t cotTheta allowed Range " << allowedCotTheta.min() << " \t, " << allowedCotTheta.max()
     << "\n\t\t hitCotTheta   Range " << hitCotTheta.min() << " \t, " << hitCotTheta.max() << "\n\t\t Cross   Range "
     << crossRange.min() << " \t, " << crossRange.max() << "\n\t\t the seed track has origin " << convRegion.convPoint()
     << " \t cotTheta " << convRegion.cotTheta() << std::endl;
#endif

  return crossRange.empty();
}

double HitPairGeneratorFromLayerPairForPhotonConversion::getCot(double dz, double dr) {
  if (std::abs(dr) > 1.e-4f)
    return dz / dr;
  else if (dz > 0)
    return 99999.f;
  else
    return -99999.f;
}
