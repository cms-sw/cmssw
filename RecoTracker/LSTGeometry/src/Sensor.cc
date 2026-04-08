#include <cmath>

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "RecoTracker/LSTGeometry/interface/Sensor.h"

using namespace lstgeometry;

// Not sure if there is functionality for this already in CMSSW
bool isInverted(unsigned int moduleId, Location location, Side side, unsigned int layer) {
  bool moduleIdIsEven = moduleId % 2 == 0;
  if (location == Location::endcap) {
    if (side == Side::NegZ) {
      return !moduleIdIsEven;
    } else if (side == Side::PosZ) {
      return moduleIdIsEven;
    }
  } else if (location == Location::barrel) {
    if (side == Side::Center) {
      if (layer <= 3) {
        return !moduleIdIsEven;
      } else if (layer >= 4) {
        return moduleIdIsEven;
      }
    } else if (side == Side::NegZ || side == Side::PosZ) {
      if (layer <= 2) {
        return !moduleIdIsEven;
      } else if (layer == 3) {
        return moduleIdIsEven;
      }
    }
  }
  return false;
}

// This differs from TrackerTopology::isLower since it considers if the module is inverted
bool isLower(unsigned int moduleId, Location location, Side side, unsigned int layer, unsigned int detId) {
  return isInverted(moduleId, location, side, layer) ? !(detId & 1) : (detId & 1);
}

bool isStrip(ModuleType moduleType, bool isInverted, bool isLower) {
  if (moduleType == ModuleType::Ph2SS)
    return true;
  if (isInverted)
    return isLower;
  return !isLower;
}

Sensor::Sensor(unsigned int detId,
               ModuleType moduleType,
               SubDetector subdet,
               Location location,
               Side side,
               unsigned int moduleId,
               unsigned int layer,
               unsigned int ring,
               float centerRho,
               float centerZ,
               float centerPhi,
               Surface const &surface)
    : moduleType(moduleType),
      centerPhi(centerPhi),
      centerX(centerRho * std::cos(centerPhi)),
      centerY(centerRho * std::sin(centerPhi)),
      centerZ(centerZ),
      extra(std::make_unique<SensorExtraData>()) {
  extra->subdet = subdet;
  extra->location = location;
  extra->side = side;
  extra->moduleId = moduleId;
  extra->layer = layer;
  extra->ring = ring;
  extra->inverted = isInverted(moduleId, location, side, layer);
  extra->centerRho = centerRho;
  extra->lower = isLower(moduleId, location, side, layer, detId);
  extra->strip = isStrip(moduleType, extra->inverted, extra->lower);
  extra->centerEta = std::asinh(centerZ / centerRho);
  extra->centerTheta = std::numbers::pi_v<float> / 2. - std::atan(centerZ / centerRho);

  const Bounds &bounds = surface.bounds();
  const RectangularPlaneBounds *rectangular_bounds = dynamic_cast<const RectangularPlaneBounds *>(&bounds);

  float wid, len;
  if (rectangular_bounds) {
    wid = rectangular_bounds->width();
    len = rectangular_bounds->length();
  } else {
    throw std::runtime_error("Sensor::Sensor: Surface bounds are not rectangular");
  }

  auto c1 = GloballyPositioned<float>::LocalPoint(-wid / 2, -len / 2, 0);
  auto c2 = GloballyPositioned<float>::LocalPoint(-wid / 2, len / 2, 0);
  auto c3 = GloballyPositioned<float>::LocalPoint(wid / 2, len / 2, 0);
  auto c4 = GloballyPositioned<float>::LocalPoint(wid / 2, -len / 2, 0);
  auto c1g = surface.toGlobal(c1);
  auto c2g = surface.toGlobal(c2);
  auto c3g = surface.toGlobal(c3);
  auto c4g = surface.toGlobal(c4);
  // store corners with z, x, y ordering
  extra->corners << c1g.z(), c1g.x(), c1g.y(), c2g.z(), c2g.x(), c2g.y(), c3g.z(), c3g.x(), c3g.y(), c4g.z(), c4g.x(),
      c4g.y();

  // Precompute min/max R, Z, Phi for convenience
  extra->minR = std::numeric_limits<float>::max();
  extra->maxR = std::numeric_limits<float>::lowest();
  extra->minZ = std::numeric_limits<float>::max();
  extra->maxZ = std::numeric_limits<float>::lowest();
  extra->minPhi = std::numeric_limits<float>::max();
  float minPosPhi = std::numeric_limits<float>::max();
  extra->maxPhi = std::numeric_limits<float>::lowest();
  float maxNegPhi = std::numeric_limits<float>::lowest();
  unsigned int nPos = 0;
  unsigned int nOverPi2 = 0;
  for (int i = 0; i < extra->corners.rows(); i++) {
    float x = extra->corners(i, 1);
    float y = extra->corners(i, 2);
    float r = std::sqrt(x * x + y * y);
    float z = extra->corners(i, 0);
    float phi = phi_mpi_pi(std::numbers::pi_v<float> + std::atan2(-extra->corners(i, 2), -extra->corners(i, 1)));

    extra->minR = std::min(extra->minR, r);
    extra->maxR = std::max(extra->maxR, r);
    extra->minZ = std::min(extra->minZ, z);
    extra->maxZ = std::max(extra->maxZ, z);
    extra->minPhi = std::min(extra->minPhi, phi);
    extra->maxPhi = std::max(extra->maxPhi, phi);

    if (phi > 0) {
      minPosPhi = std::min(minPosPhi, phi);
      nPos++;
    } else {
      maxNegPhi = std::max(maxNegPhi, phi);
    }
    if (std::fabs(phi) > std::numbers::pi_v<float> / 2.) {
      nOverPi2++;
    }
  }
  if (nOverPi2 == 4 && nPos != 4 && nPos != 0) {
    extra->minPhi = minPosPhi;
    extra->maxPhi = maxNegPhi;
  }
}
