//#define EDM_ML_DEBUG

#include "MTDDiskSectorBuilderFromDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"

#include <iomanip>

using namespace std;

namespace {

  pair<DiskSectorBounds*, GlobalVector> computeBounds(const vector<const GeomDet*>& dets) {
    // go over all corners and compute maximum deviations
    float rmin(dets.front()->surface().position().perp());
    float rmax(rmin);
    float zmin(dets.front()->surface().position().z());
    float zmax(zmin);
    float phimin(dets.front()->surface().position().phi());
    float phimax(phimin);

    for (auto const& idet : dets) {
      vector<GlobalPoint> corners = BoundingBox().corners(idet->specificSurface());
      for (auto const& i : corners) {
        float r = i.perp();
        float z = i.z();
        float phi = i.phi();
        rmin = min(rmin, r);
        rmax = max(rmax, r);
        zmin = min(zmin, z);
        zmax = max(zmax, z);
        if (Geom::phiLess(phi, phimin))
          phimin = phi;
        if (Geom::phiLess(phimax, phi))
          phimax = phi;
      }
    }

    if (!Geom::phiLess(phimin, phimax))
      edm::LogError("MTDDetLayers") << " MTDDiskSectorBuilderFromDet : "
                                    << "Something went wrong with Phi Sorting !";
    float zPos = (zmax + zmin) / 2.;
    float phiWin = phimax - phimin;
    float phiPos = (phimax + phimin) / 2.;
    float rmed = (rmin + rmax) / 2.;
    if (phiWin < 0.) {
      if ((phimin < Geom::pi() / 2.) || (phimax > -Geom::pi() / 2.)) {
        edm::LogError("MTDDetLayers") << " something strange going on, please check " << phimin << " " << phimax << " "
                                      << phiWin;
      }
      phiWin += 2. * Geom::pi();
      phiPos += Geom::pi();
    }

    GlobalVector pos(rmed * cos(phiPos), rmed * sin(phiPos), zPos);

    LogTrace("MTDDetLayers") << "MTDDiskSectorBuilderFromDet::computeBounds sector at: " << std::fixed << pos << "\n"
                             << "zmin    : " << std::setw(14) << zmin << "\n"
                             << "zmax    : " << std::setw(14) << zmax << "\n"
                             << "rmin    : " << std::setw(14) << rmin << "\n"
                             << "rmax    : " << std::setw(14) << rmax << "\n"
                             << "phi ref : " << std::setw(14) << phiPos << "\n"
                             << "phi win : " << std::setw(14) << phiWin;

    return make_pair(new DiskSectorBounds(rmin, rmax, zmin - zPos, zmax - zPos, phiWin), pos);
  }

  Surface::RotationType computeRotation(const vector<const GeomDet*>& dets, const Surface::PositionType pos) {
    GlobalVector yAxis = (GlobalVector(pos.x(), pos.y(), 0.)).unit();

    GlobalVector zAxis(0., 0., 1.);
    GlobalVector xAxis = yAxis.cross(zAxis);

    return Surface::RotationType(xAxis, yAxis);
  }

}  // namespace

BoundDiskSector* MTDDiskSectorBuilderFromDet::operator()(const vector<const GeomDet*>& dets) const {
  // check that the dets are all at about the same z
  float zcheck = dets.front()->surface().position().z();
  constexpr double tol(0.5);  // minimal safety check on z position of modules within a sector, width ~ 10 mm
  for (auto const& idet : dets) {
    float zdiff = zcheck - (*idet).surface().position().z();
    if (std::abs(zdiff) > tol) {
      edm::LogError("MTDDetLayers")
          << " MTDDiskSectorBuilderFromDet: Trying to build sector from Dets at different z positions !! Delta_z = "
          << zdiff;
    }
  }

  auto bo = computeBounds(dets);

  Surface::PositionType pos(bo.second.x(), bo.second.y(), bo.second.z());
  Surface::RotationType rot = computeRotation(dets, pos);
  return new BoundDiskSector(pos, rot, bo.first);
}
