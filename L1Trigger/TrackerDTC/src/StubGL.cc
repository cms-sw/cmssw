#include "L1Trigger/TrackerDTC/interface/StubGL.h"

#include <cmath>
#include <algorithm>

namespace trackerDTC {

  StubGL::StubGL(const StubFE& stubFE) : stubFE_(&stubFE), valid_(true), overlap_(0, stubFE.setup()->sysNumOverlap()) {
    const Setup* setup = stubFE.setup();
    // convert local coords
    const SensorModule* sm = stubFE.sm();
    const int numCol = sm->psModule() ? setup->mpaNumCol() : setup->cbcNumCol();
    const int numRow = sm->psModule() ? setup->mpaNumRow() : setup->cbcNumRow();
    // global column number in pitch units
    col_ = (stubFE.cic() + .5) * numCol + (stubFE.col() + .5) * setup->feBaseCol();
    // move to module center
    col_ -= (setup->smNumCIC() + 1) * numCol / 2.;
    // adjust for chip orientation
    col_ *= std::pow(-1, sm->signCol());
    // local row number in pitch units
    row_ = stubFE.row() * setup->feBaseRow() + .5;
    // move to chip center
    row_ -= numRow / 2.;
    // adjust for chip orientation
    row_ *= std::pow(-1, sm->signRow());
    // encoded bend
    bend_ = sm->encodeBend(stubFE.bend());
    // adjust for chip orientation
    bend_ *= std::pow(-1, sm->signBend());
    // rough global row for look up in pitch units
    fec_ = (stubFE.fec() + .5) * numRow;
    // move to module center
    fec_ -= setup->cicNumFEC() * numRow / 2;
    // adjust for chip orientation
    fec_ *= std::pow(-1, sm->signRow());
    // convert local to global coordinates
    const double y = col_ * sm->pitchCol();
    // radius of a column of strips/pixel in cm
    const double d = sm->r() + y * sm->sinTilt();
    // stub z in cm
    z_ = sm->z() + y * sm->cosTilt();
    // calulcate m and c for phi look up calulcation r = rm(rowRough) * rowFine + rc(rowRough)
    const double x0 = (fec_ - .5 * numRow) * sm->pitchRow();
    const double x1 = (fec_ + .5 * numRow) * sm->pitchRow();
    const double r0 = std::sqrt(d * d + x0 * x0) - setup->regChosenRofPhi();
    const double r1 = std::sqrt(d * d + x1 * x1) - setup->regChosenRofPhi();
    const double rC = tt::digi((r0 + r1) / 2, setup->glBaseRC());
    const double rM = tt::digi((r1 - r0) / numRow, setup->glBaseRM());
    // stub r wrt chosen RofPhi in cm
    r_ = rC + row_ * rM;
    // calulcate m and c for phi look up calulcation phi = phim(rowRough) * rowFine + phic(rowRough)
    const double phi0 = sm->phi() + std::atan2(x0, d);
    const double phi1 = sm->phi() + std::atan2(x1, d);
    const double phiC = tt::digi((phi0 + phi1) / 2., setup->glBasePhiC());
    const double phiM = tt::digi((phi1 - phi0) / numRow, setup->glBasePhiM());
    // stub phi w.r.t. detector region centre in rad
    phi_ = tt::digi(phiC + row_ * phiM, setup->glBasePhi());
    // apply "eta" cut
    valid_ = false;
    for (double r : {r0, r1}) {
      const double ratioRZ = setup->regChosenRofZ() / (r + setup->regChosenRofPhi());
      // extrapolated z at radius T assuming z0=0
      const double zT = z_ * ratioRZ;
      // extrapolated z0 window at radius T
      const double dZT = setup->regBeamWindowZ() * std::abs(1. - ratioRZ);
      if (std::abs(zT) <= setup->regMaxZT() + dZT)
        valid_ = true;
    }
    // radial (cylindrical) component of sensor separation
    const double dr = sm->sep() / (sm->cosTilt() - sm->sinTilt() * z_ / d);
    // converts bend into inv2R in 1/cm
    const double inv2ROverBend = sm->pitchRow() / dr / d;
    // inv2R in 1/cm
    const double inv2R = bend_ * inv2ROverBend;
    // inv2R uncertainty in 1/cm
    const double dInv2R = setup->smBendCut() * inv2ROverBend;
    double inv2RMin = tt::digi(inv2R - dInv2R, setup->glBaseInv2R());
    double inv2RMax = tt::digi(inv2R + dInv2R, setup->glBaseInv2R());
    // cut on pt
    if (inv2RMin > setup->regMaxInv2R() || inv2RMax < -setup->regMaxInv2R())
      valid_ = false;
    inv2RMin = std::max(inv2RMin, tt::digi(-setup->regMaxInv2R(), setup->glBaseInv2R()));
    inv2RMax = std::min(inv2RMax, tt::digi(setup->regMaxInv2R(), setup->glBaseInv2R()));
    r_ = tt::digi(r_, setup->glBaseR());
    z_ = tt::digi(z_, setup->glBaseZ());
    // range of stub extrapolated phi to radius chosenRofPhi in rad
    double phiTmin = phi_ - r_ * inv2RMin;
    double phiTmax = phi_ - r_ * inv2RMax;
    if (phiTmin > phiTmax)
      std::swap(phiTmin, phiTmax);
    // set overlap regions
    if (phiTmin < 0.)
      overlap_.set(0);
    if (phiTmax >= 0.)
      overlap_.set(1);
  }

}  // namespace trackerDTC
