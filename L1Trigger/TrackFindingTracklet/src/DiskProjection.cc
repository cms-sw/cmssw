#include "L1Trigger/TrackFindingTracklet/interface/DiskProjection.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cassert>

using namespace std;
using namespace trklet;

void DiskProjection::init(const Settings* settings,
                          int projdisk,
                          double zproj,
                          int iphiproj,
                          int irproj,
                          int iphider,
                          int irder,
                          double phiproj,
                          double rproj,
                          double phiprojder,
                          double rprojder,
                          double phiprojapprox,
                          double rprojapprox,
                          double phiprojderapprox,
                          double rprojderapprox) {
  assert(abs(projdisk) >= 1);
  assert(abs(projdisk) <= 5);

  valid_ = true;

  zproj_ = zproj;

  projdisk_ = projdisk;

  assert(iphiproj >= 0);

  fpgaphiproj_.set(iphiproj, settings->nphibitsstub(0), true, __LINE__, __FILE__);
  int iphivm = (iphiproj >> (settings->nphibitsstub(0) - 5)) & 0x7;
  if ((abs(projdisk_) % 2) == 1) {
    iphivm ^= 4;
  }
  fpgaphiprojvm_.set(iphivm, 3, true, __LINE__, __FILE__);
  fpgarproj_.set(irproj, settings->nrbitsstub(6), false, __LINE__, __FILE__);
  int irvm = irproj >> (13 - 7) & 0xf;
  fpgarprojvm_.set(irvm, 4, true, __LINE__, __FILE__);
  fpgaphiprojder_.set(iphider, settings->nbitsphiprojderL123(), false, __LINE__, __FILE__);
  fpgarprojder_.set(irder, settings->nrbitsprojderdisk(), false, __LINE__, __FILE__);

  //TODO the -3 and +3 should be evaluated and efficiency for matching hits checked.
  //This code should be migrated in the ProjectionRouter
  int rbin1 = 8.0 * (irproj * settings->krprojshiftdisk() - 3 - settings->rmindiskvm()) /
              (settings->rmaxdisk() - settings->rmindiskvm());
  int rbin2 = 8.0 * (irproj * settings->krprojshiftdisk() + 3 - settings->rmindiskvm()) /
              (settings->rmaxdisk() - settings->rmindiskvm());

  if (irproj * settings->krprojshiftdisk() < 20.0) {
    edm::LogPrint("Tracklet") << " WARNING : irproj = " << irproj << " " << irproj * settings->krprojshiftdisk() << " "
                              << projdisk_;
  }

  if (rbin1 < 0)
    rbin1 = 0;
  if (rbin2 < 0)
    rbin2 = 0;
  if (rbin2 > 7)
    rbin2 = 7;
  assert(rbin1 <= rbin2);
  assert(rbin2 - rbin1 <= 1);

  int finer = 64 *
              ((irproj * settings->krprojshiftdisk() - settings->rmindiskvm()) -
               rbin1 * (settings->rmaxdisk() - settings->rmindiskvm()) / 8.0) /
              (settings->rmaxdisk() - settings->rmindiskvm());

  if (finer < 0)
    finer = 0;
  if (finer > 15)
    finer = 15;

  int diff = rbin1 != rbin2;
  if (irder < 0)
    rbin1 += 8;

  fpgarbin1projvm_.set(rbin1, 4, true, __LINE__, __FILE__);  // first r bin
  fpgarbin2projvm_.set(diff, 1, true, __LINE__, __FILE__);   // need to check adjacent r bin

  fpgafinervm_.set(finer, 4, true, __LINE__, __FILE__);  // fine r postions starting at rbin1

  phiproj_ = phiproj;
  rproj_ = rproj;
  phiprojder_ = phiprojder;
  rprojder_ = rprojder;

  phiprojapprox_ = phiprojapprox;
  rprojapprox_ = rprojapprox;
  phiprojderapprox_ = phiprojderapprox;
  rprojderapprox_ = rprojderapprox;
}
