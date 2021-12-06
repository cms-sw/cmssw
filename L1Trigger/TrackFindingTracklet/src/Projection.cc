
#include "L1Trigger/TrackFindingTracklet/interface/Projection.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace std;
using namespace trklet;

void Projection::init(Settings const& settings,
                      unsigned int layerdisk,
                      int iphiproj,
                      int irzproj,
                      int iphider,
                      int irzder,
                      double phiproj,
                      double rzproj,
                      double phiprojder,
                      double rzprojder,
                      double phiprojapprox,
                      double rzprojapprox,
                      double phiprojderapprox,
                      double rzprojderapprox,
                      bool isPSseed) {
  assert(layerdisk < N_LAYER + N_DISK);

  valid_ = true;

  fpgaphiproj_.set(iphiproj, settings.nphibitsstub(layerdisk), true, __LINE__, __FILE__);

  if (layerdisk < N_LAYER) {
    fpgarzproj_.set(irzproj, settings.nzbitsstub(layerdisk), false, __LINE__, __FILE__);
  } else {
    fpgarzproj_.set(irzproj, settings.nrbitsstub(layerdisk), false, __LINE__, __FILE__);
  }

  if (layerdisk < N_LAYER) {
    if (layerdisk < N_PSLAYER) {
      fpgaphiprojder_.set(iphider, settings.nbitsphiprojderL123(), false, __LINE__, __FILE__);
      fpgarzprojder_.set(irzder, settings.nbitszprojderL123(), false, __LINE__, __FILE__);
    } else {
      fpgaphiprojder_.set(iphider, settings.nbitsphiprojderL456(), false, __LINE__, __FILE__);
      fpgarzprojder_.set(irzder, settings.nbitszprojderL456(), false, __LINE__, __FILE__);
    }
  } else {
    fpgaphiprojder_.set(iphider, settings.nbitsphiprojderL123(), false, __LINE__, __FILE__);
    fpgarzprojder_.set(irzder, settings.nrbitsprojderdisk(), false, __LINE__, __FILE__);
  }

  if (layerdisk < N_LAYER) {
    ////Separate the vm projections into zbins
    ////This determines the central bin:
    ////int zbin=4+(zproj.value()>>(zproj.nbits()-3));
    ////But we need some range (particularly for L5L6 seed projecting to L1-L3):
    int offset = isPSseed ? 1 : 4;

    int ztemp = fpgarzproj_.value() >> (fpgarzproj_.nbits() - settings.MEBinsBits() - NFINERZBITS);
    unsigned int zbin1 = (1 << (settings.MEBinsBits() - 1)) + ((ztemp - offset) >> NFINERZBITS);
    unsigned int zbin2 = (1 << (settings.MEBinsBits() - 1)) + ((ztemp + offset) >> NFINERZBITS);

    if (zbin1 >= settings.MEBins()) {
      zbin1 = 0;  //note that zbin1 is unsigned
    }
    if (zbin2 >= settings.MEBins()) {
      zbin2 = settings.MEBins() - 1;
    }

    assert(zbin1 <= zbin2);
    assert(zbin2 - zbin1 <= 1);

    fpgarzbin1projvm_.set(zbin1, settings.MEBinsBits(), true, __LINE__, __FILE__);  // first z bin

    int nextbin = zbin1 != zbin2;
    fpgarzbin2projvm_.set(nextbin, 1, true, __LINE__, __FILE__);  // need to check adjacent z bin?

    //fine vm z bits. Use 4 bits for fine position. starting at zbin 1
    int finez = ((1 << (settings.MEBinsBits() + NFINERZBITS - 1)) + ztemp) - (zbin1 << NFINERZBITS);

    fpgafinerzvm_.set(finez, NFINERZBITS + 1, true, __LINE__, __FILE__);  // fine z postions starting at zbin1

  } else {
    //TODO the -3 and +3 should be evaluated and efficiency for matching hits checked.
    //This code should be migrated in the ProjectionRouter
    double roffset = 3.0;
    int rbin1 = 8.0 * (irzproj * settings.krprojshiftdisk() - roffset - settings.rmindiskvm()) /
                (settings.rmaxdisk() - settings.rmindiskvm());
    int rbin2 = 8.0 * (irzproj * settings.krprojshiftdisk() + roffset - settings.rmindiskvm()) /
                (settings.rmaxdisk() - settings.rmindiskvm());

    if (rbin1 < 0) {
      rbin1 = 0;
    }
    rbin2 = clamp(rbin2, 0, 7);

    assert(rbin1 <= rbin2);
    assert(rbin2 - rbin1 <= 1);

    int finer = 64 *
                ((irzproj * settings.krprojshiftdisk() - settings.rmindiskvm()) -
                 rbin1 * (settings.rmaxdisk() - settings.rmindiskvm()) / 8.0) /
                (settings.rmaxdisk() - settings.rmindiskvm());

    finer = clamp(finer, 0, 15);

    int diff = rbin1 != rbin2;
    if (irzder < 0)
      rbin1 += 8;

    fpgarzbin1projvm_.set(rbin1, 4, true, __LINE__, __FILE__);  // first r bin
    fpgarzbin2projvm_.set(diff, 1, true, __LINE__, __FILE__);   // need to check adjacent r bin

    fpgafinerzvm_.set(finer, 4, true, __LINE__, __FILE__);  // fine r postions starting at rbin1
  }

  //fine phi bits
  int projfinephi =
      (fpgaphiproj_.value() >>
       (fpgaphiproj_.nbits() - (settings.nbitsallstubs(layerdisk) + settings.nbitsvmme(layerdisk) + NFINEPHIBITS))) &
      ((1 << NFINEPHIBITS) - 1);
  fpgafinephivm_.set(projfinephi, NFINEPHIBITS, true, __LINE__, __FILE__);  // fine phi postions

  phiproj_ = phiproj;
  rzproj_ = rzproj;
  phiprojder_ = phiprojder;
  rzprojder_ = rzprojder;

  phiprojapprox_ = phiprojapprox;
  rzprojapprox_ = rzprojapprox;
  phiprojderapprox_ = phiprojderapprox;
  rzprojderapprox_ = rzprojderapprox;
}
