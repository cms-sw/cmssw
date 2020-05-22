#include "L1Trigger/TrackFindingTracklet/interface/LayerProjection.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace trklet;

void LayerProjection::init(Settings const& settings,
                           int projlayer,
                           double rproj,
                           int iphiproj,
                           int izproj,
                           int iphider,
                           int izder,
                           double phiproj,
                           double zproj,
                           double phiprojder,
                           double zprojder,
                           double phiprojapprox,
                           double zprojapprox,
                           double phiprojderapprox,
                           double zprojderapprox) {
  assert(projlayer > 0);
  assert(projlayer <= N_LAYER);

  valid_ = true;

  rproj_ = rproj;

  projlayer_ = projlayer;

  assert(iphiproj >= 0);

  if (rproj < 60.0) {
    fpgaphiproj_.set(iphiproj, settings.nphibitsstub(0), true, __LINE__, __FILE__);
    int iphivm = (iphiproj >> (settings.nphibitsstub(0) - 5)) & 0x7;
    if ((projlayer_ % 2) == 1) {
      iphivm ^= 4;
    }
    fpgaphiprojvm_.set(iphivm, 3, true, __LINE__, __FILE__);
    fpgazproj_.set(izproj, settings.nzbitsstub(0), false, __LINE__, __FILE__);
    int izvm = izproj >> (12 - 7) & 0xf;
    fpgazprojvm_.set(izvm, 4, true, __LINE__, __FILE__);
    fpgaphiprojder_.set(iphider, settings.nbitsphiprojderL123(), false, __LINE__, __FILE__);
    fpgazprojder_.set(izder, settings.nbitszprojderL123(), false, __LINE__, __FILE__);
  } else {
    fpgaphiproj_.set(iphiproj, settings.nphibitsstub(5), true, __LINE__, __FILE__);
    int iphivm = (iphiproj >> (settings.nphibitsstub(5) - 5)) & 0x7;
    if ((projlayer_ % 2) == 1) {
      iphivm ^= 4;
    }
    fpgaphiprojvm_.set(iphivm, 3, true, __LINE__, __FILE__);
    fpgazproj_.set(izproj, settings.nzbitsstub(5), false, __LINE__, __FILE__);
    int izvm = izproj >> (8 - 7) & 0xf;
    fpgazprojvm_.set(izvm, 4, true, __LINE__, __FILE__);
    fpgaphiprojder_.set(iphider, settings.nbitsphiprojderL456(), false, __LINE__, __FILE__);
    fpgazprojder_.set(izder, settings.nbitszprojderL456(), false, __LINE__, __FILE__);
  }

  ////Separate the vm projections into zbins
  ////This determines the central bin:
  ////int zbin=4+(zproj.value()>>(zproj.nbits()-3));
  ////But we need some range (particularly for L5L6 seed projecting to L1-L3):
  unsigned int zbin1 = (1 << (settings.MEBinsBits() - 1)) +
                       (((fpgazproj_.value() >> (fpgazproj_.nbits() - settings.MEBinsBits() - 2)) - 2) >> 2);
  unsigned int zbin2 = (1 << (settings.MEBinsBits() - 1)) +
                       (((fpgazproj_.value() >> (fpgazproj_.nbits() - settings.MEBinsBits() - 2)) + 2) >> 2);
  if (zbin1 >= settings.MEBins())
    zbin1 = 0;  //note that zbin1 is unsigned
  if (zbin2 >= settings.MEBins())
    zbin2 = settings.MEBins() - 1;
  assert(zbin1 <= zbin2);
  assert(zbin2 - zbin1 <= 1);
  fpgazbin1projvm_.set(zbin1, settings.MEBinsBits(), true, __LINE__, __FILE__);  // first z bin
  if (zbin1 == zbin2)
    fpgazbin2projvm_.set(0, 1, true, __LINE__, __FILE__);  // don't need to check adjacent z bin
  else
    fpgazbin2projvm_.set(1, 1, true, __LINE__, __FILE__);  // do need to check next z bin

  //fine vm z bits. Use 4 bits for fine position. starting at zbin 1
  int finez = ((1 << (settings.MEBinsBits() + 2)) +
               (fpgazproj_.value() >> (fpgazproj_.nbits() - (settings.MEBinsBits() + 3)))) -
              (zbin1 << 3);

  fpgafinezvm_.set(finez, 4, true, __LINE__, __FILE__);  // fine z postions starting at zbin1

  phiproj_ = phiproj;
  zproj_ = zproj;
  phiprojder_ = phiprojder;
  zprojder_ = zprojder;

  phiprojapprox_ = phiprojapprox;
  zprojapprox_ = zprojapprox;
  phiprojderapprox_ = phiprojderapprox;
  zprojderapprox_ = zprojderapprox;
}
