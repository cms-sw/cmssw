#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorDisk.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorOverlap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/L1TCommon/interface/BitShift.h"

using namespace std;
using namespace trklet;

TrackletCalculatorBase::TrackletCalculatorBase(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global) {}

void TrackletCalculatorBase::exacttracklet(double r1,
                                           double z1,
                                           double phi1,
                                           double r2,
                                           double z2,
                                           double phi2,
                                           double,
                                           double& rinv,
                                           double& phi0,
                                           double& t,
                                           double& z0,
                                           double phiproj[N_LAYER - 2],
                                           double zproj[N_LAYER - 2],
                                           double phider[N_LAYER - 2],
                                           double zder[N_LAYER - 2],
                                           double phiprojdisk[N_DISK],
                                           double rprojdisk[N_DISK],
                                           double phiderdisk[N_DISK],
                                           double rderdisk[N_DISK]) {
  double deltaphi = reco::reduceRange(phi1 - phi2);

  double dist = sqrt(r2 * r2 + r1 * r1 - 2 * r1 * r2 * cos(deltaphi));

  rinv = 2 * sin(deltaphi) / dist;

  double phi1tmp = phi1 - phimin_;

  phi0 = reco::reduceRange(phi1tmp + asin(0.5 * r1 * rinv));

  double rhopsi1 = 2 * asin(0.5 * r1 * rinv) / rinv;
  double rhopsi2 = 2 * asin(0.5 * r2 * rinv) / rinv;

  t = (z1 - z2) / (rhopsi1 - rhopsi2);

  z0 = z1 - t * rhopsi1;

  for (unsigned int i = 0; i < N_LAYER - 2; i++) {
    exactproj(settings_.rmean(settings_.projlayers(iSeed_, i) - 1),
              rinv,
              phi0,
              t,
              z0,
              phiproj[i],
              zproj[i],
              phider[i],
              zder[i]);
  }

  for (unsigned int i = 0; i < N_DISK; i++) {
    exactprojdisk(settings_.zmean(i), rinv, phi0, t, z0, phiprojdisk[i], rprojdisk[i], phiderdisk[i], rderdisk[i]);
  }
}

void TrackletCalculatorBase::exacttrackletdisk(double r1,
                                               double z1,
                                               double phi1,
                                               double r2,
                                               double z2,
                                               double phi2,
                                               double,
                                               double& rinv,
                                               double& phi0,
                                               double& t,
                                               double& z0,
                                               double phiprojLayer[N_PSLAYER],  //=3 (project to PS barrel layers only)
                                               double zprojLayer[N_PSLAYER],
                                               double phiderLayer[N_PSLAYER],
                                               double zderLayer[N_PSLAYER],
                                               double phiproj[N_DISK - 2],  //=3 (max project to 3 other disks)
                                               double rproj[N_DISK - 2],
                                               double phider[N_DISK - 2],
                                               double rder[N_DISK - 2]) {
  double deltaphi = reco::reduceRange(phi1 - phi2);

  double dist = sqrt(r2 * r2 + r1 * r1 - 2 * r1 * r2 * cos(deltaphi));

  rinv = 2 * sin(deltaphi) / dist;

  double phi1tmp = phi1 - phimin_;

  phi0 = reco::reduceRange(phi1tmp + asin(0.5 * r1 * rinv));

  double rhopsi1 = 2 * asin(0.5 * r1 * rinv) / rinv;
  double rhopsi2 = 2 * asin(0.5 * r2 * rinv) / rinv;

  t = (z1 - z2) / (rhopsi1 - rhopsi2);

  z0 = z1 - t * rhopsi1;

  for (unsigned int i = 0; i < N_DISK - 2; i++) {
    exactprojdisk(settings_.zmean(settings_.projdisks(iSeed_, i) - 1),
                  rinv,
                  phi0,
                  t,
                  z0,
                  phiproj[i],
                  rproj[i],
                  phider[i],
                  rder[i]);
  }

  for (unsigned int i = 0; i < N_DISK - 2; i++) {
    exactproj(settings_.rmean(i), rinv, phi0, t, z0, phiprojLayer[i], zprojLayer[i], phiderLayer[i], zderLayer[i]);
  }
}

void TrackletCalculatorBase::exacttrackletOverlap(double r1,
                                                  double z1,
                                                  double phi1,
                                                  double r2,
                                                  double z2,
                                                  double phi2,
                                                  double,
                                                  double& rinv,
                                                  double& phi0,
                                                  double& t,
                                                  double& z0,
                                                  double phiprojLayer[N_PSLAYER],
                                                  double zprojLayer[N_PSLAYER],
                                                  double phiderLayer[N_PSLAYER],
                                                  double zderLayer[N_PSLAYER],
                                                  double phiproj[N_DISK - 2],
                                                  double rproj[N_DISK - 2],
                                                  double phider[N_DISK - 2],
                                                  double rder[N_DISK - 2]) {
  double deltaphi = reco::reduceRange(phi1 - phi2);

  double dist = sqrt(r2 * r2 + r1 * r1 - 2 * r1 * r2 * cos(deltaphi));

  rinv = 2 * sin(deltaphi) / dist;

  if (r1 > r2)
    rinv = -rinv;

  double phi1tmp = phi1 - phimin_;

  phi0 = reco::reduceRange(phi1tmp + asin(0.5 * r1 * rinv));

  double rhopsi1 = 2 * asin(0.5 * r1 * rinv) / rinv;
  double rhopsi2 = 2 * asin(0.5 * r2 * rinv) / rinv;

  t = (z1 - z2) / (rhopsi1 - rhopsi2);

  z0 = z1 - t * rhopsi1;

  for (int i = 0; i < 4; i++) {
    exactprojdisk(settings_.zmean(i + 1), rinv, phi0, t, z0, phiproj[i], rproj[i], phider[i], rder[i]);
  }

  for (int i = 0; i < 1; i++) {
    exactproj(settings_.rmean(i), rinv, phi0, t, z0, phiprojLayer[i], zprojLayer[i], phiderLayer[i], zderLayer[i]);
  }
}

void TrackletCalculatorBase::exactproj(double rproj,
                                       double rinv,
                                       double phi0,
                                       double t,
                                       double z0,
                                       double& phiproj,
                                       double& zproj,
                                       double& phider,
                                       double& zder) {
  phiproj = phi0 - asin(0.5 * rproj * rinv);
  zproj = z0 + (2 * t / rinv) * asin(0.5 * rproj * rinv);

  phider = -0.5 * rinv / sqrt(1 - pow(0.5 * rproj * rinv, 2));
  zder = t / sqrt(1 - pow(0.5 * rproj * rinv, 2));
}

void TrackletCalculatorBase::exactprojdisk(double zproj,
                                           double rinv,
                                           double phi0,
                                           double t,
                                           double z0,
                                           double& phiproj,
                                           double& rproj,
                                           double& phider,
                                           double& rder) {
  if (t < 0)
    zproj = -zproj;

  double tmp = rinv * (zproj - z0) / (2.0 * t);
  rproj = (2.0 / rinv) * sin(tmp);
  phiproj = phi0 - tmp;

  phider = -rinv / (2 * t);
  rder = cos(tmp) / t;
}

void TrackletCalculatorBase::addDiskProj(Tracklet* tracklet, int disk) {
  disk = std::abs(disk);

  FPGAWord fpgar = tracklet->proj(N_LAYER + disk - 1).fpgarzproj();

  if (fpgar.value() * settings_.krprojshiftdisk() < settings_.rmindiskvm())
    return;
  if (fpgar.value() * settings_.krprojshiftdisk() > settings_.rmaxdisk())
    return;

  FPGAWord fpgaphi = tracklet->proj(N_LAYER + disk - 1).fpgaphiproj();

  int iphivmRaw = fpgaphi.value() >> (fpgaphi.nbits() - 5);

  int iphi = iphivmRaw / (32 / settings_.nallstubs(disk + N_LAYER - 1));

  addProjectionDisk(disk, iphi, trackletprojdisks_[disk - 1][iphi], tracklet);
}

bool TrackletCalculatorBase::addLayerProj(Tracklet* tracklet, int layer) {
  assert(layer > 0);

  FPGAWord fpgaz = tracklet->proj(layer - 1).fpgarzproj();
  FPGAWord fpgaphi = tracklet->proj(layer - 1).fpgaphiproj();

  if (fpgaphi.atExtreme())
    edm::LogProblem("Tracklet") << "at extreme! " << fpgaphi.value();

  assert(!fpgaphi.atExtreme());

  if (fpgaz.atExtreme())
    return false;

  if (std::abs(fpgaz.value() * settings_.kz()) > settings_.zlength())
    return false;

  int iphivmRaw = fpgaphi.value() >> (fpgaphi.nbits() - 5);
  int iphi = iphivmRaw / (32 / settings_.nallstubs(layer - 1));

  addProjection(layer, iphi, trackletprojlayers_[layer - 1][iphi], tracklet);

  return true;
}

void TrackletCalculatorBase::addProjection(int layer,
                                           int iphi,
                                           TrackletProjectionsMemory* trackletprojs,
                                           Tracklet* tracklet) {
  if (trackletprojs == nullptr) {
    if (settings_.warnNoMem()) {
      edm::LogVerbatim("Tracklet") << "No projection memory exists in " << getName() << " for layer = " << layer
                                   << " iphi = " << iphi + 1;
    }
    return;
  }
  assert(trackletprojs != nullptr);
  trackletprojs->addProj(tracklet);
}

void TrackletCalculatorBase::addProjectionDisk(int disk,
                                               int iphi,
                                               TrackletProjectionsMemory* trackletprojs,
                                               Tracklet* tracklet) {
  if (iSeed_ == Seed::L3L4 && abs(disk) == 4)
    return;  //L3L4 projections to D3 are not used. Should be in configuration
  if (trackletprojs == nullptr) {
    if (iSeed_ == Seed::L3L4 && abs(disk) == 3)
      return;  //L3L4 projections to D3 are not used.
    if (settings_.warnNoMem()) {
      edm::LogVerbatim("Tracklet") << "No projection memory exists in " << getName() << " for disk = " << abs(disk)
                                   << " iphi = " << iphi + 1;
    }
    return;
  }
  assert(trackletprojs != nullptr);
  trackletprojs->addProj(tracklet);
}

bool TrackletCalculatorBase::goodTrackPars(bool goodrinv, bool goodz0) {
  bool success = true;
  if (!goodrinv) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " TrackletCalculatorBase irinv too large";
    }
    success = false;
  }
  if (!goodz0) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " TrackletCalculatorBase z0 cut to large";
    }
    success = false;
  }
  return success;
}

bool TrackletCalculatorBase::inSector(int iphi0, int irinv, double phi0approx, double rinvapprox) {
  double phicritapprox = phi0approx - asin(0.5 * settings_.rcrit() * rinvapprox);

  int ifactor = 0.5 * settings_.rcrit() * settings_.krinvpars() / settings_.kphi0pars() * (1 << 8);
  int iphicrit = iphi0 - (irinv >> 8) * ifactor;

  int iphicritmincut = settings_.phicritminmc() / globals_->ITC_L1L2()->phi0_final.K();
  int iphicritmaxcut = settings_.phicritmaxmc() / globals_->ITC_L1L2()->phi0_final.K();

  bool keepapprox = (phicritapprox > settings_.phicritminmc()) && (phicritapprox < settings_.phicritmaxmc()),
       keep = (iphicrit > iphicritmincut) && (iphicrit < iphicritmaxcut);
  if (settings_.debugTracklet())
    if (keepapprox && !keep)
      edm::LogVerbatim("Tracklet") << getName()
                                   << " Tracklet kept with exact phicrit cut but not approximate, phicritapprox: "
                                   << phicritapprox;
  if (settings_.usephicritapprox()) {
    return keepapprox;
  } else {
    return keep;
  }

  return true;
}

bool TrackletCalculatorBase::barrelSeeding(const Stub* innerFPGAStub,
                                           const L1TStub* innerStub,
                                           const Stub* outerFPGAStub,
                                           const L1TStub* outerStub) {
  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "TrackletCalculatorBase " << getName()
                                 << " trying stub pair in layer (inner outer): " << innerFPGAStub->layer().value()
                                 << " " << outerFPGAStub->layer().value();
  }

  assert(outerFPGAStub->layerdisk() < N_LAYER);
  assert(layerdisk1_ == (unsigned int)innerFPGAStub->layer().value());
  assert(layerdisk1_ < N_LAYER && layerdisk2_ < N_LAYER);

  double r1 = innerStub->r();
  double z1 = innerStub->z();
  double phi1 = innerStub->phi();

  double r2 = outerStub->r();
  double z2 = outerStub->z();
  double phi2 = outerStub->phi();

  double rinv, phi0, t, z0;

  double phiproj[N_LAYER - 2], zproj[N_LAYER - 2], phider[N_LAYER - 2], zder[N_LAYER - 2];
  double phiprojdisk[N_DISK], rprojdisk[N_DISK], phiderdisk[N_DISK], rderdisk[N_DISK];

  exacttracklet(r1,
                z1,
                phi1,
                r2,
                z2,
                phi2,
                outerStub->sigmaz(),
                rinv,
                phi0,
                t,
                z0,
                phiproj,
                zproj,
                phider,
                zder,
                phiprojdisk,
                rprojdisk,
                phiderdisk,
                rderdisk);

  if (settings_.useapprox()) {
    phi1 = innerFPGAStub->phiapprox(phimin_, phimax_);
    z1 = innerFPGAStub->zapprox();
    r1 = innerFPGAStub->rapprox();

    phi2 = outerFPGAStub->phiapprox(phimin_, phimax_);
    z2 = outerFPGAStub->zapprox();
    r2 = outerFPGAStub->rapprox();
  }

  double rinvapprox, phi0approx, tapprox, z0approx;
  double phiprojapprox[N_LAYER - 2], zprojapprox[N_LAYER - 2];
  double phiprojdiskapprox[N_DISK], rprojdiskapprox[N_DISK];

  IMATH_TrackletCalculator* ITC;
  if (iSeed_ == 0)
    ITC = globals_->ITC_L1L2();
  else if (iSeed_ == 1)
    ITC = globals_->ITC_L2L3();
  else if (iSeed_ == 2)
    ITC = globals_->ITC_L3L4();
  else
    ITC = globals_->ITC_L5L6();

  ITC->r1.set_fval(r1 - settings_.rmean(layerdisk1_));
  ITC->r2.set_fval(r2 - settings_.rmean(layerdisk2_));
  ITC->z1.set_fval(z1);
  ITC->z2.set_fval(z2);
  double sphi1 = angle0to2pi::make0To2pi(phi1 - phimin_);
  double sphi2 = angle0to2pi::make0To2pi(phi2 - phimin_);

  ITC->phi1.set_fval(sphi1);
  ITC->phi2.set_fval(sphi2);

  ITC->rproj0.set_fval(settings_.rmean(settings_.projlayers(iSeed_, 0) - 1));
  ITC->rproj1.set_fval(settings_.rmean(settings_.projlayers(iSeed_, 1) - 1));
  ITC->rproj2.set_fval(settings_.rmean(settings_.projlayers(iSeed_, 2) - 1));
  ITC->rproj3.set_fval(settings_.rmean(settings_.projlayers(iSeed_, 3) - 1));

  ITC->zproj0.set_fval(t > 0 ? settings_.zmean(0) : -settings_.zmean(0));
  ITC->zproj1.set_fval(t > 0 ? settings_.zmean(1) : -settings_.zmean(1));
  ITC->zproj2.set_fval(t > 0 ? settings_.zmean(2) : -settings_.zmean(2));
  ITC->zproj3.set_fval(t > 0 ? settings_.zmean(3) : -settings_.zmean(3));
  ITC->zproj4.set_fval(t > 0 ? settings_.zmean(4) : -settings_.zmean(4));

  ITC->rinv_final.calculate();
  ITC->phi0_final.calculate();
  ITC->t_final.calculate();
  ITC->z0_final.calculate();

  ITC->phiL_0_final.calculate();
  ITC->phiL_1_final.calculate();
  ITC->phiL_2_final.calculate();
  ITC->phiL_3_final.calculate();

  ITC->zL_0_final.calculate();
  ITC->zL_1_final.calculate();
  ITC->zL_2_final.calculate();
  ITC->zL_3_final.calculate();

  ITC->phiD_0_final.calculate();
  ITC->phiD_1_final.calculate();
  ITC->phiD_2_final.calculate();
  ITC->phiD_3_final.calculate();
  ITC->phiD_4_final.calculate();

  ITC->rD_0_final.calculate();
  ITC->rD_1_final.calculate();
  ITC->rD_2_final.calculate();
  ITC->rD_3_final.calculate();
  ITC->rD_4_final.calculate();

  ITC->der_phiL_final.calculate();
  ITC->der_zL_final.calculate();
  ITC->der_phiD_final.calculate();
  ITC->der_rD_final.calculate();

  //store the approximate results
  rinvapprox = ITC->rinv_final.fval();
  phi0approx = ITC->phi0_final.fval();
  tapprox = ITC->t_final.fval();
  z0approx = ITC->z0_final.fval();

  phiprojapprox[0] = ITC->phiL_0_final.fval();
  phiprojapprox[1] = ITC->phiL_1_final.fval();
  phiprojapprox[2] = ITC->phiL_2_final.fval();
  phiprojapprox[3] = ITC->phiL_3_final.fval();

  zprojapprox[0] = ITC->zL_0_final.fval();
  zprojapprox[1] = ITC->zL_1_final.fval();
  zprojapprox[2] = ITC->zL_2_final.fval();
  zprojapprox[3] = ITC->zL_3_final.fval();

  phiprojdiskapprox[0] = ITC->phiD_0_final.fval();
  phiprojdiskapprox[1] = ITC->phiD_1_final.fval();
  phiprojdiskapprox[2] = ITC->phiD_2_final.fval();
  phiprojdiskapprox[3] = ITC->phiD_3_final.fval();
  phiprojdiskapprox[4] = ITC->phiD_4_final.fval();

  rprojdiskapprox[0] = ITC->rD_0_final.fval();
  rprojdiskapprox[1] = ITC->rD_1_final.fval();
  rprojdiskapprox[2] = ITC->rD_2_final.fval();
  rprojdiskapprox[3] = ITC->rD_3_final.fval();
  rprojdiskapprox[4] = ITC->rD_4_final.fval();

  //now binary

  int irinv, iphi0, it, iz0;
  Projection projs[N_LAYER + N_DISK];

  int iphiproj[N_LAYER - 2], izproj[N_LAYER - 2];
  int iphiprojdisk[N_DISK], irprojdisk[N_DISK];

  int ir1 = innerFPGAStub->r().value();
  int iphi1 = innerFPGAStub->phi().value();
  int iz1 = innerFPGAStub->z().value();

  int ir2 = outerFPGAStub->r().value();
  int iphi2 = outerFPGAStub->phi().value();
  int iz2 = outerFPGAStub->z().value();

  iphi1 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(layerdisk1_));
  iphi2 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(layerdisk2_));
  ir1 <<= (8 - settings_.nrbitsstub(layerdisk1_));
  ir2 <<= (8 - settings_.nrbitsstub(layerdisk2_));

  iz1 <<= (settings_.nzbitsstub(0) - settings_.nzbitsstub(layerdisk1_));
  iz2 <<= (settings_.nzbitsstub(0) - settings_.nzbitsstub(layerdisk2_));

  ITC->r1.set_ival(ir1);
  ITC->r2.set_ival(ir2);
  ITC->z1.set_ival(iz1);
  ITC->z2.set_ival(iz2);
  ITC->phi1.set_ival(iphi1);
  ITC->phi2.set_ival(iphi2);

  ITC->rinv_final.calculate();
  ITC->phi0_final.calculate();
  ITC->t_final.calculate();
  ITC->z0_final.calculate();

  ITC->phiL_0_final.calculate();
  ITC->phiL_1_final.calculate();
  ITC->phiL_2_final.calculate();
  ITC->phiL_3_final.calculate();

  ITC->zL_0_final.calculate();
  ITC->zL_1_final.calculate();
  ITC->zL_2_final.calculate();
  ITC->zL_3_final.calculate();

  ITC->phiD_0_final.calculate();
  ITC->phiD_1_final.calculate();
  ITC->phiD_2_final.calculate();
  ITC->phiD_3_final.calculate();
  ITC->phiD_4_final.calculate();

  ITC->rD_0_final.calculate();
  ITC->rD_1_final.calculate();
  ITC->rD_2_final.calculate();
  ITC->rD_3_final.calculate();
  ITC->rD_4_final.calculate();

  ITC->der_phiL_final.calculate();
  ITC->der_zL_final.calculate();
  ITC->der_phiD_final.calculate();
  ITC->der_rD_final.calculate();

  //store the binary results
  irinv = ITC->rinv_final.ival();
  iphi0 = ITC->phi0_final.ival();
  it = ITC->t_final.ival();
  iz0 = ITC->z0_final.ival();

  iphiproj[0] = ITC->phiL_0_final.ival();
  iphiproj[1] = ITC->phiL_1_final.ival();
  iphiproj[2] = ITC->phiL_2_final.ival();
  iphiproj[3] = ITC->phiL_3_final.ival();

  izproj[0] = ITC->zL_0_final.ival();
  izproj[1] = ITC->zL_1_final.ival();
  izproj[2] = ITC->zL_2_final.ival();
  izproj[3] = ITC->zL_3_final.ival();

  if (!goodTrackPars(ITC->rinv_final.local_passes(), ITC->z0_final.local_passes())) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " Failed rinv or z0 cut";
    }
    return false;
  }

  if (!inSector(iphi0, irinv, phi0approx, rinvapprox)) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " Failed in sector check";
    }
    return false;
  }

  for (unsigned int i = 0; i < N_LAYER - 2; ++i) {
    //reject projection if z is out of range
    if (izproj[i] < -(1 << (settings_.nzbitsstub(0) - 1)))
      continue;
    if (izproj[i] >= (1 << (settings_.nzbitsstub(0) - 1)))
      continue;

    //reject projection if phi is out of range
    if (iphiproj[i] >= (1 << settings_.nphibitsstub(5)) - 1)
      continue;
    if (iphiproj[i] <= 0)
      continue;

    //Adjust bits for r and z projection depending on layer
    if (settings_.projlayers(iSeed_, i) <= 3) {  //TODO clean up logic
      iphiproj[i] >>= (settings_.nphibitsstub(5) - settings_.nphibitsstub(settings_.projlayers(iSeed_, i) - 1));
    } else {
      izproj[i] >>= (settings_.nzbitsstub(0) - settings_.nzbitsstub(5));
    }

    projs[settings_.projlayers(iSeed_, i) - 1].init(settings_,
                                                    settings_.projlayers(iSeed_, i) - 1,
                                                    iphiproj[i],
                                                    izproj[i],
                                                    ITC->der_phiL_final.ival(),
                                                    ITC->der_zL_final.ival(),
                                                    phiproj[i],
                                                    zproj[i],
                                                    phider[i],
                                                    zder[i],
                                                    phiprojapprox[i],
                                                    zprojapprox[i],
                                                    ITC->der_phiL_final.fval(),
                                                    ITC->der_zL_final.fval(),
                                                    !(iSeed_ == 2 || iSeed_ == 3));
  }

  iphiprojdisk[0] = ITC->phiD_0_final.ival();
  iphiprojdisk[1] = ITC->phiD_1_final.ival();
  iphiprojdisk[2] = ITC->phiD_2_final.ival();
  iphiprojdisk[3] = ITC->phiD_3_final.ival();
  iphiprojdisk[4] = ITC->phiD_4_final.ival();

  irprojdisk[0] = ITC->rD_0_final.ival();
  irprojdisk[1] = ITC->rD_1_final.ival();
  irprojdisk[2] = ITC->rD_2_final.ival();
  irprojdisk[3] = ITC->rD_3_final.ival();
  irprojdisk[4] = ITC->rD_4_final.ival();

  if (std::abs(it * ITC->t_final.K()) > 1.0) {
    for (unsigned int i = 0; i < N_DISK; ++i) {
      if (iphiprojdisk[i] <= 0)
        continue;
      if (iphiprojdisk[i] >= (1 << settings_.nphibitsstub(0)) - 1)
        continue;

      if (irprojdisk[i] < settings_.rmindisk() / ITC->rD_0_final.K() ||
          irprojdisk[i] >= settings_.rmaxdisk() / ITC->rD_0_final.K())
        continue;

      projs[i + N_LAYER].init(settings_,
                              i + N_LAYER,
                              iphiprojdisk[i],
                              irprojdisk[i],
                              ITC->der_phiD_final.ival(),
                              ITC->der_rD_final.ival(),
                              phiprojdisk[i],
                              rprojdisk[i],
                              phiderdisk[i],
                              rderdisk[i],
                              phiprojdiskapprox[i],
                              rprojdiskapprox[i],
                              ITC->der_phiD_final.fval(),
                              ITC->der_rD_final.fval(),
                              !(iSeed_ == 2 || iSeed_ == 3));
    }
  }

  if (settings_.writeMonitorData("TPars")) {
    globals_->ofstream("trackletpars.txt")
        << "Trackpars " << layerdisk1_ + 1 << "   " << rinv << " " << rinvapprox << " " << ITC->rinv_final.fval()
        << "   " << phi0 << " " << phi0approx << " " << ITC->phi0_final.fval() << "   " << t << " " << tapprox << " "
        << ITC->t_final.fval() << "   " << z0 << " " << z0approx << " " << ITC->z0_final.fval() << endl;
  }

  Tracklet* tracklet = new Tracklet(settings_,
                                    iSeed_,
                                    innerFPGAStub,
                                    nullptr,
                                    outerFPGAStub,
                                    rinv,
                                    phi0,
                                    0.0,
                                    z0,
                                    t,
                                    rinvapprox,
                                    phi0approx,
                                    0.0,
                                    z0approx,
                                    tapprox,
                                    irinv,
                                    iphi0,
                                    0,
                                    iz0,
                                    it,
                                    projs,
                                    false);

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "TrackletCalculator " << getName() << " Found tracklet for seed = " << iSeed_ << " "
                                 << iSector_ << " phi0 = " << phi0;
  }

  tracklet->setTrackletIndex(trackletpars_->nTracklets());
  tracklet->setTCIndex(TCIndex_);

  if (settings_.writeMonitorData("Seeds")) {
    ofstream fout("seeds.txt", ofstream::app);
    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
    fout.close();
  }
  trackletpars_->addTracklet(tracklet);

  if (settings_.bookHistos()) {
    HistBase* hists = globals_->histograms();
    int tp = tracklet->tpseed();
    hists->fillTrackletParams(settings_,
                              globals_,
                              iSeed_,
                              iSector_,
                              rinvapprox,
                              irinv * ITC->rinv_final.K(),
                              phi0approx,
                              iphi0 * ITC->phi0_final.K(),
                              asinh(tapprox),
                              asinh(it * ITC->t_final.K()),
                              z0approx,
                              iz0 * ITC->z0_final.K(),
                              tp);
  }

  bool addL3 = false;
  bool addL4 = false;
  bool addL5 = false;
  bool addL6 = false;
  for (unsigned int j = 0; j < N_LAYER - 2; j++) {
    int lproj = settings_.projlayers(iSeed_, j);
    bool added = false;
    if (tracklet->validProj(lproj - 1)) {
      added = addLayerProj(tracklet, lproj);
      if (added && lproj == 3)
        addL3 = true;
      if (added && lproj == 4)
        addL4 = true;
      if (added && lproj == 5)
        addL5 = true;
      if (added && lproj == 6)
        addL6 = true;
    }
  }

  for (unsigned int j = 0; j < N_DISK - 1; j++) {  //no projections to 5th disk!!
    int disk = j + 1;
    if (disk == 4 && addL3)
      continue;
    if (disk == 3 && addL4)
      continue;
    if (disk == 2 && addL5)
      continue;
    if (disk == 1 && addL6)
      continue;
    if (it < 0)
      disk = -disk;
    if (tracklet->validProj(N_LAYER + abs(disk) - 1)) {
      addDiskProj(tracklet, disk);
    }
  }

  return true;
}

bool TrackletCalculatorBase::diskSeeding(const Stub* innerFPGAStub,
                                         const L1TStub* innerStub,
                                         const Stub* outerFPGAStub,
                                         const L1TStub* outerStub) {
  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "TrackletCalculator::execute calculate disk seeds";
  }

  int sign = 1;
  if (innerFPGAStub->disk().value() < 0)
    sign = -1;

  int disk = innerFPGAStub->disk().value();
  assert(abs(disk) == 1 || abs(disk) == 3);

  assert(innerStub->isPSmodule());
  assert(outerStub->isPSmodule());

  double r1 = innerStub->r();
  double z1 = innerStub->z();
  double phi1 = innerStub->phi();

  double r2 = outerStub->r();
  double z2 = outerStub->z();
  double phi2 = outerStub->phi();

  if (r2 < r1 + 2.0) {
    return false;  //Protection... Should be handled cleaner to avoid problem with floating point calculation
  }

  double rinv, phi0, t, z0;

  double phiproj[N_PSLAYER], zproj[N_PSLAYER], phider[N_PSLAYER], zder[N_PSLAYER];
  double phiprojdisk[N_DISK - 2], rprojdisk[N_DISK - 2], phiderdisk[N_DISK - 2], rderdisk[N_DISK - 2];

  exacttrackletdisk(r1,
                    z1,
                    phi1,
                    r2,
                    z2,
                    phi2,
                    outerStub->sigmaz(),
                    rinv,
                    phi0,
                    t,
                    z0,
                    phiproj,
                    zproj,
                    phider,
                    zder,
                    phiprojdisk,
                    rprojdisk,
                    phiderdisk,
                    rderdisk);

  //Truncates floating point positions to integer representation precision
  if (settings_.useapprox()) {
    phi1 = innerFPGAStub->phiapprox(phimin_, phimax_);
    z1 = innerFPGAStub->zapprox();
    r1 = innerFPGAStub->rapprox();

    phi2 = outerFPGAStub->phiapprox(phimin_, phimax_);
    z2 = outerFPGAStub->zapprox();
    r2 = outerFPGAStub->rapprox();
  }

  double rinvapprox, phi0approx, tapprox, z0approx;
  double phiprojapprox[N_PSLAYER], zprojapprox[N_PSLAYER];
  double phiprojdiskapprox[N_DISK - 2], rprojdiskapprox[N_DISK - 2];

  IMATH_TrackletCalculatorDisk* ITC;
  if (disk == 1)
    ITC = globals_->ITC_F1F2();
  else if (disk == 3)
    ITC = globals_->ITC_F3F4();
  else if (disk == -1)
    ITC = globals_->ITC_B1B2();
  else
    ITC = globals_->ITC_B3B4();

  ITC->r1.set_fval(r1);
  ITC->r2.set_fval(r2);
  int signt = t > 0 ? 1 : -1;
  ITC->z1.set_fval(z1 - signt * settings_.zmean(layerdisk1_ - N_LAYER));
  ITC->z2.set_fval(z2 - signt * settings_.zmean(layerdisk2_ - N_LAYER));
  double sphi1 = angle0to2pi::make0To2pi(phi1 - phimin_);
  double sphi2 = angle0to2pi::make0To2pi(phi2 - phimin_);
  ITC->phi1.set_fval(sphi1);
  ITC->phi2.set_fval(sphi2);

  ITC->rproj0.set_fval(settings_.rmean(0));
  ITC->rproj1.set_fval(settings_.rmean(1));
  ITC->rproj2.set_fval(settings_.rmean(2));

  ITC->zproj0.set_fval(signt * settings_.zmean(settings_.projdisks(iSeed_, 0) - 1));
  ITC->zproj1.set_fval(signt * settings_.zmean(settings_.projdisks(iSeed_, 1) - 1));
  ITC->zproj2.set_fval(signt * settings_.zmean(settings_.projdisks(iSeed_, 2) - 1));

  ITC->rinv_final.calculate();
  ITC->phi0_final.calculate();
  ITC->t_final.calculate();
  ITC->z0_final.calculate();

  ITC->phiL_0_final.calculate();
  ITC->phiL_1_final.calculate();
  ITC->phiL_2_final.calculate();

  ITC->zL_0_final.calculate();
  ITC->zL_1_final.calculate();
  ITC->zL_2_final.calculate();

  ITC->phiD_0_final.calculate();
  ITC->phiD_1_final.calculate();
  ITC->phiD_2_final.calculate();

  ITC->rD_0_final.calculate();
  ITC->rD_1_final.calculate();
  ITC->rD_2_final.calculate();

  ITC->der_phiL_final.calculate();
  ITC->der_zL_final.calculate();
  ITC->der_phiD_final.calculate();
  ITC->der_rD_final.calculate();

  //store the approximate results
  rinvapprox = ITC->rinv_final.fval();
  phi0approx = ITC->phi0_final.fval();
  tapprox = ITC->t_final.fval();
  z0approx = ITC->z0_final.fval();

  phiprojapprox[0] = ITC->phiL_0_final.fval();
  phiprojapprox[1] = ITC->phiL_1_final.fval();
  phiprojapprox[2] = ITC->phiL_2_final.fval();

  zprojapprox[0] = ITC->zL_0_final.fval();
  zprojapprox[1] = ITC->zL_1_final.fval();
  zprojapprox[2] = ITC->zL_2_final.fval();

  phiprojdiskapprox[0] = ITC->phiD_0_final.fval();
  phiprojdiskapprox[1] = ITC->phiD_1_final.fval();
  phiprojdiskapprox[2] = ITC->phiD_2_final.fval();

  rprojdiskapprox[0] = ITC->rD_0_final.fval();
  rprojdiskapprox[1] = ITC->rD_1_final.fval();
  rprojdiskapprox[2] = ITC->rD_2_final.fval();

  //now binary

  int irinv, iphi0, it, iz0;
  int iphiproj[N_PSLAYER], izproj[N_PSLAYER];

  int iphiprojdisk[N_DISK - 2], irprojdisk[N_DISK - 2];

  int ir1 = innerFPGAStub->r().value();
  int iphi1 = innerFPGAStub->phi().value();
  int iz1 = innerFPGAStub->z().value();

  int ir2 = outerFPGAStub->r().value();
  int iphi2 = outerFPGAStub->phi().value();
  int iz2 = outerFPGAStub->z().value();

  //To get same precission as for layers.
  iphi1 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));
  iphi2 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));

  ITC->r1.set_ival(ir1);
  ITC->r2.set_ival(ir2);
  ITC->z1.set_ival(iz1);
  ITC->z2.set_ival(iz2);
  ITC->phi1.set_ival(iphi1);
  ITC->phi2.set_ival(iphi2);

  ITC->rinv_final.calculate();
  ITC->phi0_final.calculate();
  ITC->t_final.calculate();
  ITC->z0_final.calculate();

  ITC->phiL_0_final.calculate();
  ITC->phiL_1_final.calculate();
  ITC->phiL_2_final.calculate();

  ITC->zL_0_final.calculate();
  ITC->zL_1_final.calculate();
  ITC->zL_2_final.calculate();

  ITC->phiD_0_final.calculate();
  ITC->phiD_1_final.calculate();
  ITC->phiD_2_final.calculate();

  ITC->rD_0_final.calculate();
  ITC->rD_1_final.calculate();
  ITC->rD_2_final.calculate();

  ITC->der_phiL_final.calculate();
  ITC->der_zL_final.calculate();
  ITC->der_phiD_final.calculate();
  ITC->der_rD_final.calculate();

  //store the binary results
  irinv = ITC->rinv_final.ival();
  iphi0 = ITC->phi0_final.ival();
  it = ITC->t_final.ival();
  iz0 = ITC->z0_final.ival();

  iphiproj[0] = ITC->phiL_0_final.ival();
  iphiproj[1] = ITC->phiL_1_final.ival();
  iphiproj[2] = ITC->phiL_2_final.ival();

  izproj[0] = ITC->zL_0_final.ival();
  izproj[1] = ITC->zL_1_final.ival();
  izproj[2] = ITC->zL_2_final.ival();

  if (!goodTrackPars(ITC->rinv_final.local_passes(), ITC->z0_final.local_passes()))
    return false;

  if (!inSector(iphi0, irinv, phi0approx, rinvapprox))
    return false;

  Projection projs[N_LAYER + N_DISK];

  for (unsigned int i = 0; i < N_DISK - 2; ++i) {
    //Check is outside z range
    if (izproj[i] < -(1 << (settings_.nzbitsstub(0) - 1)))
      continue;
    if (izproj[i] >= (1 << (settings_.nzbitsstub(0) - 1)))
      continue;

    //Check if outside phi range
    if (iphiproj[i] >= (1 << settings_.nphibitsstub(5)) - 1)
      continue;
    if (iphiproj[i] <= 0)
      continue;

    //shift bits - allways in PS modules for disk seeding
    iphiproj[i] >>= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));

    projs[i].init(settings_,
                  i,
                  iphiproj[i],
                  izproj[i],
                  ITC->der_phiL_final.ival(),
                  ITC->der_zL_final.ival(),
                  phiproj[i],
                  zproj[i],
                  phider[i],
                  zder[i],
                  phiprojapprox[i],
                  zprojapprox[i],
                  ITC->der_phiL_final.fval(),
                  ITC->der_zL_final.fval(),
                  true);
  }

  iphiprojdisk[0] = ITC->phiD_0_final.ival();
  iphiprojdisk[1] = ITC->phiD_1_final.ival();
  iphiprojdisk[2] = ITC->phiD_2_final.ival();

  irprojdisk[0] = ITC->rD_0_final.ival();
  irprojdisk[1] = ITC->rD_1_final.ival();
  irprojdisk[2] = ITC->rD_2_final.ival();

  for (unsigned int i = 0; i < N_DISK - 2; ++i) {
    //check that phi projection in range
    if (iphiprojdisk[i] <= 0)
      continue;
    if (iphiprojdisk[i] >= (1 << settings_.nphibitsstub(0)) - 1)
      continue;

    //check that r projection in range
    if (irprojdisk[i] <= 0 || irprojdisk[i] >= settings_.rmaxdisk() / ITC->rD_0_final.K())
      continue;

    projs[settings_.projdisks(iSeed_, i) + N_LAYER - 1].init(settings_,
                                                             settings_.projdisks(iSeed_, i) + N_LAYER - 1,
                                                             iphiprojdisk[i],
                                                             irprojdisk[i],
                                                             ITC->der_phiD_final.ival(),
                                                             ITC->der_rD_final.ival(),
                                                             phiprojdisk[i],
                                                             rprojdisk[i],
                                                             phiderdisk[i],
                                                             rderdisk[i],
                                                             phiprojdiskapprox[i],
                                                             rprojdiskapprox[i],
                                                             ITC->der_phiD_final.fval(),
                                                             ITC->der_rD_final.fval(),
                                                             true);
  }

  if (settings_.writeMonitorData("TPars")) {
    globals_->ofstream("trackletparsdisk.txt")
        << "Trackpars         " << layerdisk1_ - 5 << "   " << rinv << " " << rinvapprox << " "
        << ITC->rinv_final.fval() << "   " << phi0 << " " << phi0approx << " " << ITC->phi0_final.fval() << "   " << t
        << " " << tapprox << " " << ITC->t_final.fval() << "   " << z0 << " " << z0approx << " " << ITC->z0_final.fval()
        << endl;
  }

  Tracklet* tracklet = new Tracklet(settings_,
                                    iSeed_,
                                    innerFPGAStub,
                                    nullptr,
                                    outerFPGAStub,
                                    rinv,
                                    phi0,
                                    0.0,
                                    z0,
                                    t,
                                    rinvapprox,
                                    phi0approx,
                                    0.0,
                                    z0approx,
                                    tapprox,
                                    irinv,
                                    iphi0,
                                    0,
                                    iz0,
                                    it,
                                    projs,
                                    true);

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "Found tracklet for disk seed = " << iSeed_ << " " << tracklet << " " << iSector_;
  }

  tracklet->setTrackletIndex(trackletpars_->nTracklets());
  tracklet->setTCIndex(TCIndex_);

  if (settings_.writeMonitorData("Seeds")) {
    ofstream fout("seeds.txt", ofstream::app);
    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
    fout.close();
  }
  trackletpars_->addTracklet(tracklet);

  if (tracklet->validProj(0)) {
    addLayerProj(tracklet, 1);
  }

  if (tracklet->validProj(1)) {
    addLayerProj(tracklet, 2);
  }

  for (unsigned int j = 0; j < N_DISK - 2; j++) {
    if (tracklet->validProj(N_LAYER + settings_.projdisks(iSeed_, j) - 1)) {
      addDiskProj(tracklet, sign * settings_.projdisks(iSeed_, j));
    }
  }

  return true;
}

bool TrackletCalculatorBase::overlapSeeding(const Stub* innerFPGAStub,
                                            const L1TStub* innerStub,
                                            const Stub* outerFPGAStub,
                                            const L1TStub* outerStub) {
  //Deal with overlap stubs here
  assert(outerFPGAStub->layerdisk() < N_LAYER);

  assert(innerFPGAStub->layerdisk() >= N_LAYER);

  int disk = innerFPGAStub->disk().value();

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "trying to make overlap tracklet for seed = " << iSeed_ << " " << getName();
  }

  double r1 = innerStub->r();
  double z1 = innerStub->z();
  double phi1 = innerStub->phi();

  double r2 = outerStub->r();
  double z2 = outerStub->z();
  double phi2 = outerStub->phi();

  //Protection for wrong radii. Could be handled cleaner to avoid problem with floating point calculation and with overflows in the integer calculation.
  if (r1 < r2 + 1.5) {
    return false;
  }

  double rinv, phi0, t, z0;

  double phiproj[N_PSLAYER], zproj[N_PSLAYER], phider[N_PSLAYER], zder[N_PSLAYER];
  double phiprojdisk[N_DISK - 1], rprojdisk[N_DISK - 1], phiderdisk[N_DISK - 1], rderdisk[N_DISK - 1];

  exacttrackletOverlap(r1,
                       z1,
                       phi1,
                       r2,
                       z2,
                       phi2,
                       outerStub->sigmaz(),
                       rinv,
                       phi0,
                       t,
                       z0,
                       phiproj,
                       zproj,
                       phider,
                       zder,
                       phiprojdisk,
                       rprojdisk,
                       phiderdisk,
                       rderdisk);

  //Truncates floating point positions to integer representation precision
  if (settings_.useapprox()) {
    phi1 = innerFPGAStub->phiapprox(phimin_, phimax_);
    z1 = innerFPGAStub->zapprox();
    r1 = innerFPGAStub->rapprox();

    phi2 = outerFPGAStub->phiapprox(phimin_, phimax_);
    z2 = outerFPGAStub->zapprox();
    r2 = outerFPGAStub->rapprox();
  }

  double rinvapprox, phi0approx, tapprox, z0approx;
  double phiprojapprox[N_PSLAYER], zprojapprox[N_PSLAYER];
  double phiprojdiskapprox[N_DISK - 1], rprojdiskapprox[N_DISK - 1];

  IMATH_TrackletCalculatorOverlap* ITC;
  int ll = outerFPGAStub->layer().value() + 1;
  if (ll == 1 && disk == 1)
    ITC = globals_->ITC_L1F1();
  else if (ll == 2 && disk == 1)
    ITC = globals_->ITC_L2F1();
  else if (ll == 1 && disk == -1)
    ITC = globals_->ITC_L1B1();
  else if (ll == 2 && disk == -1)
    ITC = globals_->ITC_L2B1();
  else
    throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " Invalid seeding!";

  ITC->r1.set_fval(r2 - settings_.rmean(ll - 1));
  ITC->r2.set_fval(r1);
  int signt = t > 0 ? 1 : -1;
  ITC->z1.set_fval(z2);
  ITC->z2.set_fval(z1 - signt * settings_.zmean(layerdisk2_ - N_LAYER));
  double sphi1 = angle0to2pi::make0To2pi(phi1 - phimin_);
  double sphi2 = angle0to2pi::make0To2pi(phi2 - phimin_);
  ITC->phi1.set_fval(sphi2);
  ITC->phi2.set_fval(sphi1);

  ITC->rproj0.set_fval(settings_.rmean(0));
  ITC->rproj1.set_fval(settings_.rmean(1));
  ITC->rproj2.set_fval(settings_.rmean(2));

  ITC->zproj0.set_fval(signt * settings_.zmean(1));
  ITC->zproj1.set_fval(signt * settings_.zmean(2));
  ITC->zproj2.set_fval(signt * settings_.zmean(3));
  ITC->zproj3.set_fval(signt * settings_.zmean(4));

  ITC->rinv_final.calculate();
  ITC->phi0_final.calculate();
  ITC->t_final.calculate();
  ITC->z0_final.calculate();

  ITC->phiL_0_final.calculate();
  ITC->phiL_1_final.calculate();
  ITC->phiL_2_final.calculate();

  ITC->zL_0_final.calculate();
  ITC->zL_1_final.calculate();
  ITC->zL_2_final.calculate();

  ITC->phiD_0_final.calculate();
  ITC->phiD_1_final.calculate();
  ITC->phiD_2_final.calculate();
  ITC->phiD_3_final.calculate();

  ITC->rD_0_final.calculate();
  ITC->rD_1_final.calculate();
  ITC->rD_2_final.calculate();
  ITC->rD_3_final.calculate();

  ITC->der_phiL_final.calculate();
  ITC->der_zL_final.calculate();
  ITC->der_phiD_final.calculate();
  ITC->der_rD_final.calculate();

  //store the approximate results
  rinvapprox = ITC->rinv_final.fval();
  phi0approx = ITC->phi0_final.fval();
  tapprox = ITC->t_final.fval();
  z0approx = ITC->z0_final.fval();

  phiprojapprox[0] = ITC->phiL_0_final.fval();
  phiprojapprox[1] = ITC->phiL_1_final.fval();
  phiprojapprox[2] = ITC->phiL_2_final.fval();

  zprojapprox[0] = ITC->zL_0_final.fval();
  zprojapprox[1] = ITC->zL_1_final.fval();
  zprojapprox[2] = ITC->zL_2_final.fval();

  phiprojdiskapprox[0] = ITC->phiD_0_final.fval();
  phiprojdiskapprox[1] = ITC->phiD_1_final.fval();
  phiprojdiskapprox[2] = ITC->phiD_2_final.fval();
  phiprojdiskapprox[3] = ITC->phiD_3_final.fval();

  rprojdiskapprox[0] = ITC->rD_0_final.fval();
  rprojdiskapprox[1] = ITC->rD_1_final.fval();
  rprojdiskapprox[2] = ITC->rD_2_final.fval();
  rprojdiskapprox[3] = ITC->rD_3_final.fval();

  //now binary

  int irinv, iphi0, it, iz0;
  int iphiproj[N_LAYER - 2], izproj[N_LAYER - 2];
  int iphiprojdisk[N_DISK], irprojdisk[N_DISK];

  int ir2 = innerFPGAStub->r().value();
  int iphi2 = innerFPGAStub->phi().value();
  int iz2 = innerFPGAStub->z().value();

  int ir1 = outerFPGAStub->r().value();
  int iphi1 = outerFPGAStub->phi().value();
  int iz1 = outerFPGAStub->z().value();

  //To get global precission
  ir1 = l1t::bitShift(ir1, (8 - settings_.nrbitsstub(ll - 1)));
  iphi1 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));
  iphi2 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));

  ITC->r1.set_ival(ir1);
  ITC->r2.set_ival(ir2);
  ITC->z1.set_ival(iz1);
  ITC->z2.set_ival(iz2);
  ITC->phi1.set_ival(iphi1);
  ITC->phi2.set_ival(iphi2);

  ITC->rinv_final.calculate();
  ITC->phi0_final.calculate();
  ITC->t_final.calculate();
  ITC->z0_final.calculate();

  ITC->phiL_0_final.calculate();
  ITC->phiL_1_final.calculate();
  ITC->phiL_2_final.calculate();

  ITC->zL_0_final.calculate();
  ITC->zL_1_final.calculate();
  ITC->zL_2_final.calculate();

  ITC->phiD_0_final.calculate();
  ITC->phiD_1_final.calculate();
  ITC->phiD_2_final.calculate();
  ITC->phiD_3_final.calculate();

  ITC->rD_0_final.calculate();
  ITC->rD_1_final.calculate();
  ITC->rD_2_final.calculate();
  ITC->rD_3_final.calculate();

  ITC->der_phiL_final.calculate();
  ITC->der_zL_final.calculate();
  ITC->der_phiD_final.calculate();
  ITC->der_rD_final.calculate();

  //store the binary results
  irinv = ITC->rinv_final.ival();
  iphi0 = ITC->phi0_final.ival();
  it = ITC->t_final.ival();
  iz0 = ITC->z0_final.ival();

  iphiproj[0] = ITC->phiL_0_final.ival();
  iphiproj[1] = ITC->phiL_1_final.ival();
  iphiproj[2] = ITC->phiL_2_final.ival();

  izproj[0] = ITC->zL_0_final.ival();
  izproj[1] = ITC->zL_1_final.ival();
  izproj[2] = ITC->zL_2_final.ival();

  iphiprojdisk[0] = ITC->phiD_0_final.ival();
  iphiprojdisk[1] = ITC->phiD_1_final.ival();
  iphiprojdisk[2] = ITC->phiD_2_final.ival();
  iphiprojdisk[3] = ITC->phiD_3_final.ival();

  irprojdisk[0] = ITC->rD_0_final.ival();
  irprojdisk[1] = ITC->rD_1_final.ival();
  irprojdisk[2] = ITC->rD_2_final.ival();
  irprojdisk[3] = ITC->rD_3_final.ival();

  if (!goodTrackPars(ITC->rinv_final.local_passes(), ITC->z0_final.local_passes()))
    return false;

  if (!inSector(iphi0, irinv, phi0approx, rinvapprox))
    return false;

  Projection projs[N_LAYER + N_DISK];

  for (unsigned int i = 0; i < N_DISK - 2; ++i) {
    //check that zproj is in range
    if (izproj[i] < -(1 << (settings_.nzbitsstub(0) - 1)))
      continue;
    if (izproj[i] >= (1 << (settings_.nzbitsstub(0) - 1)))
      continue;

    //check that phiproj is in range
    if (iphiproj[i] >= (1 << settings_.nphibitsstub(5)) - 1)
      continue;
    if (iphiproj[i] <= 0)
      continue;

    //adjust bits for PS modules (no 2S modules in overlap seeds)
    iphiproj[i] >>= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));

    projs[i].init(settings_,
                  i,
                  iphiproj[i],
                  izproj[i],
                  ITC->der_phiL_final.ival(),
                  ITC->der_zL_final.ival(),
                  phiproj[i],
                  zproj[i],
                  phider[i],
                  zder[i],
                  phiprojapprox[i],
                  zprojapprox[i],
                  ITC->der_phiL_final.fval(),
                  ITC->der_zL_final.fval(),
                  true);
  }

  for (int i = 0; i < 4; ++i) {
    //check that phi projection in range
    if (iphiprojdisk[i] <= 0)
      continue;
    if (iphiprojdisk[i] >= (1 << settings_.nphibitsstub(0)) - 1)
      continue;

    //check that r projection in range
    if (irprojdisk[i] <= 0 || irprojdisk[i] >= settings_.rmaxdisk() / ITC->rD_0_final.K())
      continue;

    projs[N_LAYER + i + 1].init(settings_,
                                N_LAYER + i + 1,
                                iphiprojdisk[i],
                                irprojdisk[i],
                                ITC->der_phiD_final.ival(),
                                ITC->der_rD_final.ival(),
                                phiprojdisk[i],
                                rprojdisk[i],
                                phiderdisk[i],
                                rderdisk[i],
                                phiprojdiskapprox[i],
                                rprojdiskapprox[i],
                                ITC->der_phiD_final.fval(),
                                ITC->der_rD_final.fval(),
                                true);
  }

  if (settings_.writeMonitorData("TPars")) {
    globals_->ofstream("trackletparsoverlap.txt")
        << "Trackpars " << layerdisk1_ - 5 << "   " << rinv << " " << irinv << " " << ITC->rinv_final.fval() << "   "
        << phi0 << " " << iphi0 << " " << ITC->phi0_final.fval() << "   " << t << " " << it << " "
        << ITC->t_final.fval() << "   " << z0 << " " << iz0 << " " << ITC->z0_final.fval() << endl;
  }

  Tracklet* tracklet = new Tracklet(settings_,
                                    iSeed_,
                                    innerFPGAStub,
                                    nullptr,
                                    outerFPGAStub,
                                    rinv,
                                    phi0,
                                    0.0,
                                    z0,
                                    t,
                                    rinvapprox,
                                    phi0approx,
                                    0.0,
                                    z0approx,
                                    tapprox,
                                    irinv,
                                    iphi0,
                                    0,
                                    iz0,
                                    it,
                                    projs,
                                    false,
                                    true);

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "Found tracklet in overlap seed = " << iSeed_ << " " << tracklet << " " << iSector_;
  }

  tracklet->setTrackletIndex(trackletpars_->nTracklets());
  tracklet->setTCIndex(TCIndex_);

  if (settings_.writeMonitorData("Seeds")) {
    ofstream fout("seeds.txt", ofstream::app);
    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
    fout.close();
  }
  trackletpars_->addTracklet(tracklet);

  int layer = outerFPGAStub->layer().value() + 1;

  if (layer == 2) {
    if (tracklet->validProj(0)) {
      addLayerProj(tracklet, 1);
    }
  }

  for (unsigned int disk = 2; disk < 6; disk++) {
    if (layer == 2 && disk == 5)
      continue;
    if (tracklet->validProj(N_LAYER + disk - 1)) {
      addDiskProj(tracklet, disk);
    }
  }

  return true;
}
