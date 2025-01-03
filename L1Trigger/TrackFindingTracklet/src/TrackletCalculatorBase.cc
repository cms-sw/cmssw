#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/L1TCommon/interface/BitShift.h"

using namespace std;
using namespace trklet;

TrackletCalculatorBase::TrackletCalculatorBase(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global) {}

void TrackletCalculatorBase::init(int iSeed) {
  phiHG_ = settings_.dphisectorHG();

  //Constants for coordinates and track parameter definitions
  n_phi_ = 17;
  n_r_ = 12;
  n_z_ = 11;
  n_phi0_ = 16;
  n_rinv_ = 13;
  n_t_ = 9;
  n_phidisk_ = n_phi_ - 3;
  n_rdisk_ = n_r_ - 1;

  //Constants used for tracklet parameter calculations
  n_delta0_ = 13;
  n_deltaz_ = 11;
  n_delta1_ = 13;
  n_delta2_ = 13;
  n_delta12_ = 13;
  n_a_ = 15;
  n_r6_ = 4;
  n_delta02_ = 14;
  n_x6_ = 15;
  n_HG_ = 15;

  //Constants used for projectison to layers
  n_s_ = 12;
  n_s6_ = 14;

  //Constants used for projectison to disks
  n_tinv_ = 12;
  n_y_ = 14;
  n_x_ = 14;
  n_xx6_ = 14;

  LUT_itinv_.resize(8192);

  for (int it = 0; it < 8192; it++) {
    if (it < 100) {
      LUT_itinv_[it] = 0;
    } else {
      LUT_itinv_[it] = (1 << (n_t_ + n_tinv_)) / abs(it);
    }
  }

  if (iSeed < 4) {  //FIXME - should not have hardcoded number here
    n_Deltar_ = 24;
    LUT_idrinv_.resize(512);
    for (int idr = -256; idr < 256; idr++) {
      int uidr = idr;
      if (uidr < 0)
        uidr += 512;
      int idrabs = idr + settings_.irmean(layerdisk2_) - settings_.irmean(layerdisk1_);
      LUT_idrinv_[uidr] = (1 << n_Deltar_) / idrabs;
    }
  }

  if (iSeed >= 4 && iSeed < 6) {  //FIXME - should not have hardcoded number here
    n_Deltar_ = 23;
    LUT_idrinv_.resize(512);
    for (unsigned int idr = 1; idr < 512; idr++) {
      LUT_idrinv_[idr] = (1 << n_Deltar_) / idr;
    }
  }

  if (iSeed >= 6) {  //FIXME - should not have hardcoded number here
    n_Deltar_ = 23;
    n_delta0_ = 14;
    n_deltaz_ = 9;
    n_a_ = 14;
    n_r6_ = 6;
    LUT_idrinv_.resize(1024);
    for (unsigned int idr = 1; idr < 1024; idr++) {
      LUT_idrinv_[idr] = (1 << n_Deltar_) / idr;
    }
  }
}

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

//project to layer
void TrackletCalculatorBase::projlayer(int ir, int irinv, int iphi0, int it, int iz0, int& iz, int& iphi) {
  int irtilde = ir * phiHG_ / sqrt(6.0) + 0.5;

  int is = (irtilde * irinv) >> n_s_;

  int is6 = (1 << n_s6_) + ((is * is) >> (2 + 2 * n_r_ + 2 * n_rinv_ - 2 * n_s_ - n_s6_));

  int iu = (ir * irinv) >> (n_r_ + n_rinv_ + 1 - n_phi_);

  iphi = (iphi0 << (n_phi_ - n_phi0_)) - ((iu * is6) >> n_s6_);

  int iv = (it * ir) >> (n_r_ + n_t_ - n_z_);

  iz = iz0 + ((iv * is6) >> n_s6_);
}

void TrackletCalculatorBase::projdisk(
    int iz, int irinv, int iphi0, int it, int iz0, int& ir, int& iphi, int& iderphi, int& iderr) {
  int iz0_sign = (it > 0) ? iz0 : -iz0;

  assert(abs(it) < LUT_itinv_.size());
  int itinv = LUT_itinv_[abs(it)];

  iderphi = (-irinv * itinv) >> 17;
  iderr = itinv >> 5;
  if (it < 0) {
    iderphi = -iderphi;
    iderr = -iderr;
  }

  int iw = (((iz << (n_r_ - n_z_)) - (iz0_sign << (n_r_ - n_z_))) * itinv) >> n_tinv_;

  iphi = (iphi0 >> (n_phi0_ - n_phidisk_)) - ((iw * irinv) >> (1 + n_r_ + n_rinv_ - n_phidisk_));

  int ifact = (1 << n_y_) * phiHG_ / sqrt(6.0);

  int iy = (ifact * irinv) >> n_y_;

  int ix = (iw * iy) >> n_x_;

  int ix6 = (1 << n_xx6_) - ((ix * ix) >> (2 + 2 * n_r_ + 2 * n_rinv_ - 2 * n_x_ - n_xx6_));

  ir = (iw * ix6) >> (n_r_ - n_rdisk_ + n_xx6_);
}

void TrackletCalculatorBase::calcPars(unsigned int idr,
                                      int iphi1,
                                      int ir1,
                                      int iz1,
                                      int iphi2,
                                      int ir2,
                                      int iz2,
                                      int& irinv_new,
                                      int& iphi0_new,
                                      int& iz0_new,
                                      int& it_new) {
  int idz = iz2 - iz1;

  assert(idr < LUT_idrinv_.size());
  int invdr = LUT_idrinv_[idr];

  int idelta0 = ((iphi2 - iphi1) * invdr) >> n_delta0_;
  int ideltaz = (idz * invdr) >> n_deltaz_;

  int idelta1 = (ir1 * idelta0) >> n_delta1_;
  int idelta2 = (ir2 * idelta0) >> n_delta2_;

  int idelta12 = (idelta1 * idelta2) >> n_delta12_;

  int iHG = phiHG_ * phiHG_ * (1 << n_HG_);

  int ia = ((1 << n_a_) - ((idelta12 * iHG) >> (2 * n_Deltar_ + 2 * n_phi_ + n_HG_ - 2 * n_delta0_ - n_delta1_ -
                                                n_delta2_ - n_delta12_ + 1 - n_a_)));

  int ifact = (1 << n_r6_) * phiHG_ * phiHG_ / 6.0;

  int ir6 = (ir1 + ir2) * ifact;

  int idelta02 = (idelta0 * idelta2) >> n_delta02_;

  int ix6 = (-(1 << n_x6_) + ((ir6 * idelta02) >>
                              (n_r6_ + 2 * n_Deltar_ + 2 * n_phi_ - n_x6_ - n_delta2_ - n_delta02_ - 2 * n_delta0_)));

  int it1 = (ir1 * ideltaz) >> (n_Deltar_ - n_deltaz_);

  irinv_new = ((-idelta0 * ia) >> (n_phi_ + n_a_ - n_rinv_ + n_Deltar_ - n_delta0_ - n_r_ - 1));

  iphi0_new = (iphi1 >> (n_phi_ - n_phi0_)) +
              ((idelta1 * ix6) >> (n_Deltar_ + n_x6_ + n_phi_ - n_delta0_ - n_delta1_ - n_phi0_));

  it_new = ((ideltaz * ia) >> (n_Deltar_ + n_a_ + n_z_ - n_t_ - n_deltaz_ - n_r_));

  iz0_new = iz1 + ((it1 * ix6) >> n_x6_);
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

  int iphicritmincut = settings_.phicritminmc() / settings_.kphi0pars();
  int iphicritmaxcut = settings_.phicritmaxmc() / settings_.kphi0pars();

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

  //now binary

  int ir1 = innerFPGAStub->r().value();
  int iphi1 = innerFPGAStub->phi().value();
  int iz1 = innerFPGAStub->z().value();

  int ir2 = outerFPGAStub->rvalue();
  int iphi2 = outerFPGAStub->phi().value();
  int iz2 = outerFPGAStub->z().value();

  iphi1 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(layerdisk1_));
  iphi2 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(layerdisk2_));
  ir1 <<= (8 - settings_.nrbitsstub(layerdisk1_));
  ir2 <<= (8 - settings_.nrbitsstub(layerdisk2_));

  iz1 <<= (settings_.nzbitsstub(0) - settings_.nzbitsstub(layerdisk1_));
  iz2 <<= (settings_.nzbitsstub(0) - settings_.nzbitsstub(layerdisk2_));

  //Each of ir1 and ir2 are signed 8 bit integers. idr is signed 9 bit integer
  int idr = ir2 - ir1;

  if (idr < 0)
    idr += 512;

  int irinv_new, iphi0_new, iz0_new, it_new;

  unsigned int ir1mean = settings_.irmean(layerdisk1_);
  unsigned int ir2mean = settings_.irmean(layerdisk2_);

  int ir1abs = ir1 + ir1mean;
  int ir2abs = ir2 + ir2mean;

  calcPars(idr, iphi1, ir1abs, iz1, iphi2, ir2abs, iz2, irinv_new, iphi0_new, iz0_new, it_new);

  bool rinvcut = abs(irinv_new) < settings_.rinvcut() * (120.0 * (1 << n_rinv_)) / phiHG_;
  bool z0cut = abs(iz0_new) < settings_.z0cut() * (1 << n_z_) / 120.0;
  if (iSeed_ != 0) {
    z0cut = abs(iz0_new) < 1.5 * settings_.z0cut() * (1 << n_z_) / 120.0;
  }

  if (!goodTrackPars(rinvcut, z0cut)) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " Failed rinv or z0 cut";
    }
    return false;
  }

  if (!inSector(iphi0_new, irinv_new, phi0, rinv)) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " Failed in sector check";
    }
    return false;
  }

  if (settings_.writeMonitorData("TPars")) {
    globals_->ofstream("trackletpars.txt")
        << "Trackpars " << layerdisk1_ + 1 << "   "
        << " " << rinv << " " << irinv_new << "   " << phi0 << " " << iphi0_new << "   " << t << " " << it_new << "   "
        << " " << iz0_new << endl;
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
                                    rinv,
                                    phi0,
                                    0.0,
                                    z0,
                                    t,
                                    irinv_new,
                                    iphi0_new,
                                    0,
                                    iz0_new,
                                    it_new,
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
                              rinv,
                              irinv_new * settings_.krinvpars(),
                              phi0,
                              iphi0_new * settings_.kphi0pars(),
                              asinh(t),
                              asinh(it_new * settings_.ktpars()),
                              z0,
                              iz0_new * settings_.kz0pars(),
                              tp);
  }

  return true;
}

bool TrackletCalculatorBase::diskSeeding(const Stub* innerFPGAStub,
                                         const L1TStub* innerStub,
                                         const Stub* outerFPGAStub,
                                         const L1TStub* outerStub,
                                         bool print) {
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

  //now binary

  int ir1 = innerFPGAStub->rvalue();
  int iphi1 = innerFPGAStub->phi().value();
  int iz1 = innerFPGAStub->z().value();

  int ir2 = outerFPGAStub->rvalue();
  int iphi2 = outerFPGAStub->phi().value();
  int iz2 = outerFPGAStub->z().value();

  //To get same precission as for layers.
  iphi1 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));
  iphi2 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));

  //Each of ir1 and ir2 are signed 8 bit integers. idr is signed 9 bit integer
  unsigned int idr = ir2 - ir1;

  unsigned int iz1mean = sign * settings_.izmean(layerdisk1_ - N_LAYER);
  unsigned int iz2mean = sign * settings_.izmean(layerdisk2_ - N_LAYER);

  int iz1abs = iz1 + iz1mean;
  int iz2abs = iz2 + iz2mean;

  int irinv_new, iphi0_new, iz0_new, it_new;

  calcPars(idr, iphi1, ir1, iz1abs, iphi2, ir2, iz2abs, irinv_new, iphi0_new, iz0_new, it_new);

  if (print) {
    std::cout << "=======================" << std::endl;
    std::cout << "iphi1, ir1, iz1abs : " << iphi1 << " " << ir1 << " " << iz1abs << std::endl;
    std::cout << "iphi2, ir2, iz2abs : " << iphi2 << " " << ir2 << " " << iz2abs << std::endl;
    std::cout << "idr irinv iphi0 iz0 it : " << idr << " " << irinv_new << " " << iphi0_new << " " << iz0_new << " "
              << it_new << std::endl;
  }

  bool rinvcut = abs(irinv_new) < settings_.rinvcut() * (120.0 * (1 << n_rinv_)) / phiHG_;
  bool z0cut = abs(iz0_new) < settings_.z0cut() * (1 << n_z_) / 120.0;

  if (print) {
    std::cout << "Pass cuts: " << rinvcut << " " << z0cut << " " << inSector(iphi0_new, irinv_new, phi0, rinv)
              << std::endl;
    std::cout << "rinvcut  : " << settings_.rinvmax() * (120.0 * (1 << n_rinv_)) / phiHG_ << " " << settings_.rinvmax()
              << " " << 1.0 / ((120.0 * (1 << n_rinv_)) / phiHG_) << std::endl;
  }

  if (!goodTrackPars(rinvcut, z0cut))
    return false;

  if (!inSector(iphi0_new, irinv_new, phi0, rinv))
    return false;

  if (settings_.writeMonitorData("TPars")) {
    globals_->ofstream("trackletparsdisk.txt")
        << "Trackpars         " << layerdisk1_ - 5 << "   " << rinv << " " << rinv << " " << rinv << "   " << phi0
        << " " << phi0 << " " << phi0 << "   " << t << " " << t << " " << t << "   " << z0 << " " << z0 << " " << z0
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
                                    rinv,
                                    phi0,
                                    0.0,
                                    z0,
                                    t,
                                    irinv_new,
                                    iphi0_new,
                                    0,
                                    iz0_new,
                                    it_new,
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

  return true;
}

bool TrackletCalculatorBase::overlapSeeding(const Stub* innerFPGAStub,
                                            const L1TStub* innerStub,
                                            const Stub* outerFPGAStub,
                                            const L1TStub* outerStub,
                                            bool print) {
  //Deal with overlap stubs here
  assert(outerFPGAStub->layerdisk() < N_LAYER);

  assert(innerFPGAStub->layerdisk() >= N_LAYER);

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

  //now binary

  int ir2 = innerFPGAStub->rvalue();
  int iphi2 = innerFPGAStub->phi().value();
  int iz2 = innerFPGAStub->z().value();

  int ir1 = outerFPGAStub->rvalue();
  int iphi1 = outerFPGAStub->phi().value();
  int iz1 = outerFPGAStub->z().value();

  //To get global precission
  int ll = outerFPGAStub->layer().value() + 1;
  ir1 = l1t::bitShift(ir1, (8 - settings_.nrbitsstub(ll - 1)));
  iphi1 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));
  iphi2 <<= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));

  unsigned int ir1mean = settings_.irmean(layerdisk1_);

  int ir1abs = ir1 + ir1mean;

  unsigned int idr = ir2 - ir1abs;

  if (idr >= LUT_idrinv_.size()) {
    return false;
  }

  int iz2mean = settings_.izmean(layerdisk2_ - N_LAYER);

  if (iz1 < 0) {
    iz2mean = -iz2mean;
  }

  int iz2abs = iz2 + iz2mean;

  int irinv_new, iphi0_new, iz0_new, it_new;

  calcPars(idr, iphi1, ir1abs, iz1, iphi2, ir2, iz2abs, irinv_new, iphi0_new, iz0_new, it_new);

  if (print) {
    std::cout << "======================================" << std::endl;
    std::cout << "ir:   " << ir1abs << " " << ir2 << std::endl;
    std::cout << "iphi: " << iphi1 << " " << iphi2 << std::endl;
    std::cout << "iz:   " << iz1 << " " << iz2abs << std::endl;
    std::cout << "iz2 iz2mean " << iz2 << " " << iz2mean << std::endl;
    std::cout << "pars: " << irinv_new << " " << iphi0_new << " " << iz0_new << " " << it_new << std::endl;
  }

  bool rinvcut = abs(irinv_new) < settings_.rinvcut() * (120.0 * (1 << n_rinv_)) / phiHG_;
  bool z0cut = abs(iz0_new) < settings_.z0cut() * (1 << n_z_) / 120.0;

  if (!goodTrackPars(rinvcut, z0cut))
    return false;

  if (!inSector(iphi0_new, irinv_new, phi0, rinv))
    return false;

  if (settings_.writeMonitorData("TPars")) {
    globals_->ofstream("trackletparsoverlap.txt")
        << "Trackpars " << layerdisk1_ - 5 << "   " << rinv << " " << irinv_new << " " << phi0 << " " << iphi0_new
        << "   " << t << " " << it_new << " " << z0 << " " << iz0_new << endl;
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
                                    rinv,
                                    phi0,
                                    0.0,
                                    z0,
                                    t,
                                    irinv_new,
                                    iphi0_new,
                                    0,
                                    iz0_new,
                                    it_new,
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

  return true;
}
