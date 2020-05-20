#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cmath>

using namespace std;
using namespace trklet;

Stub::Stub(const trklet::Settings* const settings) : settings_(settings) {}

Stub::Stub(L1TStub& stub, const trklet::Settings* const settings, double phiminsec, double phimaxsec)
    : settings_(settings) {
  double r = stub.r();
  double z = stub.z();
  double sbend = stub.bend();

  l1tstub_ = &stub;

  int bendbits = 4;
  if (stub.isPSmodule())
    bendbits = 3;

  int ibend = bendencode(sbend, stub.isPSmodule());

  bend_.set(ibend, bendbits, true, __LINE__, __FILE__);

  int layer = stub.layer() + 1;

  // hold the real values from L1Stub
  double stubphi = stub.phi();

  if (layer < 999) {
    disk_.set(0, 4, false, __LINE__, __FILE__);

    assert(layer >= 1 && layer <= 6);
    double rmin = settings_->rmean(layer - 1) - settings_->drmax();
    double rmax = settings_->rmean(layer - 1) + settings_->drmax();

    if (r < rmin || r > rmax) {
      edm::LogProblem("Tracklet") << "Error r, rmin, rmeas,  rmax :" << r << " " << rmin << " " << 0.5 * (rmin + rmax)
                                  << " " << rmax;
    }

    int irbits = settings_->nrbitsstub(layer - 1);

    int ir = lround((1 << irbits) * ((r - settings_->rmean(layer - 1)) / (rmax - rmin)));

    double zmin = -settings_->zlength();
    double zmax = settings_->zlength();

    if (z < zmin || z > zmax) {
      edm::LogProblem("Tracklet") << "Error z, zmin, zmax :" << z << " " << zmin << " " << zmax;
    }

    int izbits = settings_->nzbitsstub(layer - 1);

    int iz = lround((1 << izbits) * z / (zmax - zmin));

    if (z < zmin || z > zmax) {
      edm::LogProblem("Tracklet") << "Error z, zmin, zmax :" << z << " " << zmin << " " << zmax;
    }

    assert(phimaxsec - phiminsec > 0.0);

    if (stubphi < phiminsec - (phimaxsec - phiminsec) / 6.0) {
      stubphi += 2 * M_PI;
    }
    assert((phimaxsec - phiminsec) > 0.0);

    int iphibits = settings_->nphibitsstub(layer - 1);

    double deltaphi = trklet::phiRange(stubphi - phiminsec);

    int iphi = (1 << iphibits) * deltaphi / (phimaxsec - phiminsec);

    layer_.set(layer - 1, 3, true, __LINE__, __FILE__);
    r_.set(ir, irbits, false, __LINE__, __FILE__);
    z_.set(iz, izbits, false, __LINE__, __FILE__);
    phi_.set(iphi, iphibits, true, __LINE__, __FILE__);

    phicorr_.set(iphi, iphibits, true, __LINE__, __FILE__);

  } else {
    // Here we handle the hits on disks.

    int disk = stub.module();
    assert(disk >= 1 && disk <= 5);
    int sign = 1;
    if (z < 0.0)
      sign = -1;

    double zmin = sign * (settings_->zmean(disk - 1) - sign * settings_->dzmax());
    double zmax = sign * (settings_->zmean(disk - 1) + sign * settings_->dzmax());

    if ((z > zmax) || (z < zmin)) {
      edm::LogProblem("Tracklet") << "Error disk z, zmax, zmin: " << z << " " << zmax << " " << zmin;
    }

    int iz = (1 << settings->nzbitsstub(disk + 5)) * ((z - sign * settings_->zmean(disk - 1)) / std::abs(zmax - zmin));

    assert(phimaxsec - phiminsec > 0.0);
    if (stubphi < phiminsec - (phimaxsec - phiminsec) / 6.0) {
      stubphi += 2 * M_PI;
    }

    assert(phimaxsec - phiminsec > 0.0);
    if (stubphi < phiminsec - (phimaxsec - phiminsec) / 6.0) {
      stubphi += 2 * M_PI;
    }

    int iphibits = settings_->nphibitsstub(disk + 5);

    double deltaphi = trklet::phiRange(stubphi - phiminsec);

    int iphi = (1 << iphibits) * deltaphi / (phimaxsec - phiminsec);

    double rmin = 0;
    double rmax = settings_->rmaxdisk();

    if (r < rmin || r > rmax) {
      edm::LogProblem("Tracklet") << "Error disk r, rmin, rmax :" << r << " " << rmin << " " << rmax;
    }

    int ir = (1 << settings_->nrbitsstub(disk + 5)) * (r - rmin) / (rmax - rmin);

    int irSS = -1;
    if (!stub.isPSmodule()) {
      for (int i = 0; i < 10; ++i) {
        if (disk <= 2) {
          if (std::abs(r - settings_->rDSSinner(i)) < 0.2) {
            irSS = i;
            break;
          }
        } else {
          if (std::abs(r - settings_->rDSSouter(i)) < 0.2) {
            irSS = i;
            break;
          }
        }
      }
      if (irSS < 0) {
        throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " didn't find rDSS value! r = " << r
                                          << " Check that correct geometry is used!";
      }
    }
    if (irSS < 0) {
      //PS modules
      r_.set(ir, settings_->nrbitsstub(disk + 5), true, __LINE__, __FILE__);
    } else {
      //SS modules
      r_.set(irSS, 4, true, __LINE__, __FILE__);  // in case of SS modules, store index, not r itself
    }

    z_.set(iz, settings->nzbitsstub(disk + 5), false, __LINE__, __FILE__);
    phi_.set(iphi, iphibits, true, __LINE__, __FILE__);
    phicorr_.set(iphi, iphibits, true, __LINE__, __FILE__);

    disk_.set(sign * disk, 4, false, __LINE__, __FILE__);

    double alphanorm = stub.alphanorm();
    assert(std::abs(alphanorm) < 1.0);
    int ialphanew = alphanorm * (1 << (settings->nbitsalpha() - 1));
    assert(ialphanew < (1 << (settings->nbitsalpha() - 1)));
    assert(ialphanew >= -(1 << (settings->nbitsalpha() - 1)));
    alphanew_.set(ialphanew, settings->nbitsalpha(), false, __LINE__, __FILE__);
  }
}

FPGAWord Stub::iphivmFineBins(int VMbits, int finebits) const {
  unsigned int finephi = (phicorr_.value() >> (phicorr_.nbits() - VMbits - finebits)) & ((1 << finebits) - 1);
  return FPGAWord(finephi, finebits, true, __LINE__, __FILE__);
}

unsigned int Stub::phiregionaddress() const {
  int iphi = (phicorr_.value() >> (phicorr_.nbits() - settings_->nbitsallstubs(layerdisk())));
  return (iphi << 7) + stubindex_.value();
}

std::string Stub::phiregionaddressstr() const {
  int iphi = (phicorr_.value() >> (phicorr_.nbits() - settings_->nbitsallstubs(layerdisk())));
  FPGAWord phiregion(iphi, 3, true, __LINE__, __FILE__);
  return phiregion.str() + stubindex_.str();
}

void Stub::setAllStubIndex(int nstub) {
  if (nstub >= (1 << 7)) {
    if (settings_->debugTracklet())
      edm::LogPrint("Tracklet") << "Warning too large stubindex!";
    nstub = (1 << 7) - 1;
  }

  stubindex_.set(nstub, 7);
}

void Stub::setPhiCorr(int phiCorr) {
  int iphicorr = phi_.value() - phiCorr;

  if (iphicorr < 0)
    iphicorr = 0;
  if (iphicorr >= (1 << phi_.nbits()))
    iphicorr = (1 << phi_.nbits()) - 1;

  phicorr_.set(iphicorr, phi_.nbits(), true, __LINE__, __FILE__);
}

double Stub::rapprox() const {
  if (disk_.value() == 0) {
    int lr = 1 << (8 - settings_->nrbitsstub(layer_.value()));
    return r_.value() * settings_->kr() * lr + settings_->rmean(layer_.value());
  }
  return r_.value() * settings_->kr();
}

double Stub::zapprox() const {
  if (disk_.value() == 0) {
    int lz = 1;
    if (layer_.value() >= 3) {
      lz = 16;
    }
    return z_.value() * settings_->kz() * lz;
  }
  int sign = 1;
  if (disk_.value() < 0)
    sign = -1;
  if (sign < 0) {
    return (z_.value() + 1) * settings_->kz() +
           sign *
               settings_->zmean(abs(disk_.value()) -
                                1);  //Should understand why this is needed to get agreement with integer calculations
  } else {
    return z_.value() * settings_->kz() + sign * settings_->zmean(abs(disk_.value()) - 1);
  }
}

double Stub::phiapprox(double phimin, double) const {
  int lphi = 1;
  if (layer_.value() >= 3) {
    lphi = 8;
  }
  return trklet::phiRange(phimin + phi_.value() * settings_->kphi() / lphi);
}

unsigned int Stub::layerdisk() const {
  if (layer_.value() == -1)
    return 5 + abs(disk_.value());
  return layer_.value();
}
