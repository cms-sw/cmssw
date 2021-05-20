// VMRouterTable: Lookup table used by the VMRouter to route stubs and provide information about which VMStubs are needed by the TrackletEngine
#include "L1Trigger/TrackFindingTracklet/interface/VMRouterTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include <algorithm>
#include <filesystem>

using namespace std;
using namespace trklet;

VMRouterTable::VMRouterTable(Settings const& settings) : settings_(settings) {}

VMRouterTable::VMRouterTable(Settings const& settings, unsigned int layerdisk, std::string const& name)
    : settings_(settings) {
  init(layerdisk, name);
}

void VMRouterTable::init(unsigned int layerdisk, std::string const& name) {
  zbits_ = settings_.vmrlutzbits(layerdisk);
  rbits_ = settings_.vmrlutrbits(layerdisk);

  rbins_ = (1 << rbits_);
  zbins_ = (1 << zbits_);

  if (layerdisk < N_LAYER) {
    zmin_ = -settings_.zlength();
    zmax_ = settings_.zlength();
    rmin_ = settings_.rmean(layerdisk) - settings_.drmax();
    rmax_ = settings_.rmean(layerdisk) + settings_.drmax();
  } else {
    rmin_ = 0;
    rmax_ = settings_.rmaxdisk();
    zmin_ = settings_.zmean(layerdisk - N_LAYER) - settings_.dzmax();
    zmax_ = settings_.zmean(layerdisk - N_LAYER) + settings_.dzmax();
  }

  dr_ = (rmax_ - rmin_) / rbins_;
  dz_ = (zmax_ - zmin_) / zbins_;

  int NBINS = settings_.NLONGVMBINS() * settings_.NLONGVMBINS();

  for (int izbin = 0; izbin < zbins_; izbin++) {
    for (int irbin = 0; irbin < rbins_; irbin++) {
      double r = rmin_ + (irbin + 0.5) * dr_;
      double z = zmin_ + (izbin + 0.5) * dz_;

      if (settings_.combined()) {
        int iznew = izbin - (1 << (zbits_ - 1));
        if (iznew < 0)
          iznew += (1 << zbits_);
        assert(iznew >= 0);
        assert(iznew < (1 << zbits_));
        z = zmin_ + (iznew + 0.5) * dz_;
        if (layerdisk < N_LAYER) {
          int irnew = irbin - (1 << (rbits_ - 1));
          if (irnew < 0)
            irnew += (1 << rbits_);
          assert(irnew >= 0);
          assert(irnew < (1 << rbits_));
          r = rmin_ + (irnew + 0.5) * dr_;
        }
      }

      if (layerdisk > (N_LAYER - 1) && irbin < 10)  //special case for the tabulated radii in 2S disks
        r = (layerdisk <= 7) ? settings_.rDSSinner(irbin) : settings_.rDSSouter(irbin);

      int bin;
      if (layerdisk < N_LAYER) {
        double zproj = z * settings_.rmean(layerdisk) / r;
        bin = NBINS * (zproj + settings_.zlength()) / (2 * settings_.zlength());
      } else {
        double rproj = r * settings_.zmean(layerdisk - N_LAYER) / z;
        bin = NBINS * (rproj - settings_.rmindiskvm()) / (settings_.rmaxdisk() - settings_.rmindiskvm());
      }
      if (bin < 0)
        bin = 0;
      if (bin >= NBINS)
        bin = NBINS - 1;
      vmrtable_.push_back(bin);

      if (layerdisk >= N_LAYER) {
        double rproj = r * settings_.zmean(layerdisk - N_LAYER) / z;
        bin = 0.5 * NBINS * (rproj - settings_.rmindiskvm()) / (settings_.rmaxdiskvm() - settings_.rmindiskvm());
        //bin value of zero indicates that stub is out of range
        if (bin < 0)
          bin = 0;
        if (bin >= NBINS / 2)
          bin = 0;
        vmrtabletedisk_.push_back(bin);
      }

      if (layerdisk == 0 || layerdisk == 2 || layerdisk == 4 || layerdisk == 6 || layerdisk == 8) {
        vmrtableteinner_.push_back(getLookup(layerdisk + 1, z, r));
      }

      if (layerdisk == 1) {
        vmrtableteinner_.push_back(getLookup(layerdisk + 1, z, r, 1));
      }

      if (layerdisk == 1) {  //projection from L2 to D1 for L2L3D1 seeding
        vmrtableteinnerThird_.push_back(getLookup(6, z, r, 10));
      }

      if (layerdisk == 4) {  //projection from L5 to L4 for L5L6L4 seeding
        vmrtableteinnerThird_.push_back(getLookup(3, z, r));
      }

      if (layerdisk == 2) {  //projection from L3 to L5 for L3L4L2 seeding
        vmrtableteinnerThird_.push_back(getLookup(1, z, r));
      }

      if (layerdisk == 6) {  //projection from D1 to L2 for D1D2L2 seeding
        vmrtableteinnerThird_.push_back(getLookup(1, z, r));
      }

      if (layerdisk == 0 || layerdisk == 1) {
        vmrtableteinneroverlap_.push_back(getLookup(6, z, r, layerdisk + 6));
      }
    }
  }

  if (settings_.writeTable()) {
    if (!settings_.combined()) {
      // write finebin tables
      writeVMTable(settings_.tablePath(), name + "_finebin.tab", vmrtable_);
      // write barrel seed teinner tables (L1L2, L2L3, L3L4, L5L6)
      if (layerdisk == 0 || layerdisk == 1 || layerdisk == 2 || layerdisk == 4) {
        std::string fnamesuffix = "L" + to_string(layerdisk + 1) + "L" + std::to_string(layerdisk + 2);
        writeVMTable(settings_.tablePath(), "VMTableInner" + fnamesuffix + ".tab", vmrtableteinner_);
      }
      // write disk seed teinner tables (D1D2, D3D4)
      if (layerdisk == 6 || layerdisk == 8) {
        std::string fnamesuffix = "D" + to_string(layerdisk - N_LAYER + 1) + "D" + to_string(layerdisk - N_LAYER + 2);
        writeVMTable(settings_.tablePath(), "VMTableInner" + fnamesuffix + ".tab", vmrtableteinner_);
      }
      // write overlap seed teinner tables (L1D1, L2D1)
      if (layerdisk == 0 || layerdisk == 1) {
        std::string fnamesuffix = "L" + to_string(layerdisk + 1) + "D1";
        writeVMTable(settings_.tablePath(), "VMTableInner" + fnamesuffix + ".tab", vmrtableteinneroverlap_);
      }
      // write barrel teouter tables (L2, L3, L4, L6, same as finebin tables)
      if (layerdisk == 1 || layerdisk == 2 || layerdisk == 3 || layerdisk == 5) {
        std::string fnamesuffix = "L" + to_string(layerdisk + 1);
        writeVMTable(settings_.tablePath(), "VMTableOuter" + fnamesuffix + ".tab", vmrtable_);
      }
      // write disk teouter tables (D1, D2, D4)
      if (layerdisk == 6 || layerdisk == 7 || layerdisk == 9) {
        std::string fnamesuffix = "D" + to_string(layerdisk - N_LAYER + 1);
        writeVMTable(settings_.tablePath(), "VMTableOuter" + fnamesuffix + ".tab", vmrtabletedisk_);
      }
    } else {
      std::string fname = "VMRME_";
      if (layerdisk < 6) {
        fname += "L" + std::to_string(layerdisk + 1) + ".tab";
      } else {
        fname += "D" + std::to_string(layerdisk - 5) + ".tab";
      }
      writeVMTable(settings_.tablePath(), fname, vmrtable_);

      if (layerdisk == 6 || layerdisk == 7 || layerdisk == 9) {
        std::string fname = "VMRTE_D" + to_string(layerdisk - N_LAYER + 1) + ".tab";
        writeVMTable(settings_.tablePath(), fname, vmrtabletedisk_);
      }

      if (vmrtableteinner_.size() > 0) {
        ofstream out;
        if (layerdisk < 6) {
          out.open(settings_.tablePath() + "TP_L" + std::to_string(layerdisk + 1) + ".tab");
        } else {
          out.open(settings_.tablePath() + "TP_D" + std::to_string(layerdisk - 5) + ".tab");
        }
        out << "{" << endl;
        for (unsigned int i = 0; i < vmrtableteinner_.size(); i++) {
          if (i != 0) {
            out << "," << endl;
          }
          if (vmrtableteinner_[i] == -1) {
            out << 1023;
          } else {
            out << vmrtableteinner_[i];
          }
        }
        out << endl << "};" << endl;
        out.close();
      }
    }
  }
}

int VMRouterTable::getLookup(unsigned int layerdisk, double z, double r, int iseed) {
  double z0cut = settings_.z0cut();

  if (layerdisk < N_LAYER) {
    if (iseed == 1 && std::abs(z) < 52.0)
      return -1;

    double rmean = settings_.rmean(layerdisk);

    double rratio1 = rmean / (r + 0.5 * dr_);
    double rratio2 = rmean / (r - 0.5 * dr_);

    double z1 = (z - 0.5 * dz_) * rratio1 + z0cut * (rratio1 - 1.0);
    double z2 = (z + 0.5 * dz_) * rratio1 + z0cut * (rratio1 - 1.0);
    double z3 = (z - 0.5 * dz_) * rratio2 + z0cut * (rratio2 - 1.0);
    double z4 = (z + 0.5 * dz_) * rratio2 + z0cut * (rratio2 - 1.0);
    double z5 = (z - 0.5 * dz_) * rratio1 - z0cut * (rratio1 - 1.0);
    double z6 = (z + 0.5 * dz_) * rratio1 - z0cut * (rratio1 - 1.0);
    double z7 = (z - 0.5 * dz_) * rratio2 - z0cut * (rratio2 - 1.0);
    double z8 = (z + 0.5 * dz_) * rratio2 - z0cut * (rratio2 - 1.0);

    double zmin = std::min({z1, z2, z3, z4, z5, z6, z7, z8});
    double zmax = std::max({z1, z2, z3, z4, z5, z6, z7, z8});

    int NBINS = settings_.NLONGVMBINS() * settings_.NLONGVMBINS();

    int zbin1 = NBINS * (zmin + settings_.zlength()) / (2 * settings_.zlength());
    int zbin2 = NBINS * (zmax + settings_.zlength()) / (2 * settings_.zlength());

    if (zbin1 >= NBINS)
      return -1;
    if (zbin2 < 0)
      return -1;

    if (zbin2 >= NBINS)
      zbin2 = NBINS - 1;
    if (zbin1 < 0)
      zbin1 = 0;

    // This is a 10 bit word:
    // xxx|yyy|z|rrr
    // xxx is the delta z window
    // yyy is the z bin
    // z is flag to look in next bin
    // rrr first fine z bin
    // NOTE : this encoding is not efficient z is one if xxx+rrr is greater than 8
    //        and xxx is only 1,2, or 3
    //        should also reject xxx=0 as this means projection is outside range

    int value = zbin1 / 8;
    value *= 2;
    if (zbin2 / 8 - zbin1 / 8 > 0)
      value += 1;
    value *= 8;
    value += (zbin1 & 7);
    assert(value / 8 < 15);
    int deltaz = zbin2 - zbin1;
    if (deltaz > 7) {
      deltaz = 7;
    }
    assert(deltaz < 8);
    value += (deltaz << 7);

    return value;

  } else {
    if (std::abs(z) < 2.0 * z0cut)
      return -1;

    double zmean = settings_.zmean(layerdisk - N_LAYER);
    if (z < 0.0)
      zmean = -zmean;

    double r1 = (r + 0.5 * dr_) * (zmean + z0cut) / (z + 0.5 * dz_ + z0cut);
    double r2 = (r - 0.5 * dr_) * (zmean - z0cut) / (z + 0.5 * dz_ - z0cut);
    double r3 = (r + 0.5 * dr_) * (zmean + z0cut) / (z - 0.5 * dz_ + z0cut);
    double r4 = (r - 0.5 * dr_) * (zmean - z0cut) / (z - 0.5 * dz_ - z0cut);
    double r5 = (r + 0.5 * dr_) * (zmean - z0cut) / (z + 0.5 * dz_ - z0cut);
    double r6 = (r - 0.5 * dr_) * (zmean + z0cut) / (z + 0.5 * dz_ + z0cut);
    double r7 = (r + 0.5 * dr_) * (zmean - z0cut) / (z - 0.5 * dz_ - z0cut);
    double r8 = (r - 0.5 * dr_) * (zmean + z0cut) / (z - 0.5 * dz_ + z0cut);

    double rmin = std::min({r1, r2, r3, r4, r5, r6, r7, r8});
    double rmax = std::max({r1, r2, r3, r4, r5, r6, r7, r8});

    int NBINS = settings_.NLONGVMBINS() * settings_.NLONGVMBINS() / 2;

    double rmindisk = settings_.rmindiskvm();
    double rmaxdisk = settings_.rmaxdiskvm();

    if (iseed == 6)
      rmaxdisk = settings_.rmaxdiskl1overlapvm();
    if (iseed == 7)
      rmindisk = settings_.rmindiskl2overlapvm();
    if (iseed == 10)
      rmaxdisk = settings_.rmaxdisk();

    if (rmin > rmaxdisk)
      return -1;
    if (rmax > rmaxdisk)
      rmax = rmaxdisk;

    if (rmax < rmindisk)
      return -1;
    if (rmin < rmindisk)
      rmin = rmindisk;

    int rbin1 = NBINS * (rmin - settings_.rmindiskvm()) / (settings_.rmaxdiskvm() - settings_.rmindiskvm());
    int rbin2 = NBINS * (rmax - settings_.rmindiskvm()) / (settings_.rmaxdiskvm() - settings_.rmindiskvm());

    if (iseed == 10) {
      constexpr double rminspec = 40.0;
      rbin1 = NBINS * (rmin - rminspec) / (settings_.rmaxdisk() - rminspec);
      rbin2 = NBINS * (rmax - rminspec) / (settings_.rmaxdisk() - rminspec);
    }

    if (rbin2 >= NBINS)
      rbin2 = NBINS - 1;
    if (rbin1 < 0)
      rbin1 = 0;

    // This is a 9 bit word:
    // xxx|yy|z|rrr
    // xxx is the delta r window
    // yy is the r bin yy is three bits for overlaps
    // z is flag to look in next bin
    // rrr fine r bin
    // NOTE : this encoding is not efficient z is one if xxx+rrr is greater than 8
    //        and xxx is only 1,2, or 3
    //        should also reject xxx=0 as this means projection is outside range

    bool overlap = iseed == 6 || iseed == 7 || iseed == 10;

    int value = rbin1 / 8;
    if (overlap) {
      if (z < 0.0)
        value += 4;
    }
    value *= 2;
    if (rbin2 / 8 - rbin1 / 8 > 0)
      value += 1;
    value *= 8;
    value += (rbin1 & 7);
    assert(value / 8 < 15);
    int deltar = rbin2 - rbin1;
    if (deltar > 7)
      deltar = 7;
    if (overlap) {
      value += (deltar << 7);
    } else {
      value += (deltar << 6);
    }

    return value;
  }
}

int VMRouterTable::lookup(int zbin, int rbin) {
  int index = zbin * rbins_ + rbin;
  assert(index >= 0 && index < (int)vmrtable_.size());
  return vmrtable_[index];
}

int VMRouterTable::lookupdisk(int zbin, int rbin) {
  int index = zbin * rbins_ + rbin;
  assert(index >= 0 && index < (int)vmrtabletedisk_.size());
  return vmrtabletedisk_[index];
}

int VMRouterTable::lookupinner(int zbin, int rbin) {
  int index = zbin * rbins_ + rbin;
  assert(index >= 0 && index < (int)vmrtableteinner_.size());
  return vmrtableteinner_[index];
}

int VMRouterTable::lookupinneroverlap(int zbin, int rbin) {
  int index = zbin * rbins_ + rbin;
  assert(index >= 0 && index < (int)vmrtableteinneroverlap_.size());
  return vmrtableteinneroverlap_[index];
}

int VMRouterTable::lookupinnerThird(int zbin, int rbin) {
  int index = zbin * rbins_ + rbin;
  assert(index >= 0 && index < (int)vmrtableteinnerThird_.size());
  return vmrtableteinnerThird_[index];
}

void VMRouterTable::writeVMTable(std::string const& dir, std::string const& name, std::vector<int> const& table) {
  ofstream out = openfile(dir, name, __FILE__, __LINE__);

  out << "{" << endl;
  for (unsigned int i = 0; i < table.size(); i++) {
    if (i != 0) {
      out << "," << endl;
    }
    int itable = table[i];
    out << itable;
  }
  out << endl << "};" << endl;
  out.close();
}
