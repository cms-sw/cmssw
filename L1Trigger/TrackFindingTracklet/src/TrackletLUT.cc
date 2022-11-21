#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletConfigBuilder.h"
#include "L1Trigger/L1TCommon/interface/BitShift.h"

#include <filesystem>

using namespace std;
using namespace trklet;

TrackletLUT::TrackletLUT(const Settings& settings) : settings_(settings) {}

void TrackletLUT::initmatchcut(unsigned int layerdisk, MatchType type, unsigned int region) {
  char cregion = 'A' + region;

  for (unsigned int iSeed = 0; iSeed < 12; iSeed++) {
    if (type == barrelphi) {
      table_.push_back(settings_.rphimatchcut(iSeed, layerdisk) / (settings_.kphi1() * settings_.rmean(layerdisk)));
    }
    if (type == barrelz) {
      table_.push_back(settings_.zmatchcut(iSeed, layerdisk) / settings_.kz());
    }
    if (type == diskPSphi) {
      table_.push_back(settings_.rphicutPS(iSeed, layerdisk - N_LAYER) / (settings_.kphi() * settings_.kr()));
    }
    if (type == disk2Sphi) {
      table_.push_back(settings_.rphicut2S(iSeed, layerdisk - N_LAYER) / (settings_.kphi() * settings_.kr()));
    }
    if (type == disk2Sr) {
      table_.push_back(settings_.rcut2S(iSeed, layerdisk - N_LAYER) / settings_.krprojshiftdisk());
    }
    if (type == diskPSr) {
      table_.push_back(settings_.rcutPS(iSeed, layerdisk - N_LAYER) / settings_.krprojshiftdisk());
    }
  }

  name_ = settings_.combined() ? "MP_" : "MC_";

  if (type == barrelphi) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_phicut.tab";
  }
  if (type == barrelz) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_zcut.tab";
  }
  if (type == diskPSphi) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_PSphicut.tab";
  }
  if (type == disk2Sphi) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_2Sphicut.tab";
  }
  if (type == disk2Sr) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_2Srcut.tab";
  }
  if (type == diskPSr) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_PSrcut.tab";
  }

  positive_ = false;

  writeTable();
}

void TrackletLUT::initTPlut(bool fillInner,
                            unsigned int iSeed,
                            unsigned int layerdisk1,
                            unsigned int layerdisk2,
                            unsigned int nbitsfinephidiff,
                            unsigned int iTP) {
  //number of fine phi bins in sector
  int nfinephibins = settings_.nallstubs(layerdisk2) * settings_.nvmte(1, iSeed) * (1 << settings_.nfinephi(1, iSeed));
  double dfinephi = settings_.dphisectorHG() / nfinephibins;

  int outerrbits = 3;

  if (iSeed == Seed::L1L2 || iSeed == Seed::L2L3 || iSeed == Seed::L3L4 || iSeed == Seed::L5L6) {
    outerrbits = 0;
  }

  int outerrbins = (1 << outerrbits);

  double dphi[2];
  double router[2];

  unsigned int nbendbitsinner = 3;
  unsigned int nbendbitsouter = 3;
  if (iSeed == Seed::L3L4) {
    nbendbitsouter = 4;
  } else if (iSeed == Seed::L5L6) {
    nbendbitsinner = 4;
    nbendbitsouter = 4;
  }

  int nbinsfinephidiff = (1 << nbitsfinephidiff);

  for (int iphibin = 0; iphibin < nbinsfinephidiff; iphibin++) {
    int iphidiff = iphibin;
    if (iphibin >= nbinsfinephidiff / 2) {
      iphidiff = iphibin - nbinsfinephidiff;
    }
    //min and max dphi
    dphi[0] = (iphidiff - 1.5) * dfinephi;
    dphi[1] = (iphidiff + 1.5) * dfinephi;
    for (int irouterbin = 0; irouterbin < outerrbins; irouterbin++) {
      if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4 || iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
        router[0] =
            settings_.rmindiskvm() + irouterbin * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
        router[1] =
            settings_.rmindiskvm() + (irouterbin + 1) * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
      } else {
        router[0] = settings_.rmean(layerdisk2);
        router[1] = settings_.rmean(layerdisk2);
      }

      double bendinnermin = 20.0;
      double bendinnermax = -20.0;
      double bendoutermin = 20.0;
      double bendoutermax = -20.0;
      double rinvmin = 1.0;
      for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
          double rinner = 0.0;
          if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4) {
            rinner = router[i3] * settings_.zmean(layerdisk1 - N_LAYER) / settings_.zmean(layerdisk2 - N_LAYER);
          } else {
            rinner = settings_.rmean(layerdisk1);
          }
          double rinv1 = (rinner < router[i3]) ? rinv(0.0, -dphi[i2], rinner, router[i3]) : 20.0;
          double pitchinner = (rinner < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
          double pitchouter =
              (router[i3] < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
          double abendinner = bendstrip(rinner, rinv1, pitchinner);
          double abendouter = bendstrip(router[i3], rinv1, pitchouter);
          if (abendinner < bendinnermin)
            bendinnermin = abendinner;
          if (abendinner > bendinnermax)
            bendinnermax = abendinner;
          if (abendouter < bendoutermin)
            bendoutermin = abendouter;
          if (abendouter > bendoutermax)
            bendoutermax = abendouter;
          if (std::abs(rinv1) < rinvmin) {
            rinvmin = std::abs(rinv1);
          }
        }
      }

      bool passptcut = rinvmin < settings_.rinvcutte();

      if (fillInner) {
        for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
          double bend = settings_.benddecode(ibend, layerdisk1, nbendbitsinner == 3);

          bool passinner = bend <= bendinnermax + settings_.bendcutte(ibend, layerdisk1, nbendbitsinner == 3) &&
                           bend >= bendinnermin - settings_.bendcutte(ibend, layerdisk1, nbendbitsinner == 3);
          table_.push_back(passinner && passptcut);
        }
      } else {
        for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
          double bend = settings_.benddecode(ibend, layerdisk2, nbendbitsouter == 3);

          bool passouter = bend <= bendoutermax + settings_.bendcutte(ibend, layerdisk2, nbendbitsouter == 3) &&
                           bend >= bendoutermin - settings_.bendcutte(ibend, layerdisk2, nbendbitsouter == 3);
          table_.push_back(passouter && passptcut);
        }
      }
    }
  }

  positive_ = false;
  char cTP = 'A' + iTP;

  name_ = "TP_" + TrackletConfigBuilder::LayerName(layerdisk1) + TrackletConfigBuilder::LayerName(layerdisk2) + cTP;

  if (fillInner) {
    name_ += "_stubptinnercut.tab";
  } else {
    name_ += "_stubptoutercut.tab";
  }

  writeTable();
}

void TrackletLUT::initTPregionlut(unsigned int iSeed,
                                  unsigned int layerdisk1,
                                  unsigned int layerdisk2,
                                  unsigned int iAllStub,
                                  unsigned int nbitsfinephidiff,
                                  unsigned int nbitsfinephi,
                                  const TrackletLUT& tplutinner,
                                  unsigned int iTP) {
  int nirbits = 0;
  if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4 || iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
    nirbits = 3;
  }

  unsigned int nbendbitsinner = 3;

  if (iSeed == Seed::L5L6) {
    nbendbitsinner = 4;
  }

  for (int innerfinephi = 0; innerfinephi < (1 << nbitsfinephi); innerfinephi++) {
    for (int innerbend = 0; innerbend < (1 << nbendbitsinner); innerbend++) {
      for (int ir = 0; ir < (1 << nirbits); ir++) {
        unsigned int usereg = 0;
        for (unsigned int ireg = 0; ireg < settings_.nvmte(1, iSeed); ireg++) {
          bool match = false;
          for (int ifinephiouter = 0; ifinephiouter < (1 << settings_.nfinephi(1, iSeed)); ifinephiouter++) {
            int outerfinephi = iAllStub * (1 << (nbitsfinephi - settings_.nbitsallstubs(layerdisk2))) +
                               ireg * (1 << settings_.nfinephi(1, iSeed)) + ifinephiouter;
            int idphi = outerfinephi - innerfinephi;
            bool inrange = (idphi < (1 << (nbitsfinephidiff - 1))) && (idphi >= -(1 << (nbitsfinephidiff - 1)));
            if (idphi < 0)
              idphi = idphi + (1 << nbitsfinephidiff);
            int idphi1 = idphi;
            if (iSeed >= 4)
              idphi1 = (idphi << 3) + ir;
            int ptinnerindexnew = l1t::bitShift(idphi1, nbendbitsinner) + innerbend;
            match = match || (inrange && tplutinner.lookup(ptinnerindexnew));
          }
          if (match) {
            usereg = usereg | (1 << ireg);
          }
        }

        table_.push_back(usereg);
      }
    }
  }

  positive_ = false;
  char cTP = 'A' + iTP;

  name_ = "TP_" + TrackletConfigBuilder::LayerName(layerdisk1) + TrackletConfigBuilder::LayerName(layerdisk2) + cTP +
          "_usereg.tab";

  writeTable();
}

void TrackletLUT::initteptlut(bool fillInner,
                              bool fillTEMem,
                              unsigned int iSeed,
                              unsigned int layerdisk1,
                              unsigned int layerdisk2,
                              unsigned int innerphibits,
                              unsigned int outerphibits,
                              double innerphimin,
                              double innerphimax,
                              double outerphimin,
                              double outerphimax,
                              const std::string& innermem,
                              const std::string& outermem) {
  int outerrbits = 0;
  if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4 || iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
    outerrbits = 3;
  }

  int outerrbins = (1 << outerrbits);
  int innerphibins = (1 << innerphibits);
  int outerphibins = (1 << outerphibits);

  double phiinner[2];
  double phiouter[2];
  double router[2];

  unsigned int nbendbitsinner = 3;
  unsigned int nbendbitsouter = 3;
  if (iSeed == Seed::L3L4) {
    nbendbitsouter = 4;
  }
  if (iSeed == Seed::L5L6) {
    nbendbitsinner = 4;
    nbendbitsouter = 4;
  }

  if (fillTEMem) {
    if (fillInner) {
      table_.resize((1 << nbendbitsinner), false);
    } else {
      table_.resize((1 << nbendbitsouter), false);
    }
  }

  for (int iphiinnerbin = 0; iphiinnerbin < innerphibins; iphiinnerbin++) {
    phiinner[0] = innerphimin + iphiinnerbin * (innerphimax - innerphimin) / innerphibins;
    phiinner[1] = innerphimin + (iphiinnerbin + 1) * (innerphimax - innerphimin) / innerphibins;
    for (int iphiouterbin = 0; iphiouterbin < outerphibins; iphiouterbin++) {
      phiouter[0] = outerphimin + iphiouterbin * (outerphimax - outerphimin) / outerphibins;
      phiouter[1] = outerphimin + (iphiouterbin + 1) * (outerphimax - outerphimin) / outerphibins;
      for (int irouterbin = 0; irouterbin < outerrbins; irouterbin++) {
        if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4 || iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
          router[0] =
              settings_.rmindiskvm() + irouterbin * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
          router[1] = settings_.rmindiskvm() +
                      (irouterbin + 1) * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
        } else {
          router[0] = settings_.rmean(layerdisk2);
          router[1] = settings_.rmean(layerdisk2);
        }

        double bendinnermin = 20.0;
        double bendinnermax = -20.0;
        double bendoutermin = 20.0;
        double bendoutermax = -20.0;
        double rinvmin = 1.0;
        for (int i1 = 0; i1 < 2; i1++) {
          for (int i2 = 0; i2 < 2; i2++) {
            for (int i3 = 0; i3 < 2; i3++) {
              double rinner = 0.0;
              if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4) {
                rinner = router[i3] * settings_.zmean(layerdisk1 - N_LAYER) / settings_.zmean(layerdisk2 - N_LAYER);
              } else {
                rinner = settings_.rmean(layerdisk1);
              }
              double rinv1 = (rinner < router[i3]) ? -rinv(phiinner[i1], phiouter[i2], rinner, router[i3]) : -20.0;
              double pitchinner =
                  (rinner < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
              double pitchouter =
                  (router[i3] < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
              double abendinner = bendstrip(rinner, rinv1, pitchinner);
              double abendouter = bendstrip(router[i3], rinv1, pitchouter);
              if (abendinner < bendinnermin)
                bendinnermin = abendinner;
              if (abendinner > bendinnermax)
                bendinnermax = abendinner;
              if (abendouter < bendoutermin)
                bendoutermin = abendouter;
              if (abendouter > bendoutermax)
                bendoutermax = abendouter;
              if (std::abs(rinv1) < rinvmin) {
                rinvmin = std::abs(rinv1);
              }
            }
          }
        }

        bool passptcut = rinvmin < settings_.rinvcutte();

        if (fillInner) {
          for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
            double bend = settings_.benddecode(ibend, layerdisk1, nbendbitsinner == 3);

            bool passinner = bend > bendinnermin - settings_.bendcutte(ibend, layerdisk1, nbendbitsinner == 3) &&
                             bend < bendinnermax + settings_.bendcutte(ibend, layerdisk1, nbendbitsinner == 3);

            if (fillTEMem) {
              if (passinner) {
                table_[ibend] = 1;
              }
            } else {
              table_.push_back(passinner && passptcut);
            }
          }
        } else {
          for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
            double bend = settings_.benddecode(ibend, layerdisk2, nbendbitsouter == 3);

            bool passouter = bend > bendoutermin - settings_.bendcutte(ibend, layerdisk2, nbendbitsouter == 3) &&
                             bend < bendoutermax + settings_.bendcutte(ibend, layerdisk2, nbendbitsouter == 3);
            if (fillTEMem) {
              if (passouter) {
                table_[ibend] = 1;
              }
            } else {
              table_.push_back(passouter && passptcut);
            }
          }
        }
      }
    }
  }

  positive_ = false;

  if (fillTEMem) {
    if (fillInner) {
      name_ = "VMSTE_" + innermem + "_vmbendcut.tab";
    } else {
      name_ = "VMSTE_" + outermem + "_vmbendcut.tab";
    }
  } else {
    name_ = "TE_" + innermem.substr(0, innermem.size() - 2) + "_" + outermem.substr(0, outermem.size() - 2);
    if (fillInner) {
      name_ += "_stubptinnercut.tab";
    } else {
      name_ += "_stubptoutercut.tab";
    }
  }

  writeTable();
}

void TrackletLUT::initProjectionBend(double k_phider,
                                     unsigned int idisk,
                                     unsigned int nrbits,
                                     unsigned int nphiderbits) {
  unsigned int nsignbins = 2;
  unsigned int nrbins = 1 << (nrbits);
  unsigned int nphiderbins = 1 << (nphiderbits);

  for (unsigned int isignbin = 0; isignbin < nsignbins; isignbin++) {
    for (unsigned int irbin = 0; irbin < nrbins; irbin++) {
      int ir = irbin;
      if (ir > (1 << (nrbits - 1)))
        ir -= (1 << nrbits);
      ir = l1t::bitShift(ir, (settings_.nrbitsstub(N_LAYER) - nrbits));
      for (unsigned int iphiderbin = 0; iphiderbin < nphiderbins; iphiderbin++) {
        int iphider = iphiderbin;
        if (iphider > (1 << (nphiderbits - 1)))
          iphider -= (1 << nphiderbits);
        iphider = l1t::bitShift(iphider, (settings_.nbitsphiprojderL123() - nphiderbits));

        double rproj = ir * settings_.krprojshiftdisk();
        double phider = iphider * k_phider;
        double t = settings_.zmean(idisk) / rproj;

        if (isignbin)
          t = -t;

        double rinv = -phider * (2.0 * t);

        double stripPitch = (rproj < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
        double bendproj = bendstrip(rproj, rinv, stripPitch);

        static double maxbend = (1 << NRINVBITS) - 1;

        int ibendproj = 2.0 * bendproj + 0.5 * maxbend;
        if (ibendproj < 0)
          ibendproj = 0;
        if (ibendproj > maxbend)
          ibendproj = maxbend;

        table_.push_back(ibendproj);
      }
    }
  }

  positive_ = false;
  name_ = settings_.combined() ? "MP_" : "PR_";
  name_ += "ProjectionBend_" + TrackletConfigBuilder::LayerName(N_LAYER + idisk) + ".tab";

  writeTable();
}

void TrackletLUT::initBendMatch(unsigned int layerdisk) {
  unsigned int nrinv = NRINVBITS;
  double rinvhalf = 0.5 * ((1 << nrinv) - 1);

  bool barrel = layerdisk < N_LAYER;
  bool isPSmodule = layerdisk < N_PSLAYER;
  double stripPitch = settings_.stripPitch(isPSmodule);

  if (barrel) {
    unsigned int nbits = isPSmodule ? N_BENDBITS_PS : N_BENDBITS_2S;

    for (unsigned int irinv = 0; irinv < (1u << nrinv); irinv++) {
      double rinv = (irinv - rinvhalf) * (1 << (settings_.nbitsrinv() - nrinv)) * settings_.krinvpars();

      double projbend = bendstrip(settings_.rmean(layerdisk), rinv, stripPitch);
      for (unsigned int ibend = 0; ibend < (1u << nbits); ibend++) {
        double stubbend = settings_.benddecode(ibend, layerdisk, isPSmodule);
        bool pass = std::abs(stubbend - projbend) < settings_.bendcutme(ibend, layerdisk, isPSmodule);
        table_.push_back(pass);
      }
    }
  } else {
    for (unsigned int iprojbend = 0; iprojbend < (1u << nrinv); iprojbend++) {
      double projbend = 0.5 * (iprojbend - rinvhalf);
      for (unsigned int ibend = 0; ibend < (1 << N_BENDBITS_2S); ibend++) {
        double stubbend = settings_.benddecode(ibend, layerdisk, false);
        bool pass = std::abs(stubbend - projbend) < settings_.bendcutme(ibend, layerdisk, false);
        table_.push_back(pass);
      }
    }
    for (unsigned int iprojbend = 0; iprojbend < (1u << nrinv); iprojbend++) {
      double projbend = 0.5 * (iprojbend - rinvhalf);
      for (unsigned int ibend = 0; ibend < (1 << N_BENDBITS_PS); ibend++) {
        double stubbend = settings_.benddecode(ibend, layerdisk, true);
        bool pass = std::abs(stubbend - projbend) < settings_.bendcutme(ibend, layerdisk, true);
        table_.push_back(pass);
      }
    }
  }

  positive_ = false;

  name_ = "METable_" + TrackletConfigBuilder::LayerName(layerdisk) + ".tab";

  writeTable();
}

void TrackletLUT::initVMRTable(unsigned int layerdisk, VMRTableType type, int region) {
  unsigned int zbits = settings_.vmrlutzbits(layerdisk);
  unsigned int rbits = settings_.vmrlutrbits(layerdisk);

  unsigned int rbins = (1 << rbits);
  unsigned int zbins = (1 << zbits);

  double zmin, zmax, rmin, rmax;

  if (layerdisk < N_LAYER) {
    zmin = -settings_.zlength();
    zmax = settings_.zlength();
    rmin = settings_.rmean(layerdisk) - settings_.drmax();
    rmax = settings_.rmean(layerdisk) + settings_.drmax();
  } else {
    rmin = 0;
    rmax = settings_.rmaxdisk();
    zmin = settings_.zmean(layerdisk - N_LAYER) - settings_.dzmax();
    zmax = settings_.zmean(layerdisk - N_LAYER) + settings_.dzmax();
  }

  double dr = (rmax - rmin) / rbins;
  double dz = (zmax - zmin) / zbins;

  int NBINS = settings_.NLONGVMBINS() * settings_.NLONGVMBINS();

  for (unsigned int izbin = 0; izbin < zbins; izbin++) {
    for (unsigned int irbin = 0; irbin < rbins; irbin++) {
      double r = rmin + (irbin + 0.5) * dr;
      double z = zmin + (izbin + 0.5) * dz;

      if (settings_.combined()) {
        int iznew = izbin - (1 << (zbits - 1));
        if (iznew < 0)
          iznew += (1 << zbits);
        assert(iznew >= 0);
        assert(iznew < (1 << zbits));
        z = zmin + (iznew + 0.5) * dz;
        if (layerdisk < N_LAYER) {
          int irnew = irbin - (1 << (rbits - 1));
          if (irnew < 0)
            irnew += (1 << rbits);
          assert(irnew >= 0);
          assert(irnew < (1 << rbits));
          r = rmin + (irnew + 0.5) * dr;
        }
      }

      if (layerdisk >= N_LAYER && irbin < 10)  //special case for the tabulated radii in 2S disks
        r = (layerdisk < N_LAYER + 2) ? settings_.rDSSinner(irbin) : settings_.rDSSouter(irbin);

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

      if (type == VMRTableType::me) {
        table_.push_back(bin);
      }

      if (type == VMRTableType::disk) {
        if (layerdisk >= N_LAYER) {
          double rproj = r * settings_.zmean(layerdisk - N_LAYER) / z;
          bin = 0.5 * NBINS * (rproj - settings_.rmindiskvm()) / (settings_.rmaxdiskvm() - settings_.rmindiskvm());
          //bin value of zero indicates that stub is out of range
          if (bin < 0)
            bin = 0;
          if (bin >= NBINS / 2)
            bin = 0;
          table_.push_back(bin);
        }
      }

      if (type == VMRTableType::inner) {
        if (layerdisk == LayerDisk::L1 || layerdisk == LayerDisk::L3 || layerdisk == LayerDisk::L5 ||
            layerdisk == LayerDisk::D1 || layerdisk == LayerDisk::D3) {
          table_.push_back(getVMRLookup(layerdisk + 1, z, r, dz, dr));
        }
        if (layerdisk == LayerDisk::L2) {
          table_.push_back(getVMRLookup(layerdisk + 1, z, r, dz, dr, Seed::L2L3));
        }
      }

      if (type == VMRTableType::inneroverlap) {
        if (layerdisk == LayerDisk::L1 || layerdisk == LayerDisk::L2) {
          table_.push_back(getVMRLookup(6, z, r, dz, dr, layerdisk + 6));
        }
      }

      if (type == VMRTableType::innerthird) {
        if (layerdisk == LayerDisk::L2) {  //projection from L2 to D1 for L2L3D1 seeding
          table_.push_back(getVMRLookup(LayerDisk::D1, z, r, dz, dr, Seed::L2L3D1));
        }

        if (layerdisk == LayerDisk::L5) {  //projection from L5 to L4 for L5L6L4 seeding
          table_.push_back(getVMRLookup(LayerDisk::L4, z, r, dz, dr));
        }

        if (layerdisk == LayerDisk::L3) {  //projection from L3 to L5 for L3L4L2 seeding
          table_.push_back(getVMRLookup(LayerDisk::L2, z, r, dz, dr));
        }

        if (layerdisk == LayerDisk::D1) {  //projection from D1 to L2 for D1D2L2 seeding
          table_.push_back(getVMRLookup(LayerDisk::L2, z, r, dz, dr));
        }
      }
    }
  }

  if (settings_.combined()) {
    if (type == VMRTableType::me) {
      positive_ = false;
      name_ = "VMRME_" + TrackletConfigBuilder::LayerName(layerdisk) + ".tab";
    }
    if (type == VMRTableType::disk) {
      positive_ = false;
      name_ = "VMRTE_" + TrackletConfigBuilder::LayerName(layerdisk) + ".tab";
    }
    if (type == VMRTableType::inner) {
      positive_ = true;
      nbits_ = 10;
      name_ = "TP_" + TrackletConfigBuilder::LayerName(layerdisk) + TrackletConfigBuilder::LayerName(layerdisk + 1) +
              ".tab";
    }

    if (type == VMRTableType::inneroverlap) {
      positive_ = true;
      nbits_ = 10;
      name_ = "TP_" + TrackletConfigBuilder::LayerName(layerdisk) + TrackletConfigBuilder::LayerName(N_LAYER) + ".tab";
    }

  } else {
    if (type == VMRTableType::me) {
      //This if a hack where the same memory is used in both ME and TE modules
      if (layerdisk == LayerDisk::L2 || layerdisk == LayerDisk::L3 || layerdisk == LayerDisk::L4 ||
          layerdisk == LayerDisk::L6) {
        positive_ = false;
        name_ = "VMTableOuter" + TrackletConfigBuilder::LayerName(layerdisk) + ".tab";
        writeTable();
      }

      assert(region >= 0);
      char cregion = 'A' + region;
      name_ = "VMR_" + TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_finebin.tab";
      positive_ = false;
    }

    if (type == VMRTableType::inner) {
      positive_ = false;
      name_ = "VMTableInner" + TrackletConfigBuilder::LayerName(layerdisk) +
              TrackletConfigBuilder::LayerName(layerdisk + 1) + ".tab";
    }

    if (type == VMRTableType::inneroverlap) {
      positive_ = false;
      name_ = "VMTableInner" + TrackletConfigBuilder::LayerName(layerdisk) + TrackletConfigBuilder::LayerName(N_LAYER) +
              ".tab";
    }

    if (type == VMRTableType::disk) {
      positive_ = false;
      name_ = "VMTableOuter" + TrackletConfigBuilder::LayerName(layerdisk) + ".tab";
    }
  }

  writeTable();
}

int TrackletLUT::getVMRLookup(unsigned int layerdisk, double z, double r, double dz, double dr, int iseed) const {
  double z0cut = settings_.z0cut();

  if (layerdisk < N_LAYER) {
    if (iseed == Seed::L2L3 && std::abs(z) < 52.0)
      return -1;

    double rmean = settings_.rmean(layerdisk);

    double rratio1 = rmean / (r + 0.5 * dr);
    double rratio2 = rmean / (r - 0.5 * dr);

    double z1 = (z - 0.5 * dz) * rratio1 + z0cut * (rratio1 - 1.0);
    double z2 = (z + 0.5 * dz) * rratio1 + z0cut * (rratio1 - 1.0);
    double z3 = (z - 0.5 * dz) * rratio2 + z0cut * (rratio2 - 1.0);
    double z4 = (z + 0.5 * dz) * rratio2 + z0cut * (rratio2 - 1.0);
    double z5 = (z - 0.5 * dz) * rratio1 - z0cut * (rratio1 - 1.0);
    double z6 = (z + 0.5 * dz) * rratio1 - z0cut * (rratio1 - 1.0);
    double z7 = (z - 0.5 * dz) * rratio2 - z0cut * (rratio2 - 1.0);
    double z8 = (z + 0.5 * dz) * rratio2 - z0cut * (rratio2 - 1.0);

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

    double r1 = (r + 0.5 * dr) * (zmean + z0cut) / (z + 0.5 * dz + z0cut);
    double r2 = (r - 0.5 * dr) * (zmean - z0cut) / (z + 0.5 * dz - z0cut);
    double r3 = (r + 0.5 * dr) * (zmean + z0cut) / (z - 0.5 * dz + z0cut);
    double r4 = (r - 0.5 * dr) * (zmean - z0cut) / (z - 0.5 * dz - z0cut);
    double r5 = (r + 0.5 * dr) * (zmean - z0cut) / (z + 0.5 * dz - z0cut);
    double r6 = (r - 0.5 * dr) * (zmean + z0cut) / (z + 0.5 * dz + z0cut);
    double r7 = (r + 0.5 * dr) * (zmean - z0cut) / (z - 0.5 * dz - z0cut);
    double r8 = (r - 0.5 * dr) * (zmean + z0cut) / (z - 0.5 * dz + z0cut);

    double rmin = std::min({r1, r2, r3, r4, r5, r6, r7, r8});
    double rmax = std::max({r1, r2, r3, r4, r5, r6, r7, r8});

    int NBINS = settings_.NLONGVMBINS() * settings_.NLONGVMBINS() / 2;

    double rmindisk = settings_.rmindiskvm();
    double rmaxdisk = settings_.rmaxdiskvm();

    if (iseed == Seed::L1D1)
      rmaxdisk = settings_.rmaxdiskl1overlapvm();
    if (iseed == Seed::L2D1)
      rmindisk = settings_.rmindiskl2overlapvm();
    if (iseed == Seed::L2L3D1)
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

    if (iseed == Seed::L2L3D1) {
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

    bool overlap = iseed == Seed::L1D1 || iseed == Seed::L2D1 || iseed == Seed::L2L3D1;

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

void TrackletLUT::initPhiCorrTable(unsigned int layerdisk, unsigned int rbits) {
  bool psmodule = layerdisk < N_PSLAYER;

  unsigned int bendbits = psmodule ? N_BENDBITS_PS : N_BENDBITS_2S;

  unsigned int rbins = (1 << rbits);

  double rmean = settings_.rmean(layerdisk);
  double drmax = settings_.drmax();

  double dr = 2.0 * drmax / rbins;

  unsigned int bendbins = (1 << bendbits);

  for (unsigned int ibend = 0; ibend < bendbins; ibend++) {
    for (unsigned int irbin = 0; irbin < rbins; irbin++) {
      int value = getphiCorrValue(layerdisk, ibend, irbin, rmean, dr, drmax);
      table_.push_back(value);
    }
  }

  name_ = "VMPhiCorrL" + std::to_string(layerdisk + 1) + ".tab";
  nbits_ = 14;
  positive_ = false;

  writeTable();
}

int TrackletLUT::getphiCorrValue(
    unsigned int layerdisk, unsigned int ibend, unsigned int irbin, double rmean, double dr, double drmax) const {
  bool psmodule = layerdisk < N_PSLAYER;

  double bend = -settings_.benddecode(ibend, layerdisk, psmodule);

  //for the rbin - calculate the distance to the nominal layer radius
  double Delta = (irbin + 0.5) * dr - drmax;

  //calculate the phi correction - this is a somewhat approximate formula
  double dphi = (Delta / 0.18) * bend * settings_.stripPitch(psmodule) / rmean;

  double kphi = psmodule ? settings_.kphi() : settings_.kphi1();

  int idphi = dphi / kphi;

  return idphi;
}

// Write LUT table.
void TrackletLUT::writeTable() const {
  if (!settings_.writeTable()) {
    return;
  }

  if (name_.empty()) {
    return;
  }

  ofstream out = openfile(settings_.tablePath(), name_, __FILE__, __LINE__);

  out << "{" << endl;
  for (unsigned int i = 0; i < table_.size(); i++) {
    if (i != 0) {
      out << "," << endl;
    }

    int itable = table_[i];
    if (positive_) {
      if (table_[i] < 0) {
        itable = (1 << nbits_) - 1;
      }
    }

    out << itable;
  }
  out << endl << "};" << endl;
  out.close();
}

int TrackletLUT::lookup(unsigned int index) const {
  assert(index < table_.size());
  return table_[index];
}
