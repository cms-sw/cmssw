#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletConfigBuilder.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <filesystem>

using namespace std;
using namespace trklet;

TrackletLUT::TrackletLUT(const Settings& settings) : settings_(settings), setup_(settings.setup()) {}

std::vector<const tt::SensorModule*> TrackletLUT::getSensorModules(
    unsigned int layerdisk, bool isPS, std::array<double, 2> tan_range, unsigned int nzbins, unsigned int zbin) {
  //Returns a vector of SensorModules using T. Schuh's Setup and SensorModule classes.
  //Can be used 3 ways:
  //Default: No specified tan_range or nzbins, returns all SensorModules in specified layerdisk (unique in |z|)
  //tan_range: Returns modules in given tan range, where the min and max tan(theta) are measured from 0 -/+ z0 to account for displaced tracks
  //zbins: Returns modules in specified z bin (2 zbins = (Flat, Tilted), 13 zbins = (Flat, TR1, ..., TR12). Only for tilted barrel

  bool use_tan_range = !(tan_range[0] == -1 and tan_range[1] == -1);
  bool use_zbins = (nzbins > 1);

  bool barrel = layerdisk < N_LAYER;

  int layerId = barrel ? layerdisk + 1 : layerdisk + N_LAYER - 1;

  std::vector<const tt::SensorModule*> sensorModules;

  double z0 = settings_.z0cut();

  for (auto& sm : setup_->sensorModules()) {
    if (sm.layerId() != layerId || sm.z() < 0 || sm.psModule() != isPS) {
      continue;
    }

    if (use_tan_range) {
      const double term = (sm.numColumns() / 2 - 0.5) * sm.pitchCol();
      double rmin = sm.r() - term * std::abs(sm.sinTilt());
      double rmax = sm.r() + term * std::abs(sm.sinTilt());

      double zmin = std::abs(sm.z()) - term * std::abs(sm.cosTilt());
      double zmax = std::abs(sm.z()) + term * std::abs(sm.cosTilt());

      //z0_max is swapped here so that the comparison down 5 lines is from same origin (+/- z0)
      double mod_tan_max = tan_theta(rmin, zmax, z0, false);
      double mod_tan_min = tan_theta(rmax, zmin, z0, true);

      if (mod_tan_max >= tan_range[0] && mod_tan_min <= tan_range[1]) {
        sensorModules.push_back(&sm);
      }
    } else if (use_zbins) {
      assert(layerdisk < 3);

      if (nzbins == 2) {
        bool useFlat = (zbin == 0);
        bool isFlat = (sm.tilt() == 0);

        if (useFlat and isFlat)
          sensorModules.push_back(&sm);
        else if (!useFlat and !isFlat)
          sensorModules.push_back(&sm);
      } else if (nzbins == 13) {
        if (sm.ringId(setup_) == zbin)
          sensorModules.push_back(&sm);
      } else {
        throw cms::Exception("Unspecified number of z bins");
      }
    } else {
      sensorModules.push_back(&sm);
    }
  }

  //Remove Duplicate Modules
  static constexpr double delta = 1.e-3;
  auto smallerR = [](const tt::SensorModule* lhs, const tt::SensorModule* rhs) { return lhs->r() < rhs->r(); };
  auto smallerZ = [](const tt::SensorModule* lhs, const tt::SensorModule* rhs) { return lhs->z() < rhs->z(); };
  auto equalRZ = [](const tt::SensorModule* lhs, const tt::SensorModule* rhs) {
    return abs(lhs->r() - rhs->r()) < delta && abs(lhs->z() - rhs->z()) < delta;
  };
  stable_sort(sensorModules.begin(), sensorModules.end(), smallerR);
  stable_sort(sensorModules.begin(), sensorModules.end(), smallerZ);
  sensorModules.erase(unique(sensorModules.begin(), sensorModules.end(), equalRZ), sensorModules.end());

  return sensorModules;
}

std::array<double, 2> TrackletLUT::getTanRange(const std::vector<const tt::SensorModule*>& sensorModules) {
  //Given a set of modules returns a range in tan(theta), the angle is measured in the r-z(+/-z0) plane from the r-axis

  std::array<double, 2> tan_range = {{2147483647, 0}};  //(tan_min, tan_max)

  double z0 = settings_.z0cut();

  for (auto sm : sensorModules) {
    const double term = (sm->numColumns() / 2 - 0.5) * sm->pitchCol();
    double rmin = sm->r() - term * std::abs(sm->sinTilt());
    double rmax = sm->r() + term * std::abs(sm->sinTilt());

    double zmin = std::abs(sm->z()) - term * sm->pitchCol() * sm->cosTilt();
    double zmax = std::abs(sm->z()) + term * sm->cosTilt();

    double mod_tan_max = tan_theta(rmin, zmax, z0, true);  //(r, z, z0, bool z0_max), z0_max measures from +/- z0
    double mod_tan_min = tan_theta(rmax, zmin, z0, false);

    if (mod_tan_min < tan_range[0])
      tan_range[0] = mod_tan_min;
    if (mod_tan_max > tan_range[1])
      tan_range[1] = mod_tan_max;
  }
  return tan_range;
}

std::vector<std::array<double, 2>> TrackletLUT::getBendCut(unsigned int layerdisk,
                                                           const std::vector<const tt::SensorModule*>& sensorModules,
                                                           bool isPS,
                                                           double FEbendcut) {
  //Finds range of bendstrip for given SensorModules as a function of the encoded bend. Returns in format (mid, half_range).
  //This uses the stub windows provided by T. Schuh's SensorModule class to determine the bend encoding. TODO test changes in stub windows
  //Any other change to the bend encoding requires changes here, perhaps a function that given (FEbend, isPS, stub window) and outputs an encoded bend
  //would be useful for consistency.

  unsigned int bendbits = isPS ? 3 : 4;

  std::vector<std::array<double, 2>> bendpars;    // mid, cut
  std::vector<std::array<double, 2>> bendminmax;  // min, max

  //Initialize array
  for (int i = 0; i < 1 << bendbits; i++) {
    bendpars.push_back({{99, 0}});
    bendminmax.push_back({{99, -99}});
  }

  //Loop over modules
  for (auto sm : sensorModules) {
    int window = sm->windowSize();  //Half-strip units
    const vector<double>& encodingBend = setup_->encodingBend(window, isPS);

    //Loop over FEbends
    for (int ibend = 0; ibend <= 2 * window; ibend++) {
      int FEbend = ibend - window;                                                 //Half-strip units
      double BEbend = setup_->stubAlgorithm()->degradeBend(isPS, window, FEbend);  //Full strip units

      const auto pos = std::find(encodingBend.begin(), encodingBend.end(), std::abs(BEbend));
      int bend = std::signbit(BEbend) ? (1 << bendbits) - distance(encodingBend.begin(), pos)
                                      : distance(encodingBend.begin(), pos);  //Encoded bend

      double bendmin = FEbend / 2.0 - FEbendcut;  //Full Strip units
      double bendmax = FEbend / 2.0 + FEbendcut;

      //Convert to bendstrip, calculate at module edges (z min, r max) and  (z max, r min)
      double z_mod[2];
      double r_mod[2];

      z_mod[0] = std::abs(sm->z()) + (sm->numColumns() / 2 - 0.5) * sm->pitchCol() * sm->cosTilt();  //z max
      z_mod[1] = std::abs(sm->z()) - (sm->numColumns() / 2 - 0.5) * sm->pitchCol() * sm->cosTilt();  //z min

      r_mod[0] = sm->r() - (sm->numColumns() / 2 - 0.5) * sm->pitchCol() * std::abs(sm->sinTilt());  //r min
      r_mod[1] = sm->r() + (sm->numColumns() / 2 - 0.5) * sm->pitchCol() * std::abs(sm->sinTilt());  //r max

      for (int i = 0; i < 2; i++) {  // 2 points to cover range in tan(theta) = z/r
        double CF = std::abs(sm->sinTilt()) * (z_mod[i] / r_mod[i]) + sm->cosTilt();

        double cbendmin =
            convertFEBend(bendmin, sm->sep(), settings_.sensorSpacing2S(), CF, (layerdisk < N_LAYER), r_mod[i]);
        double cbendmax =
            convertFEBend(bendmax, sm->sep(), settings_.sensorSpacing2S(), CF, (layerdisk < N_LAYER), r_mod[i]);

        if (cbendmin < bendminmax[bend][0])
          bendminmax.at(bend)[0] = cbendmin;
        if (cbendmax > bendminmax[bend][1])
          bendminmax.at(bend)[1] = cbendmax;
      }
    }
  }
  //Convert min, max to mid, cut for ease of use
  for (int i = 0; i < 1 << bendbits; i++) {
    double mid = (bendminmax[i][1] + bendminmax[i][0]) / 2;
    double cut = (bendminmax[i][1] - bendminmax[i][0]) / 2;

    bendpars[i][0] = mid;
    bendpars[i][1] = cut;
  }

  return bendpars;
}

void TrackletLUT::initmatchcut(unsigned int layerdisk, MatchType type, unsigned int region) {
  char cregion = 'A' + region;

  for (unsigned int iSeed = 0; iSeed < N_SEED; iSeed++) {
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
  if (type == alphainner) {
    for (unsigned int i = 0; i < N_DSS_MOD * 2; i++) {
      table_.push_back((1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                       (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSinner(i) * settings_.rDSSinner(i)) /
                       settings_.kphi());
    }
  }
  if (type == alphaouter) {
    for (unsigned int i = 0; i < N_DSS_MOD * 2; i++) {
      table_.push_back((1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                       (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSouter(i) * settings_.rDSSouter(i)) /
                       settings_.kphi());
    }
  }
  if (type == rSSinner) {
    for (unsigned int i = 0; i < N_DSS_MOD * 2; i++) {
      table_.push_back(settings_.rDSSinner(i) / settings_.kr());
    }
  }
  if (type == rSSouter) {
    for (unsigned int i = 0; i < N_DSS_MOD * 2; i++) {
      table_.push_back(settings_.rDSSouter(i) / settings_.kr());
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
  if (type == alphainner) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_alphainner.tab";
  }
  if (type == alphaouter) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_alphaouter.tab";
  }
  if (type == rSSinner) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_rDSSinner.tab";
  }
  if (type == rSSouter) {
    name_ += TrackletConfigBuilder::LayerName(layerdisk) + "PHI" + cregion + "_rDSSouter.tab";
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

  bool isPSinner;
  bool isPSouter;

  if (iSeed == Seed::L3L4) {
    isPSinner = true;
    isPSouter = false;
  } else if (iSeed == Seed::L5L6) {
    isPSinner = false;
    isPSouter = false;
  } else {
    isPSinner = true;
    isPSouter = true;
  }

  unsigned int nbendbitsinner = isPSinner ? N_BENDBITS_PS : N_BENDBITS_2S;
  unsigned int nbendbitsouter = isPSouter ? N_BENDBITS_PS : N_BENDBITS_2S;

  double z0 = settings_.z0cut();

  int nbinsfinephidiff = (1 << nbitsfinephidiff);

  for (int iphibin = 0; iphibin < nbinsfinephidiff; iphibin++) {
    int iphidiff = iphibin;
    if (iphibin >= nbinsfinephidiff / 2) {
      iphidiff = iphibin - nbinsfinephidiff;
    }
    //min and max dphi
    //ramge of dphi to consider due to resolution
    double deltaphi = 1.5;
    dphi[0] = (iphidiff - deltaphi) * dfinephi;
    dphi[1] = (iphidiff + deltaphi) * dfinephi;
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

      //Determine bend cuts using geometry
      std::vector<std::array<double, 2>> bend_cuts_inner;
      std::vector<std::array<double, 2>> bend_cuts_outer;

      if (settings_.useCalcBendCuts) {
        std::vector<const tt::SensorModule*> sminner;
        std::vector<const tt::SensorModule*> smouter;

        if (iSeed == Seed::L1L2 || iSeed == Seed::L2L3 || iSeed == Seed::L3L4 || iSeed == Seed::L5L6) {
          double outer_tan_max = tan_theta(settings_.rmean(layerdisk2), settings_.zlength(), z0, true);
          std::array<double, 2> tan_range = {{0, outer_tan_max}};

          smouter = getSensorModules(layerdisk2, isPSouter, tan_range);
          sminner = getSensorModules(layerdisk1, isPSinner, tan_range);

        } else if (iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
          double outer_tan_min = tan_theta(router[1], settings_.zmindisk(layerdisk2 - N_LAYER), z0, false);
          double outer_tan_max = tan_theta(router[0], settings_.zmaxdisk(layerdisk2 - N_LAYER), z0, true);

          smouter = getSensorModules(layerdisk2, isPSouter, {{outer_tan_min, outer_tan_max}});
          std::array<double, 2> tan_range = getTanRange(smouter);
          sminner = getSensorModules(layerdisk1, isPSinner, tan_range);

        } else {  // D1D2 D3D4

          double outer_tan_min = tan_theta(router[1], settings_.zmindisk(layerdisk2 - N_LAYER), z0, false);
          double outer_tan_max = tan_theta(router[0], settings_.zmaxdisk(layerdisk2 - N_LAYER), z0, true);

          smouter = getSensorModules(layerdisk2, isPSouter, {{outer_tan_min, outer_tan_max}});

          std::array<double, 2> tan_range = getTanRange(smouter);
          sminner = getSensorModules(layerdisk1, isPSinner, tan_range);
        }

        bend_cuts_inner = getBendCut(layerdisk1, sminner, isPSinner, settings_.bendcutTE(iSeed, true));
        bend_cuts_outer = getBendCut(layerdisk2, smouter, isPSouter, settings_.bendcutTE(iSeed, false));

      } else {
        for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
          double mid = settings_.benddecode(ibend, layerdisk1, isPSinner);
          double cut = settings_.bendcutte(ibend, layerdisk1, isPSinner);
          bend_cuts_inner.push_back({{mid, cut}});
        }
        for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
          double mid = settings_.benddecode(ibend, layerdisk2, isPSouter);
          double cut = settings_.bendcutte(ibend, layerdisk2, isPSouter);
          bend_cuts_outer.push_back({{mid, cut}});
        }
      }

      double bendinnermin = 20.0;
      double bendinnermax = -20.0;
      double bendoutermin = 20.0;
      double bendoutermax = -20.0;
      double rinvmin = 1.0;
      double rinvmax = -1.0;
      double absrinvmin = 1.0;

      for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
          double rinner = 0.0;
          if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4) {
            rinner = router[i3] * settings_.zmean(layerdisk1 - N_LAYER) / settings_.zmean(layerdisk2 - N_LAYER);
          } else {
            rinner = settings_.rmean(layerdisk1);
          }
          if (settings_.useCalcBendCuts) {
            if (rinner >= router[i3])
              continue;
          }
          double rinv1 = (rinner < router[i3]) ? rinv(0.0, -dphi[i2], rinner, router[i3]) : 20.0;
          double pitchinner = (rinner < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
          double pitchouter =
              (router[i3] < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
          double abendinner = bendstrip(rinner, rinv1, pitchinner, settings_.sensorSpacing2S());
          double abendouter = bendstrip(router[i3], rinv1, pitchouter, settings_.sensorSpacing2S());
          if (abendinner < bendinnermin)
            bendinnermin = abendinner;
          if (abendinner > bendinnermax)
            bendinnermax = abendinner;
          if (abendouter < bendoutermin)
            bendoutermin = abendouter;
          if (abendouter > bendoutermax)
            bendoutermax = abendouter;
          if (std::abs(rinv1) < absrinvmin)
            absrinvmin = std::abs(rinv1);
          if (rinv1 > rinvmax)
            rinvmax = rinv1;
          if (rinv1 < rinvmin)
            rinvmin = rinv1;
        }
      }

      bool passptcut;
      double bendfac;
      double rinvcutte = settings_.rinvcutte();

      if (settings_.useCalcBendCuts) {
        double lowrinvcutte =
            rinvcutte / 3;  //Somewhat arbitrary value, allows for better acceptance in bins with low rinv (high pt)
        passptcut = rinvmin < rinvcutte and rinvmax > -rinvcutte;
        bendfac = (rinvmin < lowrinvcutte and rinvmax > -lowrinvcutte)
                      ? 1.05
                      : 1.0;  //Somewhat arbirary value, bend cuts are 5% larger in bins with low rinv (high pt)
      } else {
        passptcut = absrinvmin < rinvcutte;
        bendfac = 1.0;
      }

      if (fillInner) {
        for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
          double bendminfac = (isPSinner and (ibend == 2 or ibend == 3)) ? bendfac : 1.0;
          double bendmaxfac = (isPSinner and (ibend == 6 or ibend == 5)) ? bendfac : 1.0;

          double mid = bend_cuts_inner.at(ibend)[0];
          double cut = bend_cuts_inner.at(ibend)[1];

          bool passinner = mid + cut * bendmaxfac > bendinnermin && mid - cut * bendminfac < bendinnermax;

          table_.push_back(passinner && passptcut);
        }
      } else {
        for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
          double bendminfac = (isPSouter and (ibend == 2 or ibend == 3)) ? bendfac : 1.0;
          double bendmaxfac = (isPSouter and (ibend == 6 or ibend == 5)) ? bendfac : 1.0;

          double mid = bend_cuts_outer.at(ibend)[0];
          double cut = bend_cuts_outer.at(ibend)[1];

          bool passouter = mid + cut * bendmaxfac > bendoutermin && mid - cut * bendminfac < bendoutermax;

          table_.push_back(passouter && passptcut);
        }
      }
    }
  }

  nbits_ = 8;

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
            int ptinnerindexnew = (idphi1 << nbendbitsinner) + innerbend;
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

  bool isPSinner;
  bool isPSouter;

  if (iSeed == Seed::L3L4) {
    isPSinner = true;
    isPSouter = false;
  } else if (iSeed == Seed::L5L6) {
    isPSinner = false;
    isPSouter = false;
  } else {
    isPSinner = true;
    isPSouter = true;
  }

  unsigned int nbendbitsinner = isPSinner ? N_BENDBITS_PS : N_BENDBITS_2S;
  unsigned int nbendbitsouter = isPSouter ? N_BENDBITS_PS : N_BENDBITS_2S;

  if (fillTEMem) {
    if (fillInner) {
      table_.resize((1 << nbendbitsinner), false);
    } else {
      table_.resize((1 << nbendbitsouter), false);
    }
  }

  double z0 = settings_.z0cut();

  for (int irouterbin = 0; irouterbin < outerrbins; irouterbin++) {
    if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4 || iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
      router[0] = settings_.rmindiskvm() + irouterbin * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
      router[1] =
          settings_.rmindiskvm() + (irouterbin + 1) * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
    } else {
      router[0] = settings_.rmean(layerdisk2);
      router[1] = settings_.rmean(layerdisk2);
    }

    //Determine bend cuts using geometry
    std::vector<std::array<double, 2>> bend_cuts_inner;
    std::vector<std::array<double, 2>> bend_cuts_outer;

    if (settings_.useCalcBendCuts) {
      std::vector<const tt::SensorModule*> sminner;
      std::vector<const tt::SensorModule*> smouter;

      if (iSeed == Seed::L1L2 || iSeed == Seed::L2L3 || iSeed == Seed::L3L4 || iSeed == Seed::L5L6) {
        double outer_tan_max = tan_theta(settings_.rmean(layerdisk2), settings_.zlength(), z0, true);
        std::array<double, 2> tan_range = {{0, outer_tan_max}};

        smouter = getSensorModules(layerdisk2, isPSouter, tan_range);
        sminner = getSensorModules(layerdisk1, isPSinner, tan_range);

      } else if (iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
        double outer_tan_min = tan_theta(router[1], settings_.zmindisk(layerdisk2 - N_LAYER), z0, false);
        double outer_tan_max = tan_theta(router[0], settings_.zmaxdisk(layerdisk2 - N_LAYER), z0, true);

        smouter = getSensorModules(layerdisk2, isPSouter, {{outer_tan_min, outer_tan_max}});
        std::array<double, 2> tan_range = getTanRange(smouter);
        sminner = getSensorModules(layerdisk1, isPSinner, tan_range);

      } else {  // D1D2 D3D4

        double outer_tan_min = tan_theta(router[1], settings_.zmindisk(layerdisk2 - N_LAYER), z0, false);
        double outer_tan_max = tan_theta(router[0], settings_.zmaxdisk(layerdisk2 - N_LAYER), z0, true);

        smouter = getSensorModules(layerdisk2, isPSouter, {{outer_tan_min, outer_tan_max}});

        std::array<double, 2> tan_range = getTanRange(smouter);
        sminner = getSensorModules(layerdisk1, isPSinner, tan_range);
      }

      bend_cuts_inner = getBendCut(layerdisk1, sminner, isPSinner, settings_.bendcutTE(iSeed, true));
      bend_cuts_outer = getBendCut(layerdisk2, smouter, isPSouter, settings_.bendcutTE(iSeed, false));

    } else {
      for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
        double mid = settings_.benddecode(ibend, layerdisk1, nbendbitsinner == 3);
        double cut = settings_.bendcutte(ibend, layerdisk1, nbendbitsinner == 3);
        bend_cuts_inner.push_back({{mid, cut}});
      }
      for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
        double mid = settings_.benddecode(ibend, layerdisk2, nbendbitsouter == 3);
        double cut = settings_.bendcutte(ibend, layerdisk2, nbendbitsouter == 3);
        bend_cuts_outer.push_back({{mid, cut}});
      }
    }

    for (int iphiinnerbin = 0; iphiinnerbin < innerphibins; iphiinnerbin++) {
      phiinner[0] = innerphimin + iphiinnerbin * (innerphimax - innerphimin) / innerphibins;
      phiinner[1] = innerphimin + (iphiinnerbin + 1) * (innerphimax - innerphimin) / innerphibins;
      for (int iphiouterbin = 0; iphiouterbin < outerphibins; iphiouterbin++) {
        phiouter[0] = outerphimin + iphiouterbin * (outerphimax - outerphimin) / outerphibins;
        phiouter[1] = outerphimin + (iphiouterbin + 1) * (outerphimax - outerphimin) / outerphibins;

        double bendinnermin = 20.0;
        double bendinnermax = -20.0;
        double bendoutermin = 20.0;
        double bendoutermax = -20.0;
        double rinvmin = 1.0;
        double rinvmax = -1.0;
        double absrinvmin = 1.0;

        for (int i1 = 0; i1 < 2; i1++) {
          for (int i2 = 0; i2 < 2; i2++) {
            for (int i3 = 0; i3 < 2; i3++) {
              double rinner = 0.0;
              if (iSeed == Seed::D1D2 || iSeed == Seed::D3D4) {
                rinner = router[i3] * settings_.zmean(layerdisk1 - N_LAYER) / settings_.zmean(layerdisk2 - N_LAYER);
              } else {
                rinner = settings_.rmean(layerdisk1);
              }

              if (settings_.useCalcBendCuts) {
                if (rinner >= router[i3])
                  continue;
              }

              double rinv1 = (rinner < router[i3]) ? -rinv(phiinner[i1], phiouter[i2], rinner, router[i3]) : -20.0;
              double pitchinner =
                  (rinner < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
              double pitchouter =
                  (router[i3] < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);

              double abendinner = bendstrip(rinner, rinv1, pitchinner, settings_.sensorSpacing2S());
              double abendouter = bendstrip(router[i3], rinv1, pitchouter, settings_.sensorSpacing2S());

              if (abendinner < bendinnermin)
                bendinnermin = abendinner;
              if (abendinner > bendinnermax)
                bendinnermax = abendinner;
              if (abendouter < bendoutermin)
                bendoutermin = abendouter;
              if (abendouter > bendoutermax)
                bendoutermax = abendouter;
              if (std::abs(rinv1) < absrinvmin)
                absrinvmin = std::abs(rinv1);
              if (rinv1 > rinvmax)
                rinvmax = rinv1;
              if (rinv1 < rinvmin)
                rinvmin = rinv1;
            }
          }
        }

        double lowrinvcutte = 0.002;

        bool passptcut;
        double bendfac;

        if (settings_.useCalcBendCuts) {
          passptcut = rinvmin < settings_.rinvcutte() and rinvmax > -settings_.rinvcutte();
          bendfac = (rinvmin < lowrinvcutte and rinvmax > -lowrinvcutte) ? 1.05 : 1.0;  // Better acceptance for high pt
        } else {
          passptcut = absrinvmin < settings_.rinvcutte();
          bendfac = 1.0;
        }

        if (fillInner) {
          for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
            double bendminfac = (isPSinner and (ibend == 2 or ibend == 3)) ? bendfac : 1.0;
            double bendmaxfac = (isPSinner and (ibend == 6 or ibend == 5)) ? bendfac : 1.0;

            double mid = bend_cuts_inner.at(ibend)[0];
            double cut = bend_cuts_inner.at(ibend)[1];

            bool passinner = mid + cut * bendmaxfac > bendinnermin && mid - cut * bendminfac < bendinnermax;

            if (fillTEMem) {
              if (passinner)
                table_[ibend] = 1;
            } else {
              table_.push_back(passinner && passptcut);
            }
          }
        } else {
          for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
            double bendminfac = (isPSouter and (ibend == 2 or ibend == 3)) ? bendfac : 1.0;
            double bendmaxfac = (isPSouter and (ibend == 6 or ibend == 5)) ? bendfac : 1.0;

            double mid = bend_cuts_outer.at(ibend)[0];
            double cut = bend_cuts_outer.at(ibend)[1];

            bool passouter = mid + cut * bendmaxfac > bendoutermin && mid - cut * bendminfac < bendoutermax;

            if (fillTEMem) {
              if (passouter)
                table_[ibend] = 1;
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
      ir = ir << (settings_.nrbitsstub(N_LAYER) - nrbits);
      for (unsigned int iphiderbin = 0; iphiderbin < nphiderbins; iphiderbin++) {
        int iphider = iphiderbin;
        if (iphider > (1 << (nphiderbits - 1)))
          iphider -= (1 << nphiderbits);
        iphider = iphider << (settings_.nbitsphiprojderL123() - nphiderbits);

        double rproj = ir * settings_.krprojshiftdisk();
        double phider = iphider * k_phider;
        double t = settings_.zmean(idisk) / rproj;

        if (isignbin)
          t = -t;

        double rinv = -phider * (2.0 * t);

        double stripPitch = (rproj < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
        double bendproj = bendstrip(rproj, rinv, stripPitch, settings_.sensorSpacing2S());

        constexpr double maxbend = (1 << NRINVBITS) - 1;

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

void TrackletLUT::initProjectionDiskRadius(int nrbits) {
  //When a projection to a disk is considered this offset and added and subtracted to calculate
  //the bin the projection is pointing to. This is to account for resolution effects such that
  //projections that are near a bin boundary will be assigned to both bins. The value (3 cm) should
  //cover the uncertanty in the resolution.
  double roffset = 3.0;

  for (unsigned int ir = 0; ir < (1u << nrbits); ir++) {
    double r = ir * settings_.rmaxdisk() / (1u << nrbits);

    int rbin1 =
        (1 << N_RZBITS) * (r - roffset - settings_.rmindiskvm()) / (settings_.rmaxdisk() - settings_.rmindiskvm());
    int rbin2 =
        (1 << N_RZBITS) * (r + roffset - settings_.rmindiskvm()) / (settings_.rmaxdisk() - settings_.rmindiskvm());

    if (rbin1 < 0) {
      rbin1 = 0;
    }
    rbin2 = clamp(rbin2, 0, ((1 << N_RZBITS) - 1));

    assert(rbin1 <= rbin2);
    assert(rbin2 - rbin1 <= 1);

    int d = rbin1 != rbin2;

    int finer =
        (1 << (N_RZBITS + NFINERZBITS)) *
        ((r - settings_.rmindiskvm()) - rbin1 * (settings_.rmaxdisk() - settings_.rmindiskvm()) / (1 << N_RZBITS)) /
        (settings_.rmaxdisk() - settings_.rmindiskvm());

    finer = clamp(finer, 0, ((1 << (NFINERZBITS + 1)) - 1));

    //Pack the data in a 8 bit word (ffffrrrd) where f is finer, r is rbin1, and d is difference
    int N_DIFF_FLAG = 1;  // Single bit for bool flag

    int word = (finer << (N_RZBITS + N_DIFF_FLAG)) + (rbin1 << N_DIFF_FLAG) + d;

    table_.push_back(word);
  }

  //Size of the data word from above (8 bits)
  nbits_ = NFINERZBITS + 1 + N_RZBITS + 1;
  positive_ = true;
  name_ = "ProjectionDiskRadius.tab";
  writeTable();
}

void TrackletLUT::initBendMatch(unsigned int layerdisk) {
  unsigned int nrinv = NRINVBITS;
  double rinvhalf = 0.5 * ((1 << nrinv) - 1);

  bool barrel = layerdisk < N_LAYER;

  if (barrel) {
    bool isPSmodule = layerdisk < N_PSLAYER;
    double stripPitch = settings_.stripPitch(isPSmodule);
    unsigned int nbits = isPSmodule ? N_BENDBITS_PS : N_BENDBITS_2S;

    std::vector<std::array<double, 2>> bend_cuts;

    if (settings_.useCalcBendCuts) {
      double bendcutFE = settings_.bendcutME(layerdisk, isPSmodule);
      std::vector<const tt::SensorModule*> sm = getSensorModules(layerdisk, isPSmodule);
      bend_cuts = getBendCut(layerdisk, sm, isPSmodule, bendcutFE);

    } else {
      for (unsigned int ibend = 0; ibend < (1u << nbits); ibend++) {
        double mid = settings_.benddecode(ibend, layerdisk, isPSmodule);
        double cut = settings_.bendcutte(ibend, layerdisk, isPSmodule);
        bend_cuts.push_back({{mid, cut}});
      }
    }

    for (unsigned int irinv = 0; irinv < (1u << nrinv); irinv++) {
      double rinv = (irinv - rinvhalf) * (1 << (settings_.nbitsrinv() - nrinv)) * settings_.krinvpars();

      double projbend = bendstrip(settings_.rmean(layerdisk), rinv, stripPitch, settings_.sensorSpacing2S());
      for (unsigned int ibend = 0; ibend < (1u << nbits); ibend++) {
        double mid = bend_cuts[ibend][0];
        double cut = bend_cuts[ibend][1];

        double pass = mid + cut > projbend && mid - cut < projbend;

        table_.push_back(pass);
      }
    }
  } else {
    std::vector<std::array<double, 2>> bend_cuts_2S;
    std::vector<std::array<double, 2>> bend_cuts_PS;

    if (settings_.useCalcBendCuts) {
      double bendcutFE2S = settings_.bendcutME(layerdisk, false);
      std::vector<const tt::SensorModule*> sm2S = getSensorModules(layerdisk, false);
      bend_cuts_2S = getBendCut(layerdisk, sm2S, false, bendcutFE2S);

      double bendcutFEPS = settings_.bendcutME(layerdisk, true);
      std::vector<const tt::SensorModule*> smPS = getSensorModules(layerdisk, true);
      bend_cuts_PS = getBendCut(layerdisk, smPS, true, bendcutFEPS);

    } else {
      for (unsigned int ibend = 0; ibend < (1 << N_BENDBITS_2S); ibend++) {
        double mid = settings_.benddecode(ibend, layerdisk, false);
        double cut = settings_.bendcutme(ibend, layerdisk, false);
        bend_cuts_2S.push_back({{mid, cut}});
      }
      for (unsigned int ibend = 0; ibend < (1 << N_BENDBITS_PS); ibend++) {
        double mid = settings_.benddecode(ibend, layerdisk, true);
        double cut = settings_.bendcutme(ibend, layerdisk, true);
        bend_cuts_PS.push_back({{mid, cut}});
      }
    }

    for (unsigned int iprojbend = 0; iprojbend < (1u << nrinv); iprojbend++) {
      double projbend = 0.5 * (iprojbend - rinvhalf);
      for (unsigned int ibend = 0; ibend < (1 << N_BENDBITS_2S); ibend++) {
        double mid = bend_cuts_2S[ibend][0];
        double cut = bend_cuts_2S[ibend][1];

        double pass = mid + cut > projbend && mid - cut < projbend;

        table_.push_back(pass);
      }
    }
    for (unsigned int iprojbend = 0; iprojbend < (1u << nrinv); iprojbend++) {  //Should this be binned in r?
      double projbend = 0.5 * (iprojbend - rinvhalf);
      for (unsigned int ibend = 0; ibend < (1 << N_BENDBITS_PS); ibend++) {
        double mid = bend_cuts_PS[ibend][0];
        double cut = bend_cuts_PS[ibend][1];

        double pass = mid + cut > projbend && mid - cut < projbend;

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

      unsigned int NRING =
          5;  //number of 2S rings in disks. This is multiplied below by two since we have two halfs of a module
      if (layerdisk >= N_LAYER && irbin < 2 * NRING)  //special case for the tabulated radii in 2S disks
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
      nbits_ = 2 * settings_.NLONGVMBITS();
      positive_ = false;
      name_ = "VMRME_" + TrackletConfigBuilder::LayerName(layerdisk) + ".tab";
    }
    if (type == VMRTableType::disk) {
      nbits_ = 2 * settings_.NLONGVMBITS();
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
    double constexpr zcutL2L3 = 52.0;  //Stubs closer to IP in z will not be used for L2L3 seeds
    if (iseed == Seed::L2L3 && std::abs(z) < zcutL2L3)
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

  std::vector<std::array<double, 2>> bend_vals;

  if (settings_.useCalcBendCuts) {
    std::vector<const tt::SensorModule*> sm = getSensorModules(layerdisk, psmodule);
    bend_vals = getBendCut(layerdisk, sm, psmodule);

  } else {
    for (int ibend = 0; ibend < 1 << bendbits; ibend++) {
      bend_vals.push_back({{settings_.benddecode(ibend, layerdisk, layerdisk < N_PSLAYER), 0}});
    }
  }

  for (int ibend = 0; ibend < 1 << bendbits; ibend++) {
    for (unsigned int irbin = 0; irbin < rbins; irbin++) {
      double bend = -bend_vals[ibend][0];
      int value = getphiCorrValue(layerdisk, bend, irbin, rmean, dr, drmax);
      table_.push_back(value);
    }
  }

  name_ = "VMPhiCorrL" + std::to_string(layerdisk + 1) + ".tab";
  nbits_ = 14;
  positive_ = false;

  writeTable();
}

int TrackletLUT::getphiCorrValue(
    unsigned int layerdisk, double bend, unsigned int irbin, double rmean, double dr, double drmax) const {
  bool psmodule = layerdisk < N_PSLAYER;

  //for the rbin - calculate the distance to the nominal layer radius
  double Delta = (irbin + 0.5) * dr - drmax;

  //calculate the phi correction - this is a somewhat approximate formula
  double drnom = 0.18;  //This is the nominal module separation for which bend is referenced
  double dphi = (Delta / drnom) * bend * settings_.stripPitch(psmodule) / rmean;

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

  string name = name_;

  name[name_.size() - 3] = 'd';
  name[name_.size() - 2] = 'a';
  name[name_.size() - 1] = 't';

  out = openfile(settings_.tablePath(), name, __FILE__, __LINE__);

  int width = (nbits_ + 3) / 4;

  for (unsigned int i = 0; i < table_.size(); i++) {
    int itable = table_[i];
    if (positive_) {
      if (table_[i] < 0) {
        itable = (1 << nbits_) - 1;
      }
    }

    out << uppercase << setfill('0') << setw(width) << hex << itable << dec << endl;
  }

  out.close();
}

int TrackletLUT::lookup(unsigned int index) const {
  assert(index < table_.size());
  return table_[index];
}
