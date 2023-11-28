#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <mutex>

#include "L1Trigger/TrackFindingTracklet/interface/TrackletConfigBuilder.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#endif

using namespace std;
using namespace trklet;

TrackletConfigBuilder::TrackletConfigBuilder(const Settings& settings, const tt::Setup* setup) : settings_(settings) {
  NSector_ = N_SECTOR;
  rcrit_ = settings.rcrit();

  combinedmodules_ = settings.combined();

  extended_ = settings.extended();

  rinvmax_ = settings.rinvmax();

  rmaxdisk_ = settings.rmaxdisk();
  zlength_ = settings.zlength();

  for (int i = 0; i < N_LAYER; i++) {
    rmean_[i] = settings.rmean(i);
  }

  for (int i = 0; i < N_DISK; i++) {
    zmean_[i] = settings.zmean(i);
  }

  dphisectorHG_ = settings.dphisectorHG();

  for (int layerdisk = 0; layerdisk < N_LAYER + N_DISK; layerdisk++) {
    NRegions_[layerdisk] = settings.nallstubs(layerdisk);
    NVMME_[layerdisk] = settings.nvmme(layerdisk);
  }

  for (unsigned int iseed = 0; iseed < N_SEED_PROMPT; iseed++) {
    NVMTE_[iseed] = std::pair<unsigned int, unsigned int>(settings.nvmte(0, iseed), settings.nvmte(1, iseed));
    NTC_[iseed] = settings.NTC(iseed);
  }

  initGeom();

  buildTE();

  buildTC();

  buildProjections();

  setDTCphirange(setup);

  if (settings_.writeConfig()) {
    static std::once_flag runOnce;  // Only one thread should call this.
    std::call_once(runOnce, &TrackletConfigBuilder::writeDTCphirange, this);
  }
}

//--- Calculate phi range of modules read by each DTC.

#ifdef CMSSW_GIT_HASH

void TrackletConfigBuilder::setDTCphirange(const tt::Setup* setup) {
  list<DTCinfo> vecDTCinfo_unsorted;

  // Loop over DTCs in this tracker nonant.
  unsigned int numDTCsPerSector = setup->numDTCsPerRegion();
  for (unsigned int dtcId = 0; dtcId < numDTCsPerSector; dtcId++) {
    typedef std::pair<float, float> PhiRange;
    std::map<int, PhiRange> dtcPhiRange;

    // Loop over all tracker nonants, taking worst case not all identical.
    for (unsigned int iSector = 0; iSector < N_SECTOR; iSector++) {
      unsigned int dtcId_regI = iSector * numDTCsPerSector + dtcId;
      const std::vector<tt::SensorModule*>& dtcModules = setup->dtcModules(dtcId_regI);
      for (const tt::SensorModule* sm : dtcModules) {
        // Convert layer number to Hybrid convention.
        int layer = sm->layerId();  // Barrel = 1-6, Endcap = 11-15;
        if (sm->barrel()) {
          layer--;  // Barrel 0-5
        } else {
          const int endcapOffsetHybrid = 5;
          layer -= endcapOffsetHybrid;  // Layer 6-19
        }
        // Inner radius of module.
        float r = sm->r() - 0.5 * sm->numColumns() * sm->pitchCol() * fabs(sm->sinTilt());
        // phi with respect to tracker nonant centre.
        float phiMin = sm->phi() - 0.5 * sm->numRows() * sm->pitchRow() / r;
        float phiMax = sm->phi() + 0.5 * sm->numRows() * sm->pitchRow() / r;
        // Hybrid measures phi w.r.t. lower edge of tracker nonant.
        const float phiOffsetHybrid = 0.5 * dphisectorHG_;
        phiMin += phiOffsetHybrid;
        phiMax += phiOffsetHybrid;
        if (dtcPhiRange.find(layer) == dtcPhiRange.end()) {
          dtcPhiRange[layer] = {phiMin, phiMax};
        } else {
          dtcPhiRange.at(layer).first = std::min(phiMin, dtcPhiRange.at(layer).first);
          dtcPhiRange.at(layer).second = std::max(phiMax, dtcPhiRange.at(layer).second);
        }
      }
    }
    for (const auto& p : dtcPhiRange) {
      const unsigned int numSlots = setup->numATCASlots();
      std::string dtcName = settings_.slotToDTCname(dtcId % numSlots);
      if (dtcId >= numSlots)
        dtcName = "neg" + dtcName;
      DTCinfo info;
      info.name = dtcName;
      info.layer = p.first;
      info.phimin = p.second.first;
      info.phimax = p.second.second;
      vecDTCinfo_unsorted.push_back(info);
    }
  }

  // Put DTCinfo vector in traditional order (PS first). (Needed?)
  for (const DTCinfo& info : vecDTCinfo_unsorted) {
    string dtcname = info.name;
    if (dtcname.find("PS") != std::string::npos) {
      vecDTCinfo_.push_back(info);
    }
  }
  for (const DTCinfo& info : vecDTCinfo_unsorted) {
    string dtcname = info.name;
    if (dtcname.find("PS") == std::string::npos) {
      vecDTCinfo_.push_back(info);
    }
  }
}

//--- Write DTC phi ranges to file to support stand-alone emulation.
//--- (Only needed to support stand-alone emulation)

void TrackletConfigBuilder::writeDTCphirange() const {
  bool first = true;
  for (const DTCinfo& info : vecDTCinfo_) {
    string dirName = settings_.tablePath();
    string fileName = dirName + "../dtcphirange.dat";
    std::ofstream out;
    openfile(out, first, dirName, fileName, __FILE__, __LINE__);
    if (first) {
      out << "// layer & phi ranges of modules read by each DTC" << endl;
      out << "// (Used by stand-alone emulation)" << endl;
    }
    out << info.name << " " << info.layer << " " << info.phimin << " " << info.phimax << endl;
    out.close();
    first = false;
  }
}

#else

//--- Set DTC phi ranges from .txt file (stand-alone operation only)

void TrackletConfigBuilder::setDTCphirange(const tt::Setup* setup) {
  // This file previously written by writeDTCphirange().
  const string fname = "../data/dtcphirange.txt";
  if (vecDTCinfo_.empty()) {  // Only run once per thread.
    std::ifstream str_dtc;
    str_dtc.open(fname);
    assert(str_dtc.good());
    string line;
    while (ifstream, getline(line)) {
      std::istringstream iss(line);
      DTCinfo info;
      iss >> info.name >> info.layer >> info.phimin >> info.phimax;
      vecDTCinfo_.push_back(info);
    }
    str_dtc.close();
  }
}

#endif

//--- Helper fcn. to get the layers/disks for a seed

std::pair<unsigned int, unsigned int> TrackletConfigBuilder::seedLayers(unsigned int iSeed) {
  return std::pair<unsigned int, unsigned int>(settings_.seedlayers(0, iSeed), settings_.seedlayers(1, iSeed));
}

//--- Method to initialize the regions and VM in each layer

void TrackletConfigBuilder::initGeom() {
  for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
    double dphi = dphisectorHG_ / NRegions_[ilayer];
    for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
      std::vector<std::pair<unsigned int, unsigned int> > emptyVec;
      projections_[ilayer].push_back(emptyVec);
      // FIX: sector doesn't have hourglass shape
      double phimin = dphi * iReg;
      double phimax = phimin + dphi;
      std::pair<double, double> tmp(phimin, phimax);
      allStubs_[ilayer].push_back(tmp);
      double dphiVM = dphi / NVMME_[ilayer];
      for (unsigned int iVM = 0; iVM < NVMME_[ilayer]; iVM++) {
        double phivmmin = phimin + iVM * dphiVM;
        double phivmmax = phivmmin + dphiVM;
        std::pair<double, double> tmp(phivmmin, phivmmax);
        VMStubsME_[ilayer].push_back(tmp);
      }
    }
  }
  for (unsigned int iseed = 0; iseed < N_SEED_PROMPT; iseed++) {
    unsigned int l1 = seedLayers(iseed).first;
    unsigned int l2 = seedLayers(iseed).second;
    unsigned int nVM1 = NVMTE_[iseed].first;
    unsigned int nVM2 = NVMTE_[iseed].second;
    double dphiVM = dphisectorHG_ / (nVM1 * NRegions_[l1]);
    for (unsigned int iVM = 0; iVM < nVM1 * NRegions_[l1]; iVM++) {
      double phivmmin = iVM * dphiVM;
      double phivmmax = phivmmin + dphiVM;
      std::pair<double, double> tmp(phivmmin, phivmmax);
      VMStubsTE_[iseed].first.push_back(tmp);
    }
    dphiVM = dphisectorHG_ / (nVM2 * NRegions_[l2]);
    for (unsigned int iVM = 0; iVM < nVM2 * NRegions_[l2]; iVM++) {
      double phivmmin = iVM * dphiVM;
      double phivmmax = phivmmin + dphiVM;
      std::pair<double, double> tmp(phivmmin, phivmmax);
      VMStubsTE_[iseed].second.push_back(tmp);
    }
  }
}

//--- Helper fcn to get the radii of the two layers in a seed

std::pair<double, double> TrackletConfigBuilder::seedRadii(unsigned int iseed) {
  std::pair<unsigned int, unsigned int> seedlayers = seedLayers(iseed);

  unsigned int l1 = seedlayers.first;
  unsigned int l2 = seedlayers.second;

  double r1, r2;

  if (iseed < 4) {  //barrel seeding
    r1 = rmean_[l1];
    r2 = rmean_[l2];
  } else if (iseed < 6) {   //disk seeding
    r1 = rmean_[0] + 40.0;  //FIX: Somewhat of a hack - but allows finding all the regions
    //when projecting to L1
    r2 = r1 * zmean_[l2 - 6] / zmean_[l1 - 6];
  } else {  //overlap seeding
    r1 = rmean_[l1];
    r2 = r1 * zmean_[l2 - 6] / zlength_;
  }

  return std::pair<double, double>(r1, r2);
}

//--- Helper function to determine if a pair of VM memories form valid TE

bool TrackletConfigBuilder::validTEPair(unsigned int iseed, unsigned int iTE1, unsigned int iTE2) {
  double rinvmin = 999.9;
  double rinvmax = -999.9;

  double phi1[2] = {VMStubsTE_[iseed].first[iTE1].first, VMStubsTE_[iseed].first[iTE1].second};
  double phi2[2] = {VMStubsTE_[iseed].second[iTE2].first, VMStubsTE_[iseed].second[iTE2].second};

  std::pair<double, double> seedradii = seedRadii(iseed);

  for (unsigned int i1 = 0; i1 < 2; i1++) {
    for (unsigned int i2 = 0; i2 < 2; i2++) {
      double arinv = rinv(seedradii.first, phi1[i1], seedradii.second, phi2[i2]);
      if (arinv < rinvmin)
        rinvmin = arinv;
      if (arinv > rinvmax)
        rinvmax = arinv;
    }
  }

  if (rinvmin > rinvmax_)
    return false;
  if (rinvmax < -rinvmax_)
    return false;

  return true;
}

//--- Builds the list of TE for each seeding combination

void TrackletConfigBuilder::buildTE() {
  for (unsigned int iseed = 0; iseed < N_SEED_PROMPT; iseed++) {
    for (unsigned int i1 = 0; i1 < VMStubsTE_[iseed].first.size(); i1++) {
      for (unsigned int i2 = 0; i2 < VMStubsTE_[iseed].second.size(); i2++) {
        if (validTEPair(iseed, i1, i2)) {
          std::pair<unsigned int, unsigned int> tmp(i1, i2);
          // Contains pairs of indices of all valid VM pairs in seeding layers
          TE_[iseed].push_back(tmp);
        }
      }
    }
  }
}

//--- Builds the lists of TC for each seeding combination

void TrackletConfigBuilder::buildTC() {
  for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
    unsigned int nTC = NTC_[iSeed];
    std::vector<std::pair<unsigned int, unsigned int> >& TEs = TE_[iSeed];
    std::vector<std::vector<unsigned int> >& TCs = TC_[iSeed];

    //Very naive method to group TEs in TC

    double invnTC = nTC * (1.0 / TEs.size());

    for (unsigned int iTE = 0; iTE < TEs.size(); iTE++) {
      int iTC = invnTC * iTE;
      assert(iTC < (int)nTC);
      if (iTC >= (int)TCs.size()) {
        std::vector<unsigned int> tmp;
        tmp.push_back(iTE);
        TCs.push_back(tmp);
      } else {
        TCs[iTC].push_back(iTE);
      }
    }
  }
}

//--- Helper fcn to return the phi range of a projection of a tracklet from a TC

std::pair<double, double> TrackletConfigBuilder::seedPhiRange(double rproj, unsigned int iSeed, unsigned int iTC) {
  std::vector<std::vector<unsigned int> >& TCs = TC_[iSeed];

  std::pair<double, double> seedradii = seedRadii(iSeed);

  double phimin = 999.0;
  double phimax = -999.0;
  for (unsigned int iTE = 0; iTE < TCs[iTC].size(); iTE++) {
    unsigned int theTE = TCs[iTC][iTE];
    unsigned int l1TE = TE_[iSeed][theTE].first;
    unsigned int l2TE = TE_[iSeed][theTE].second;
    double phi1[2] = {VMStubsTE_[iSeed].first[l1TE].first, VMStubsTE_[iSeed].first[l1TE].second};
    double phi2[2] = {VMStubsTE_[iSeed].second[l2TE].first, VMStubsTE_[iSeed].second[l2TE].second};
    for (unsigned int i1 = 0; i1 < 2; i1++) {
      for (unsigned int i2 = 0; i2 < 2; i2++) {
        double aphi = phi(seedradii.first, phi1[i1], seedradii.second, phi2[i2], rproj);
        if (aphi < phimin)
          phimin = aphi;
        if (aphi > phimax)
          phimax = aphi;
      }
    }
  }
  return std::pair<double, double>(phimin, phimax);
}

//--- Finds the projections needed for each seeding combination

void TrackletConfigBuilder::buildProjections() {
  set<string> emptyProjStandard = {
      "TPROJ_L1L2H_L3PHIB", "TPROJ_L1L2E_L3PHIC", "TPROJ_L1L2K_L3PHIC", "TPROJ_L1L2H_L3PHID", "TPROJ_L1L2F_L5PHIA",
      "TPROJ_L1L2G_L5PHID", "TPROJ_L1L2A_L6PHIA", "TPROJ_L1L2J_L6PHIB", "TPROJ_L1L2C_L6PHIC", "TPROJ_L1L2L_L6PHID",
      "TPROJ_L3L4D_D1PHIB", "TPROJ_L2L3A_D1PHIC", "TPROJ_L3L4A_D1PHIC", "TPROJ_L1L2G_D2PHIA", "TPROJ_L1D1D_D2PHIA",
      "TPROJ_L1D1E_D2PHIA", "TPROJ_L1L2J_D2PHIB", "TPROJ_L3L4D_D2PHIB", "TPROJ_L1D1A_D2PHIB", "TPROJ_L1D1F_D2PHIB",
      "TPROJ_L1D1G_D2PHIB", "TPROJ_L1L2C_D2PHIC", "TPROJ_L2L3A_D2PHIC", "TPROJ_L3L4A_D2PHIC", "TPROJ_L1D1B_D2PHIC",
      "TPROJ_L1D1C_D2PHIC", "TPROJ_L1D1H_D2PHIC", "TPROJ_L2D1A_D2PHIC", "TPROJ_L1L2F_D2PHID", "TPROJ_L1D1D_D2PHID",
      "TPROJ_L1D1E_D2PHID", "TPROJ_L1L2G_D3PHIA", "TPROJ_L1D1D_D3PHIA", "TPROJ_L1D1E_D3PHIA", "TPROJ_L1L2J_D3PHIB",
      "TPROJ_L1D1A_D3PHIB", "TPROJ_L1D1F_D3PHIB", "TPROJ_L1D1G_D3PHIB", "TPROJ_L1L2C_D3PHIC", "TPROJ_L2L3A_D3PHIC",
      "TPROJ_L1D1B_D3PHIC", "TPROJ_L1D1C_D3PHIC", "TPROJ_L1D1H_D3PHIC", "TPROJ_L2D1A_D3PHIC", "TPROJ_L1L2F_D3PHID",
      "TPROJ_L1D1D_D3PHID", "TPROJ_L1D1E_D3PHID", "TPROJ_L1L2G_D4PHIA", "TPROJ_L1D1D_D4PHIA", "TPROJ_L1D1E_D4PHIA",
      "TPROJ_L1L2J_D4PHIB", "TPROJ_L1D1G_D4PHIB", "TPROJ_L1L2C_D4PHIC", "TPROJ_L2L3A_D4PHIC", "TPROJ_L1D1B_D4PHIC",
      "TPROJ_L2D1A_D4PHIC", "TPROJ_L1L2F_D4PHID", "TPROJ_L1D1D_D4PHID", "TPROJ_L1D1E_D5PHIA", "TPROJ_L1D1G_D5PHIB",
      "TPROJ_L1D1B_D5PHIC", "TPROJ_L1D1D_D5PHID"};

  set<string> emptyProjCombined = {
      "TPROJ_L1L2J_L6PHIB", "TPROJ_L1L2C_L6PHIC", "TPROJ_L1L2G_D1PHIA", "TPROJ_L1L2J_D1PHIB", "TPROJ_L2L3D_D1PHIB",
      "TPROJ_L3L4D_D1PHIB", "TPROJ_L1L2C_D1PHIC", "TPROJ_L2L3A_D1PHIC", "TPROJ_L3L4A_D1PHIC", "TPROJ_L1L2F_D1PHID",
      "TPROJ_L1L2G_D2PHIA", "TPROJ_L1D1E_D2PHIA", "TPROJ_L1L2J_D2PHIB", "TPROJ_L2L3D_D2PHIB", "TPROJ_L3L4D_D2PHIB",
      "TPROJ_L1D1G_D2PHIB", "TPROJ_L1L2C_D2PHIC", "TPROJ_L2L3A_D2PHIC", "TPROJ_L3L4A_D2PHIC", "TPROJ_L1D1B_D2PHIC",
      "TPROJ_L2D1A_D2PHIC", "TPROJ_L1L2F_D2PHID", "TPROJ_L1D1D_D2PHID", "TPROJ_L1L2G_D3PHIA", "TPROJ_L1D1E_D3PHIA",
      "TPROJ_L1L2J_D3PHIB", "TPROJ_L2L3D_D3PHIB", "TPROJ_L1D1G_D3PHIB", "TPROJ_L1L2C_D3PHIC", "TPROJ_L2L3A_D3PHIC",
      "TPROJ_L1D1B_D3PHIC", "TPROJ_L2D1A_D3PHIC", "TPROJ_L1L2F_D3PHID", "TPROJ_L1D1D_D3PHID", "TPROJ_L1L2G_D4PHIA",
      "TPROJ_L1D1E_D4PHIA", "TPROJ_L1L2J_D4PHIB", "TPROJ_L2L3D_D4PHIB", "TPROJ_L1D1G_D4PHIB", "TPROJ_L1L2C_D4PHIC",
      "TPROJ_L2L3A_D4PHIC", "TPROJ_L1D1B_D4PHIC", "TPROJ_L2D1A_D4PHIC", "TPROJ_L1L2F_D4PHID", "TPROJ_L1D1D_D4PHID",
      "TPROJ_L1D1E_D5PHIA", "TPROJ_L1D1G_D5PHIB", "TPROJ_L1D1B_D5PHIC", "TPROJ_L1D1D_D5PHID"};

  for (unsigned int iseed = 0; iseed < N_SEED_PROMPT; iseed++) {
    std::vector<std::vector<unsigned int> >& TCs = TC_[iseed];

    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      if (matchport_[iseed][ilayer] == -1)
        continue;
      for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
        for (unsigned int iTC = 0; iTC < TCs.size(); iTC++) {
          double rproj = rmaxdisk_;
          if (ilayer < 6)
            rproj = rmean_[ilayer];
          std::pair<double, double> phiRange = seedPhiRange(rproj, iseed, iTC);
          if (phiRange.first < allStubs_[ilayer][iReg].second && phiRange.second > allStubs_[ilayer][iReg].first) {
            std::pair<unsigned int, unsigned int> tmp(iseed, iTC);  //seedindex and TC
            string projName = TPROJName(iseed, iTC, ilayer, iReg);
            if (combinedmodules_) {
              if (emptyProjCombined.find(projName) == emptyProjCombined.end()) {
                projections_[ilayer][iReg].push_back(tmp);
              }
            } else {
              if (emptyProjStandard.find(projName) == emptyProjStandard.end()) {
                projections_[ilayer][iReg].push_back(tmp);
              }
            }
          }
        }
      }
    }
  }
}

//--- Helper function to calculate the phi position of a seed at radius r that is formed
//--- by two stubs at (r1,phi1) and (r2, phi2)

double TrackletConfigBuilder::phi(double r1, double phi1, double r2, double phi2, double r) {
  double rhoinv = rinv(r1, phi1, r2, phi2);
  if (std::abs(rhoinv) > rinvmax_) {
    rhoinv = rinvmax_ * rhoinv / std::abs(rhoinv);
  }
  return phi1 + asin(0.5 * r * rhoinv) - asin(0.5 * r1 * rhoinv);
}

//--- Helper function to calculate rinv for two stubs at (r1,phi1) and (r2,phi2)

double TrackletConfigBuilder::rinv(double r1, double phi1, double r2, double phi2) {
  double deltaphi = phi1 - phi2;
  return 2 * sin(deltaphi) / sqrt(r2 * r2 + r1 * r1 - 2 * r1 * r2 * cos(deltaphi));
}

std::string TrackletConfigBuilder::iSeedStr(unsigned int iSeed) const {
  static std::string name[8] = {"L1L2", "L2L3", "L3L4", "L5L6", "D1D2", "D3D4", "L1D1", "L2D1"};

  assert(iSeed < 8);
  return name[iSeed];
}

std::string TrackletConfigBuilder::numStr(unsigned int i) {
  static std::string num[32] = {"1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10", "11",
                                "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22",
                                "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"};
  assert(i < 32);
  return num[i];
}

std::string TrackletConfigBuilder::iTCStr(unsigned int iTC) const {
  static std::string name[12] = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"};

  assert(iTC < 12);
  return name[iTC];
}

std::string TrackletConfigBuilder::iRegStr(unsigned int iReg, unsigned int iSeed) const {
  static std::string name[8] = {"A", "B", "C", "D", "E", "F", "G", "H"};

  static std::string nameOverlap[8] = {"X", "Y", "Z", "W", "Q", "R", "S", "T"};

  static std::string nameL2L3[4] = {"I", "J", "K", "L"};

  if (iSeed == Seed::L2L3) {
    assert(iReg < 4);
    return nameL2L3[iReg];
  }
  if (iSeed == Seed::L1D1 || iSeed == Seed::L2D1) {
    assert(iReg < 8);
    return nameOverlap[iReg];
  }
  assert(iReg < 8);
  return name[iReg];
}

std::string TrackletConfigBuilder::TCName(unsigned int iSeed, unsigned int iTC) const {
  if (combinedmodules_) {
    return "TP_" + iSeedStr(iSeed) + iTCStr(iTC);
  } else {
    return "TC_" + iSeedStr(iSeed) + iTCStr(iTC);
  }
}

std::string TrackletConfigBuilder::LayerName(unsigned int ilayer) {
  return ilayer < 6 ? ("L" + numStr(ilayer)) : ("D" + numStr(ilayer - 6));
}

std::string TrackletConfigBuilder::TPROJName(unsigned int iSeed,
                                             unsigned int iTC,
                                             unsigned int ilayer,
                                             unsigned int ireg) const {
  return "TPROJ_" + iSeedStr(iSeed) + iTCStr(iTC) + "_" + LayerName(ilayer) + "PHI" + iTCStr(ireg);
}

std::string TrackletConfigBuilder::PRName(unsigned int ilayer, unsigned int ireg) const {
  if (combinedmodules_) {
    return "MP_" + LayerName(ilayer) + "PHI" + iTCStr(ireg);
  } else {
    return "PR_" + LayerName(ilayer) + "PHI" + iTCStr(ireg);
  }
}

void TrackletConfigBuilder::writeProjectionMemories(std::ostream& os, std::ostream& memories, std::ostream&) {
  // Each TC (e.g. TC_L1L2D) writes a projection memory (TPROJ) for each layer the seed projects to,
  // with name indicating the TC and which layer & phi region it projects to (e.g. TPROJ_L1L2D_L3PHIA).
  //
  // Each PR (e.g. PR_L3PHIA) reads all TPROJ memories for the given layer & phi region.

  for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
    for (unsigned int ireg = 0; ireg < projections_[ilayer].size(); ireg++) {
      for (unsigned int imem = 0; imem < projections_[ilayer][ireg].size(); imem++) {
        unsigned int iSeed = projections_[ilayer][ireg][imem].first;
        unsigned int iTC = projections_[ilayer][ireg][imem].second;

        memories << "TrackletProjections: " + TPROJName(iSeed, iTC, ilayer, ireg) + " [54]" << std::endl;

        os << TPROJName(iSeed, iTC, ilayer, ireg) << " input=> " << TCName(iSeed, iTC) << ".projout"
           << LayerName(ilayer) << "PHI" << iTCStr(ireg) << " output=> " << PRName(ilayer, ireg) << ".projin"
           << std::endl;
      }
    }
  }
}

std::string TrackletConfigBuilder::SPName(unsigned int l1,
                                          unsigned int ireg1,
                                          unsigned int ivm1,
                                          unsigned int l2,
                                          unsigned int ireg2,
                                          unsigned int ivm2,
                                          unsigned int iseed) const {
  return "SP_" + LayerName(l1) + "PHI" + iRegStr(ireg1, iseed) + numStr(ivm1) + "_" + LayerName(l2) + "PHI" +
         iRegStr(ireg2, iseed) + numStr(ivm2);
}

std::string TrackletConfigBuilder::SPDName(unsigned int l1,
                                           unsigned int ireg1,
                                           unsigned int ivm1,
                                           unsigned int l2,
                                           unsigned int ireg2,
                                           unsigned int ivm2,
                                           unsigned int l3,
                                           unsigned int ireg3,
                                           unsigned int ivm3,
                                           unsigned int iseed) const {
  return "SPD_" + LayerName(l1) + "PHI" + iRegStr(ireg1, iseed) + numStr(ivm1) + "_" + LayerName(l2) + "PHI" +
         iRegStr(ireg2, iseed) + numStr(ivm2) + "_" + LayerName(l3) + "PHI" + iRegStr(ireg3, iseed) + numStr(ivm3);
}

std::string TrackletConfigBuilder::TEName(unsigned int l1,
                                          unsigned int ireg1,
                                          unsigned int ivm1,
                                          unsigned int l2,
                                          unsigned int ireg2,
                                          unsigned int ivm2,
                                          unsigned int iseed) const {
  return "TE_" + LayerName(l1) + "PHI" + iRegStr(ireg1, iseed) + numStr(ivm1) + "_" + LayerName(l2) + "PHI" +
         iRegStr(ireg2, iseed) + numStr(ivm2);
}

std::string TrackletConfigBuilder::TEDName(unsigned int l1,
                                           unsigned int ireg1,
                                           unsigned int ivm1,
                                           unsigned int l2,
                                           unsigned int ireg2,
                                           unsigned int ivm2,
                                           unsigned int iseed) const {
  return "TED_" + LayerName(l1) + "PHI" + iRegStr(ireg1, iseed) + numStr(ivm1) + "_" + LayerName(l2) + "PHI" +
         iRegStr(ireg2, iseed) + numStr(ivm2);
}

std::string TrackletConfigBuilder::TParName(unsigned int l1, unsigned int l2, unsigned int l3, unsigned int itc) const {
  return "TPAR_" + LayerName(l1) + LayerName(l2) + LayerName(l3) + iTCStr(itc);
}

std::string TrackletConfigBuilder::TCDName(unsigned int l1, unsigned int l2, unsigned int l3, unsigned int itc) const {
  return "TCD_" + LayerName(l1) + LayerName(l2) + LayerName(l3) + iTCStr(itc);
}

std::string TrackletConfigBuilder::TPROJName(unsigned int l1,
                                             unsigned int l2,
                                             unsigned int l3,
                                             unsigned int itc,
                                             unsigned int projlayer,
                                             unsigned int projreg) const {
  return "TPROJ_" + LayerName(l1) + LayerName(l2) + LayerName(l3) + iTCStr(itc) + "_" + LayerName(projlayer) + "PHI" +
         iTCStr(projreg);
}

std::string TrackletConfigBuilder::FTName(unsigned int l1, unsigned int l2, unsigned int l3) const {
  return "FT_" + LayerName(l1) + LayerName(l2) + LayerName(l3);
}

std::string TrackletConfigBuilder::TREName(unsigned int l1,
                                           unsigned int ireg1,
                                           unsigned int l2,
                                           unsigned int ireg2,
                                           unsigned int iseed,
                                           unsigned int count) const {
  return "TRE_" + LayerName(l1) + iRegStr(ireg1, iseed) + LayerName(l2) + iRegStr(ireg2, iseed) + "_" + numStr(count);
}

std::string TrackletConfigBuilder::STName(unsigned int l1,
                                          unsigned int ireg1,
                                          unsigned int l2,
                                          unsigned int ireg2,
                                          unsigned int l3,
                                          unsigned int ireg3,
                                          unsigned int iseed,
                                          unsigned int count) const {
  return "ST_" + LayerName(l1) + iRegStr(ireg1, iseed) + LayerName(l2) + iRegStr(ireg2, iseed) + "_" + LayerName(l3) +
         iRegStr(ireg3, iseed) + "_" + numStr(count);
}

void TrackletConfigBuilder::writeSPMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  // Each TE reads one VM in two seed layers, finds stub pairs & writes to a StubPair ("SP") memory.
  //
  // Each TC reads several StubPair (SP) memories, each containing a pair of VMs of two seeding layers.
  // Several TC are created for each layer pair, and the SP distributed between them.
  // If TC name is TC_L1L2C, "C" indicates this is the 3rd TC in L1L2.

  if (combinedmodules_)
    return;

  for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
    for (unsigned int iTC = 0; iTC < TC_[iSeed].size(); iTC++) {
      for (unsigned int iTE = 0; iTE < TC_[iSeed][iTC].size(); iTE++) {
        unsigned int theTE = TC_[iSeed][iTC][iTE];

        unsigned int TE1 = TE_[iSeed][theTE].first;
        unsigned int TE2 = TE_[iSeed][theTE].second;

        unsigned int l1 = seedLayers(iSeed).first;
        unsigned int l2 = seedLayers(iSeed).second;

        memories << "StubPairs: "
                 << SPName(l1, TE1 / NVMTE_[iSeed].first, TE1, l2, TE2 / NVMTE_[iSeed].second, TE2, iSeed) << " [12]"
                 << std::endl;
        modules << "TrackletEngine: "
                << TEName(l1, TE1 / NVMTE_[iSeed].first, TE1, l2, TE2 / NVMTE_[iSeed].second, TE2, iSeed) << std::endl;

        os << SPName(l1, TE1 / NVMTE_[iSeed].first, TE1, l2, TE2 / NVMTE_[iSeed].second, TE2, iSeed) << " input=> "
           << TEName(l1, TE1 / NVMTE_[iSeed].first, TE1, l2, TE2 / NVMTE_[iSeed].second, TE2, iSeed)
           << ".stubpairout output=> " << TCName(iSeed, iTC) << ".stubpairin" << std::endl;
      }
    }
  }
}

void TrackletConfigBuilder::writeSPDMemories(std::ostream& wires, std::ostream& memories, std::ostream& modules) {
  // Similar to writeSPMemories, but for displaced (=extended) tracking,
  // with seeds based on triplets of layers.

  if (!extended_)
    return;

  vector<string> stubTriplets[N_SEED];

  for (unsigned int iSeed = N_SEED_PROMPT; iSeed < N_SEED; iSeed++) {
    int layerdisk1 = settings_.seedlayers(0, iSeed);
    int layerdisk2 = settings_.seedlayers(1, iSeed);
    int layerdisk3 = settings_.seedlayers(2, iSeed);

    unsigned int nallstub1 = settings_.nallstubs(layerdisk1);
    unsigned int nallstub2 = settings_.nallstubs(layerdisk2);
    unsigned int nallstub3 = settings_.nallstubs(layerdisk3);

    unsigned int nvm1 = settings_.nvmte(0, iSeed);
    unsigned int nvm2 = settings_.nvmte(1, iSeed);
    unsigned int nvm3 = settings_.nvmte(2, iSeed);

    int count = 0;
    for (unsigned int ireg1 = 0; ireg1 < nallstub1; ireg1++) {
      for (unsigned int ireg2 = 0; ireg2 < nallstub2; ireg2++) {
        for (unsigned int ireg3 = 0; ireg3 < nallstub3; ireg3++) {
          count++;
          memories << "StubTriplets: " << STName(layerdisk1, ireg1, layerdisk2, ireg2, layerdisk3, ireg3, iSeed, count)
                   << " [18]" << std::endl;
          stubTriplets[iSeed].push_back(STName(layerdisk1, ireg1, layerdisk2, ireg2, layerdisk3, ireg3, iSeed, count));
        }
      }
    }

    for (unsigned int ireg1 = 0; ireg1 < nallstub1; ireg1++) {
      for (unsigned int ivm1 = 0; ivm1 < nvm1; ivm1++) {
        for (unsigned int ireg2 = 0; ireg2 < nallstub2; ireg2++) {
          for (unsigned int ivm2 = 0; ivm2 < nvm2; ivm2++) {
            int count = 0;

            modules << "TrackletEngineDisplaced: "
                    << TEDName(layerdisk1, ireg1, ireg1 * nvm1 + ivm1, layerdisk2, ireg2, ireg2 * nvm2 + ivm2, iSeed)
                    << std::endl;

            for (unsigned int ireg3 = 0; ireg3 < nallstub3; ireg3++) {
              for (unsigned int ivm3 = 0; ivm3 < nvm3; ivm3++) {
                count++;

                memories << "StubPairsDisplaced: "
                         << SPDName(layerdisk1,
                                    ireg1,
                                    ireg1 * nvm1 + ivm1,
                                    layerdisk2,
                                    ireg2,
                                    ireg2 * nvm2 + ivm2,
                                    layerdisk3,
                                    ireg3,
                                    ireg3 * nvm3 + ivm3,
                                    iSeed)
                         << " [12]" << std::endl;

                modules << "TripletEngine: " << TREName(layerdisk1, ireg1, layerdisk2, ireg2, iSeed, count)
                        << std::endl;

                wires << SPDName(layerdisk1,
                                 ireg1,
                                 ireg1 * nvm1 + ivm1,
                                 layerdisk2,
                                 ireg2,
                                 ireg2 * nvm2 + ivm2,
                                 layerdisk3,
                                 ireg3,
                                 ireg3 * nvm3 + ivm3,
                                 iSeed)
                      << " input=> "
                      << TEDName(layerdisk1, ireg1, ireg1 * nvm1 + ivm1, layerdisk2, ireg2, ireg2 * nvm2 + ivm2, iSeed)
                      << ".stubpairout output=> " << TREName(layerdisk1, ireg1, layerdisk2, ireg2, iSeed, count)
                      << ".stubpair"
                      << "1"
                      << "in" << std::endl;
              }
            }
          }
        }
      }
    }

    unsigned int nTC = 10;
    for (unsigned int itc = 0; itc < nTC; itc++) {
      for (int iproj = 0; iproj < 4; iproj++) {
        int ilay = settings_.projlayers(iSeed, iproj);
        if (ilay > 0) {
          unsigned int nallstub = settings_.nallstubs(ilay - 1);
          for (unsigned int ireg = 0; ireg < nallstub; ireg++) {
            memories << "TrackletProjections: " << TPROJName(layerdisk1, layerdisk2, layerdisk3, itc, ilay - 1, ireg)
                     << " [54]" << std::endl;
          }
        }

        int idisk = settings_.projdisks(iSeed, iproj);
        if (idisk > 0) {
          unsigned int nallstub = settings_.nallstubs(idisk + 5);
          for (unsigned int ireg = 0; ireg < nallstub; ireg++) {
            memories << "TrackletProjections: " << TPROJName(layerdisk1, layerdisk2, layerdisk3, itc, idisk + 5, ireg)
                     << " [54]" << std::endl;

            wires << TPROJName(layerdisk1, layerdisk2, layerdisk3, itc, idisk + 5, ireg) << " input=> "
                  << TCDName(layerdisk1, layerdisk2, layerdisk3, itc) << ".projout" << LayerName(idisk + 1) << "PHI"
                  << iTCStr(ireg) << " output=> "
                  << "PR_" << LayerName(idisk + 1) << "PHI" << iTCStr(ireg) << ".projin" << std::endl;
          }
        }
      }

      memories << "TrackletParameters: " << TParName(layerdisk1, layerdisk2, layerdisk3, itc) << " [56]" << std::endl;

      modules << "TrackletCalculatorDisplaced: " << TCDName(layerdisk1, layerdisk2, layerdisk3, itc) << std::endl;
    }

    unsigned int nST = stubTriplets[iSeed].size();
    for (unsigned int iST = 0; iST < nST; iST++) {
      unsigned int iTC = (iST * nTC) / nST;
      assert(iTC < nTC);
      string stname = stubTriplets[iSeed][iST];
      string trename = "TRE_" + stname.substr(3, 6) + "_";
      unsigned int stlen = stname.size();
      if (stname[stlen - 2] == '_')
        trename += stname.substr(stlen - 1, 1);
      if (stname[stlen - 3] == '_')
        trename += stname.substr(stlen - 2, 2);
      wires << stname << " input=> " << trename << ".stubtripout output=> "
            << TCDName(layerdisk1, layerdisk2, layerdisk3, iTC) << ".stubtriplet" << ((iST * nTC) % nST) << "in"
            << std::endl;
    }

    modules << "FitTrack: " << FTName(layerdisk1, layerdisk2, layerdisk3) << std::endl;
  }
}

void TrackletConfigBuilder::writeAPMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  // The AllProjection memories (e.g. AP_L2PHIA) contain the intercept point of the projection to
  // a layer. Each is written by one PR module of similar name (e.g. PR_L2PHIA), and read by
  // a MC (e.g. MC_L2PHIA).

  if (combinedmodules_)
    return;

  for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
    for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
      memories << "AllProj: AP_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << " [56]" << std::endl;
      modules << "ProjectionRouter: PR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << std::endl;

      os << "AP_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << " input=> PR_" << LayerName(ilayer) << "PHI"
         << iTCStr(iReg) << ".allprojout output=> MC_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".allprojin"
         << std::endl;
    }
  }
}

void TrackletConfigBuilder::writeCMMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  // The CandidateMatch memory (e.g. CM_L1PHIA1) are each written by ME module of similar name
  // (e.g. ME_L1PHIA1) and contain indices of matching (tracklet projections,stubs) in the specified
  // VM region.
  // All CM memories in a given phi region (e.g. L1PHIA) are read by a MC module (e.g. MC_L1PHIA) that
  // does more precise matching.

  if (combinedmodules_)
    return;

  for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
    for (unsigned int iME = 0; iME < NVMME_[ilayer] * NRegions_[ilayer]; iME++) {
      memories << "CandidateMatch: CM_" << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << iME + 1
               << " [12]" << std::endl;
      modules << "MatchEngine: ME_" << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << iME + 1
              << std::endl;

      os << "CM_" << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << iME + 1 << " input=> ME_"
         << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << iME + 1 << ".matchout output=> MC_"
         << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << ".matchin" << std::endl;
    }
  }
}

void TrackletConfigBuilder::writeVMPROJMemories(std::ostream& os, std::ostream& memories, std::ostream&) {
  // The VMPROJ memories (e.g. VMPROJ_L2PHIA1) written by a PR module each correspond to projections to
  // a single VM region in a layer. Each is filled by the PR using all projections (TPROJ) to this VM
  // from different seeding layers.
  //
  // Each VMPROJ memory is read by a ME module, which matches the projection to stubs.

  if (combinedmodules_)
    return;

  for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
    for (unsigned int iME = 0; iME < NVMME_[ilayer] * NRegions_[ilayer]; iME++) {
      memories << "VMProjections: VMPROJ_" << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << iME + 1
               << " [13]" << std::endl;

      os << "VMPROJ_" << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << iME + 1 << " input=> PR_"
         << LayerName(ilayer) << "PHI" << iTCStr(iME / NVMME_[ilayer]) << ".vmprojout"
         << "PHI" << iTCStr(iME / NVMME_[ilayer]) << iME + 1 << " output=> ME_" << LayerName(ilayer) << "PHI"
         << iTCStr(iME / NVMME_[ilayer]) << iME + 1 << ".vmprojin" << std::endl;
    }
  }
}

void TrackletConfigBuilder::writeFMMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  // All FullMatch (e.g. FM_L2L3_L1PHIA) memories corresponding to a matches between stubs & tracklets
  // in a given region (e.g. L1PHIA) from all seeding layers, are written by a MC module (e.g. MC_L1PHIA).
  //
  // All FullMatch memories corresponding to a given seed pair are read by the TrackBuilder (e.g. FT_L1L2),
  // which checks if the track has stubs in enough layers.

  if (combinedmodules_) {
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
        modules << "MatchProcessor: MP_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << std::endl;
        for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
          if (matchport_[iSeed][ilayer] == -1)
            continue;
          memories << "FullMatch: FM_" << iSeedStr(iSeed) << "_" << LayerName(ilayer) << "PHI" << iTCStr(iReg)
                   << " [36]" << std::endl;
          os << "FM_" << iSeedStr(iSeed) << "_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << " input=> MP_"
             << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".matchout1 output=> FT_" << iSeedStr(iSeed)
             << ".fullmatch" << matchport_[iSeed][ilayer] << "in" << iReg + 1 << std::endl;
        }
      }
    }
  } else {
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
        modules << "MatchCalculator: MC_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << std::endl;
        for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
          if (matchport_[iSeed][ilayer] == -1)
            continue;
          memories << "FullMatch: FM_" << iSeedStr(iSeed) << "_" << LayerName(ilayer) << "PHI" << iTCStr(iReg)
                   << " [36]" << std::endl;
          os << "FM_" << iSeedStr(iSeed) << "_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << " input=> MC_"
             << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".matchout1 output=> FT_" << iSeedStr(iSeed)
             << ".fullmatch" << matchport_[iSeed][ilayer] << "in" << iReg + 1 << std::endl;
        }
      }
    }
  }
}

void TrackletConfigBuilder::writeASMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  // Each VMR writes AllStub memories (AS) for a single phi region (e.g. PHIC),
  // merging data from all DTCs related to this phi region. It does so by merging data from
  // the IL memories written by all IRs for this phi region. The wiring map lists all
  // IL memories that feed (">") into a single VMR ("VMR_L1PHIC") that writes to the
  // an AS memory ("AS_L1PHIC").
  // Multiple copies of each AS memory exist where several modules in chain want to read it.

  if (combinedmodules_) {
    //First write AS memories used by MatchProcessor
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
        memories << "AllStubs: AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n1"
                 << " [42]" << std::endl;
        if (combinedmodules_) {
          modules << "VMRouterCM: VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << std::endl;
        } else {
          modules << "VMRouter: VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << std::endl;
        }
        os << "AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n1"
           << " input=> VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".allstubout output=> MP_"
           << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".allstubin" << std::endl;
      }
    }

    //Next write AS memories used by TrackletProcessor
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (int iReg = 0; iReg < (int)NRegions_[ilayer]; iReg++) {
        for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
          unsigned int l1 = seedLayers(iSeed).first;
          unsigned int l2 = seedLayers(iSeed).second;

          if (ilayer != l1 && ilayer != l2)
            continue;

          bool inner = ilayer == l1;

          for (unsigned int iTC = 0; iTC < TC_[iSeed].size(); iTC++) {
            int nTCReg = TC_[iSeed].size() / NRegions_[l2];

            int iTCReg = iTC / nTCReg;

            int jTCReg = iTC % nTCReg;

            if (ilayer == l2) {
              if (iTCReg != iReg)
                continue;
            }

            string ext = "";

            if (ilayer == l1) {
              int ratio = NRegions_[l1] / NRegions_[l2];
              int min = iTCReg * ratio - 1 + jTCReg;
              int max = (iTCReg + 1) * ratio - (nTCReg - jTCReg - 1);
              if ((int)iReg < min || (int)iReg > max)
                continue;

              if (max - min >= 2) {
                ext = "M";
                if (iReg == min)
                  ext = "R";
                if (iReg == max)
                  ext = "L";
              }

              if (max - min == 1) {
                if (nTCReg == 2) {
                  assert(0);
                  if (jTCReg == 0) {
                    if (iReg == min)
                      ext = "R";
                    if (iReg == max)
                      ext = "B";
                  }
                  if (jTCReg == 1) {
                    if (iReg == min)
                      ext = "A";
                    if (iReg == max)
                      ext = "L";
                  }
                }
                if (nTCReg == 3) {
                  if (jTCReg == 0) {
                    if (iReg == min)
                      ext = "A";
                    if (iReg == max)
                      ext = "F";
                  }
                  if (jTCReg == 1) {
                    if (iReg == min)
                      ext = "E";
                    if (iReg == max)
                      ext = "D";
                  }
                  if (jTCReg == 2) {
                    if (iReg == min)
                      ext = "C";
                    if (iReg == max)
                      ext = "B";
                  }
                }
              }
              assert(!ext.empty());
            }

            if (ext.empty()) {
              ext = "_" + LayerName(l1) + iTCStr(iTC);
            }

            if (iSeed < 4) {  //Barrel seeding
              ext = "_B" + ext;
            } else if (iSeed > 5) {
              ext = "_O" + ext;
            } else {
              ext = "_D" + ext;
            }

            if (inner) {
              memories << "AllInnerStubs: ";
            } else {
              memories << "AllStubs: ";
            }
            memories << "AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ext << " [42]" << std::endl;
            os << "AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ext << " input=> VMR_" << LayerName(ilayer)
               << "PHI" << iTCStr(iReg) << ".all" << (inner ? "inner" : "") << "stubout output=> TP_" << iSeedStr(iSeed)
               << iTCStr(iTC);
            if (inner) {
              os << ".innerallstubin" << std::endl;
            } else {
              os << ".outerallstubin" << std::endl;
            }
          }
        }
      }
    }

  } else {
    //First write AS memories used by MatchCalculator
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
        memories << "AllStubs: AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n1"
                 << " [42]" << std::endl;
        if (combinedmodules_) {
          modules << "VMRouterCM: VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << std::endl;
        } else {
          modules << "VMRouter: VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << std::endl;
        }
        os << "AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n1"
           << " input=> VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".allstubout output=> MC_"
           << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".allstubin" << std::endl;
      }
    }

    //Next write AS memories used by TrackletCalculator
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
        unsigned int nmem = 1;

        for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
          unsigned int l1 = seedLayers(iSeed).first;
          unsigned int l2 = seedLayers(iSeed).second;

          if (ilayer != l1 && ilayer != l2)
            continue;

          for (unsigned int iTC = 0; iTC < TC_[iSeed].size(); iTC++) {
            bool used = false;
            // Each TC processes data from several TEs.
            for (unsigned int iTE = 0; iTE < TC_[iSeed][iTC].size(); iTE++) {
              unsigned int theTE = TC_[iSeed][iTC][iTE];

              unsigned int TE1 = TE_[iSeed][theTE].first;  // VM in inner/outer layer of this TE.
              unsigned int TE2 = TE_[iSeed][theTE].second;

              if (l1 == ilayer && iReg == TE1 / NVMTE_[iSeed].first)
                used = true;
              if (l2 == ilayer && iReg == TE2 / NVMTE_[iSeed].second)
                used = true;
            }

            if (used) {
              nmem++;  // Another copy of memory
              memories << "AllStubs: AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n" << nmem << " [42]"
                       << std::endl;
              os << "AS_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n" << nmem << " input=> VMR_"
                 << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".allstubout output=> TC_" << iSeedStr(iSeed)
                 << iTCStr(iTC);
              if (ilayer == l1) {
                os << ".innerallstubin" << std::endl;
              } else {
                os << ".outerallstubin" << std::endl;
              }
            }
          }
        }
      }
    }
  }
}

void TrackletConfigBuilder::writeVMSMemories(std::ostream& os, std::ostream& memories, std::ostream&) {
  // Each VMR writes to Virtual Module memories ("VMS") to be used later by the ME or TE etc.
  // Memory VMSTE_L1PHIC9-12 is the memory for small phi region C in L1 for the TE module.
  // Numbers 9-12 correspond to the 4 VMs in this phi region.
  //
  // Each TE reads one VMS memory in each seeding layer.

  if (combinedmodules_) {
    //First write VMS memories used by MatchProcessor
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (unsigned int iReg = 0; iReg < NRegions_[ilayer]; iReg++) {
        memories << "VMStubsME: VMSME_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n1 [18]" << std::endl;
        os << "VMSME_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << "n1"
           << " input=> VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".vmstuboutPHI" << iTCStr(iReg)
           << " output=> MP_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".vmstubin" << std::endl;
      }
    }

    //Next write VMS memories used by TrackletProcessor
    for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
      //FIXME - code could be cleaner
      unsigned int l1 = seedLayers(iSeed).first;
      unsigned int l2 = seedLayers(iSeed).second;

      unsigned int ilayer = seedLayers(iSeed).second;

      //for(unsigned int iReg=0;iReg<NRegions_[ilayer];iReg++){

      unsigned int nTCReg = TC_[iSeed].size() / NRegions_[l2];

      for (unsigned int iReg = 0; iReg < NRegions_[l2]; iReg++) {
        unsigned int nmem = 0;
        //Hack since we use same module twice
        if (iSeed == Seed::L2D1) {
          nmem = 2;
        }

        for (unsigned iTC = 0; iTC < nTCReg; iTC++) {
          nmem++;
          memories << "VMStubsTE: VMSTE_" << LayerName(ilayer) << "PHI" << iRegStr(iReg, iSeed) << "n" << nmem
                   << " [18]" << std::endl;
          os << "VMSTE_" << LayerName(ilayer) << "PHI" << iRegStr(iReg, iSeed) << "n" << nmem << " input=> VMR_"
             << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".vmstubout_seed_" << iSeed << " output=> TP_"
             << LayerName(l1) << LayerName(l2) << iTCStr(iReg * nTCReg + iTC) << ".outervmstubin" << std::endl;
        }
      }
    }

  } else {
    //First write VMS memories used by MatchEngine
    for (unsigned int ilayer = 0; ilayer < N_LAYER + N_DISK; ilayer++) {
      for (unsigned int iVMME = 0; iVMME < NVMME_[ilayer] * NRegions_[ilayer]; iVMME++) {
        unsigned int iReg = iVMME / NVMME_[ilayer];
        memories << "VMStubsME: VMSME_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << iVMME + 1 << "n1 [18]"
                 << std::endl;
        os << "VMSME_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << iVMME + 1 << "n1"
           << " input=> VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".vmstuboutMEPHI" << iTCStr(iReg)
           << iVMME + 1 << " output=> ME_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << iVMME + 1 << ".vmstubin"
           << std::endl;
      }
    }

    // Next write VMS memories used by TrackletEngine
    // Each TE processes one VM region in inner + outer seeding layers, and needs its own copy of input memories.
    for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
      for (unsigned int innerouterseed = 0; innerouterseed < 2; innerouterseed++) {
        //FIXME - code could be cleaner
        unsigned int l1 = seedLayers(iSeed).first;
        unsigned int l2 = seedLayers(iSeed).second;

        unsigned int NVMTE1 = NVMTE_[iSeed].first;
        unsigned int NVMTE2 = NVMTE_[iSeed].second;

        unsigned int ilayer = l1;
        unsigned int NVMTE = NVMTE1;
        if (innerouterseed == 1) {
          ilayer = l2;
          NVMTE = NVMTE2;
        }

        for (unsigned int iVMTE = 0; iVMTE < NVMTE * NRegions_[ilayer]; iVMTE++) {
          unsigned int iReg = iVMTE / NVMTE;

          unsigned int nmem = 0;

          if (iSeed == Seed::L2D1) {
            nmem = 4;
          }

          for (unsigned int iTE = 0; iTE < TE_[iSeed].size(); iTE++) {
            unsigned int TE1 = TE_[iSeed][iTE].first;  // VM region in inner/outer layer of this TE
            unsigned int TE2 = TE_[iSeed][iTE].second;

            bool used = false;

            if (innerouterseed == 0 && iVMTE == TE1)
              used = true;
            if (innerouterseed == 1 && iVMTE == TE2)
              used = true;

            if (!used)
              continue;

            string inorout = "I";
            if (innerouterseed == 1)
              inorout = "O";

            nmem++;  // Add another copy of memory.
            memories << "VMStubsTE: VMSTE_" << LayerName(ilayer) << "PHI" << iRegStr(iReg, iSeed) << iVMTE + 1 << "n"
                     << nmem << " [18]" << std::endl;
            os << "VMSTE_" << LayerName(ilayer) << "PHI" << iRegStr(iReg, iSeed) << iVMTE + 1 << "n" << nmem
               << " input=> VMR_" << LayerName(ilayer) << "PHI" << iTCStr(iReg) << ".vmstuboutTE" << inorout << "PHI"
               << iRegStr(iReg, iSeed) << iVMTE + 1 << " output=> TE_" << LayerName(l1) << "PHI"
               << iRegStr(TE1 / NVMTE1, iSeed) << TE1 + 1 << "_" << LayerName(l2) << "PHI"
               << iRegStr(TE2 / NVMTE2, iSeed) << TE2 + 1;
            if (innerouterseed == 0) {
              os << ".innervmstubin" << std::endl;
            } else {
              os << ".outervmstubin" << std::endl;
            }
          }
        }
      }
    }
  }
}

void TrackletConfigBuilder::writeTPARMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  // Each TC module (e.g. TC_L1L2A) stores helix params in a single TPAR memory of similar name
  // (e.g. TPAR_L1L2A). The TPAR is subsequently read by the TrackBuilder (FT).

  if (combinedmodules_) {
    for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
      for (unsigned int iTP = 0; iTP < TC_[iSeed].size(); iTP++) {
        memories << "TrackletParameters: TPAR_" << iSeedStr(iSeed) << iTCStr(iTP) << " [56]" << std::endl;
        modules << "TrackletProcessor: TP_" << iSeedStr(iSeed) << iTCStr(iTP) << std::endl;
        os << "TPAR_" << iSeedStr(iSeed) << iTCStr(iTP) << " input=> TP_" << iSeedStr(iSeed) << iTCStr(iTP)
           << ".trackpar output=> FT_" << iSeedStr(iSeed) << ".tparin" << std::endl;
      }
    }
  } else {
    for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
      for (unsigned int iTC = 0; iTC < TC_[iSeed].size(); iTC++) {
        memories << "TrackletParameters: TPAR_" << iSeedStr(iSeed) << iTCStr(iTC) << " [56]" << std::endl;
        modules << "TrackletCalculator: TC_" << iSeedStr(iSeed) << iTCStr(iTC) << std::endl;
        os << "TPAR_" << iSeedStr(iSeed) << iTCStr(iTC) << " input=> TC_" << iSeedStr(iSeed) << iTCStr(iTC)
           << ".trackpar output=> FT_" << iSeedStr(iSeed) << ".tparin" << std::endl;
      }
    }
  }
}

void TrackletConfigBuilder::writeTFMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
    memories << "TrackFit: TF_" << iSeedStr(iSeed) << " [126]" << std::endl;
    modules << "FitTrack: FT_" << iSeedStr(iSeed) << std::endl;
    os << "TF_" << iSeedStr(iSeed) << " input=> FT_" << iSeedStr(iSeed) << ".trackout output=> PD.trackin" << std::endl;
  }
}

void TrackletConfigBuilder::writeCTMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  modules << "PurgeDuplicate: PD" << std::endl;

  for (unsigned int iSeed = 0; iSeed < N_SEED_PROMPT; iSeed++) {
    memories << "CleanTrack: CT_" << iSeedStr(iSeed) << " [126]" << std::endl;
    os << "CT_" << iSeedStr(iSeed) << " input=> PD.trackout output=>" << std::endl;
  }
}

void TrackletConfigBuilder::writeILMemories(std::ostream& os, std::ostream& memories, std::ostream& modules) {
  // Each Input Router (IR) reads stubs from one DTC (e.g. PS10G_1) & sends them
  // to 4-8 InputLink (IL) memories (labelled PHIA-PHIH), each corresponding to a small
  // phi region of a nonant, for each tracklet layer (L1-L6 or D1-D5) that the DTC
  // reads. The InputLink memories have names such as IL_L1PHIC_PS10G_1 to reflect this.

  string olddtc = "";
  for (const DTCinfo& info : vecDTCinfo_) {
    string dtcname = info.name;
    if (olddtc != dtcname) {
      // Write one entry per DTC, with each DTC connected to one IR.
      modules << "InputRouter: IR_" << dtcname << "_A" << std::endl;
      modules << "InputRouter: IR_" << dtcname << "_B" << std::endl;
      memories << "DTCLink: DL_" << dtcname << "_A [36]" << std::endl;
      memories << "DTCLink: DL_" << dtcname << "_B [36]" << std::endl;
      os << "DL_" << dtcname << "_A"
         << " input=> output=> IR_" << dtcname << "_A.stubin" << std::endl;
      os << "DL_" << dtcname << "_B"
         << " input=> output=> IR_" << dtcname << "_B.stubin" << std::endl;
    }
    olddtc = dtcname;
  }

  for (const DTCinfo& info : vecDTCinfo_) {
    string dtcname = info.name;
    int layerdisk = info.layer;

    for (unsigned int iReg = 0; iReg < NRegions_[layerdisk]; iReg++) {
      //--- Ian Tomalin's proposed bug fix
      double phiminDTC_A = info.phimin - M_PI / N_SECTOR;  // Phi range of each DTC.
      double phimaxDTC_A = info.phimax - M_PI / N_SECTOR;
      double phiminDTC_B = info.phimin + M_PI / N_SECTOR;  // Phi range of each DTC.
      double phimaxDTC_B = info.phimax + M_PI / N_SECTOR;
      if (allStubs_[layerdisk][iReg].second > phiminDTC_A && allStubs_[layerdisk][iReg].first < phimaxDTC_A) {
        memories << "InputLink: IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_A"
                 << " [36]" << std::endl;
        os << "IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_A"
           << " input=> IR_" << dtcname << "_A.stubout output=> VMR_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg)
           << ".stubin" << std::endl;
      }
      if (allStubs_[layerdisk][iReg].second > phiminDTC_B && allStubs_[layerdisk][iReg].first < phimaxDTC_B) {
        memories << "InputLink: IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_B"
                 << " [36]" << std::endl;
        os << "IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_B"
           << " input=> IR_" << dtcname << "_B.stubout output=> VMR_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg)
           << ".stubin" << std::endl;
      }
      //--- Original (buggy) code
      /*
      double phiminDTC = info.phimin; // Phi range of each DTC.
      double phimaxDTC = info.phimax;

      if (allStubs_[layerdisk][iReg].first > phimaxDTC && allStubs_[layerdisk][iReg].second < phiminDTC) 
        continue;

      // Phi region range must be entirely contained in this DTC to keep this connection.
      if (allStubs_[layerdisk][iReg].second < phimaxDTC) {
        memories << "InputLink: IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_A"
                 << " [36]" << std::endl;
        os << "IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_A"
           << " input=> IR_" << dtcname << "_A.stubout output=> VMR_" << LayerName(layerdisk) << "PHI"
           << iTCStr(iReg) << ".stubin" << std::endl;
      }

      if (allStubs_[layerdisk][iReg].first > phiminDTC) {
        memories << "InputLink: IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_B"
                 << " [36]" << std::endl;
        os << "IL_" << LayerName(layerdisk) << "PHI" << iTCStr(iReg) << "_" << dtcname << "_B"
           << " input=> IR_" << dtcname << "_B.stubout output=> VMR_" << LayerName(layerdisk) << "PHI"
           << iTCStr(iReg) << ".stubin" << std::endl;
      }
*/
    }
  }
}

//--- Fill streams used to write wiring map to file

void TrackletConfigBuilder::writeAll(std::ostream& wires, std::ostream& memories, std::ostream& modules) {
  writeILMemories(wires, memories, modules);
  writeASMemories(wires, memories, modules);
  writeVMSMemories(wires, memories, modules);
  writeSPMemories(wires, memories, modules);
  writeSPDMemories(wires, memories, modules);
  writeProjectionMemories(wires, memories, modules);
  writeTPARMemories(wires, memories, modules);
  writeVMPROJMemories(wires, memories, modules);
  writeAPMemories(wires, memories, modules);
  writeCMMemories(wires, memories, modules);
  writeFMMemories(wires, memories, modules);
  writeTFMemories(wires, memories, modules);
  writeCTMemories(wires, memories, modules);
}
