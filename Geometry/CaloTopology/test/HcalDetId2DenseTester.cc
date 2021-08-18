#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HcalDetId2DenseTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HcalDetId2DenseTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTestFile(const HcalTopology& topology);
  void doTestHcalDetId(const HcalTopology& topology);
  void doTestHcalCalibDetId(const HcalTopology& topology);
  void doTestOnlyHcalCalibDetId(const HcalTopology& topology);
  void doTestOneCell(const HcalTopology&, const std::string&, const HcalCalibDetId&, int&, int&, int&, int&);
  std::vector<std::string> splitString(const std::string& fLine);
  // ----------member data ---------------------------
  const std::string fileName_;
  const bool testCalib_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tokTopo_;
};

HcalDetId2DenseTester::HcalDetId2DenseTester(const edm::ParameterSet& iC)
    : fileName_(iC.getUntrackedParameter<std::string>("fileName", "")),
      testCalib_(iC.getUntrackedParameter<bool>("testCalib", false)),
      tokTopo_{esConsumes<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{})} {}

void HcalDetId2DenseTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("fileName", "");
  desc.addUntracked<bool>("testCalib", false);
  descriptions.add("hcalDetId2DenseTester", desc);
}

void HcalDetId2DenseTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  const auto& topo = iSetup.getData(tokTopo_);
  if (!testCalib_) {
    doTestFile(topo);
    doTestHcalDetId(topo);
    doTestHcalCalibDetId(topo);
  } else {
    doTestOnlyHcalCalibDetId(topo);
  }
}

void HcalDetId2DenseTester::doTestFile(const HcalTopology& topology) {
  if (!fileName_.empty()) {
    std::ifstream fInput(fileName_);
    if (!fInput.good()) {
      std::cout << "Cannot open file " << fileName_ << std::endl;
    } else {
      char buffer[1024];
      unsigned int all(0), total(0), good(0), ok(0), bad(0);
      while (fInput.getline(buffer, 1024)) {
        ++all;
        if (buffer[0] == '#')
          continue;  //ignore comment
        std::vector<std::string> items = splitString(std::string(buffer));
        if (items.size() != 4) {
          std::cout << "Ignore  line: " << buffer << std::endl;
        } else {
          ++total;
          int ieta = std::atoi(items[1].c_str());
          ;
          int iphi = std::atoi(items[2].c_str());
          ;
          std::string error;
          if ((items[0] == "HB") || (items[0] == "HE") || (items[0] == "HF") || (items[0] == "HO")) {
            HcalSubdetector sd(HcalBarrel);
            if (items[0] == "HE") {
              sd = HcalEndcap;
            } else if (items[0] == "HF") {
              sd = HcalForward;
            } else if (items[0] == "HO") {
              sd = HcalOuter;
            }
            int depth = std::atoi(items[3].c_str());
            HcalDetId cell(sd, ieta, iphi, depth);
            if (topology.valid(cell)) {
              ++ok;
              unsigned int dense = topology.detId2denseId(DetId(cell));
              DetId id = topology.denseId2detId(dense);
              if (cell == HcalDetId(id)) {
                error = "";
                ++good;
              } else {
                error = "***ERROR***";
                ++bad;
              }
              std::cout << total << " " << cell << " Dense Index " << dense << " gives back " << HcalDetId(id) << " "
                        << error << std::endl;
            }

          } else if (items[0].find("CALIB_") == 0) {
            HcalSubdetector sd = HcalOther;
            if (items[0].find("HB") != std::string::npos)
              sd = HcalBarrel;
            else if (items[0].find("HE") != std::string::npos)
              sd = HcalEndcap;
            else if (items[0].find("HO") != std::string::npos)
              sd = HcalOuter;
            else if (items[0].find("HF") != std::string::npos)
              sd = HcalForward;
            int channel = std::atoi(items[3].c_str());
            HcalCalibDetId cell(sd, ieta, iphi, channel);
            if (topology.validCalib(cell)) {
              ++ok;
              unsigned int dense = topology.detId2denseIdCALIB(DetId(cell));
              DetId id = topology.denseId2detIdCALIB(dense);
              if (cell == HcalCalibDetId(id)) {
                error = "";
                ++good;
              } else {
                error = "***ERROR***";
                ++bad;
              }
              std::cout << total << " " << cell << " Dense Index " << dense << " gives back " << HcalCalibDetId(id)
                        << " " << error << std::endl;
            }
          } else if ((items[0] == "HOX") || (items[0] == "HBX") || (items[0] == "HEX")) {
            HcalCalibDetId cell =
                ((items[0] == "HOX") ? (HcalCalibDetId(HcalCalibDetId::HOCrosstalk, ieta, iphi))
                                     : ((items[0] == "HBX") ? (HcalCalibDetId(HcalCalibDetId::HBX, ieta, iphi))
                                                            : (HcalCalibDetId(HcalCalibDetId::HEX, ieta, iphi))));
            if (topology.validCalib(cell)) {
              ++ok;
              unsigned int dense = topology.detId2denseIdCALIB(DetId(cell));
              DetId id = topology.denseId2detIdCALIB(dense);
              if (cell == HcalCalibDetId(id)) {
                error = "";
                ++good;
              } else {
                error = "***ERROR***";
                ++bad;
              }
              std::cout << good << " " << cell << " Dense Index " << dense << " gives back " << HcalCalibDetId(id)
                        << " " << error << std::endl;
            }
          }
        }
      }
      fInput.close();
      std::cout << "Reads total of " << all << ":" << total << " records with " << good << ":" << ok << " good and "
                << bad << " bad DetId's" << std::endl;
    }
  }
}

void HcalDetId2DenseTester::doTestHcalDetId(const HcalTopology& topology) {
  const int n = 48;
  HcalSubdetector sds[n] = {HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,
                            HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalEndcap,  HcalEndcap,
                            HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalEndcap,
                            HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,
                            HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,
                            HcalOuter,   HcalForward, HcalForward, HcalForward, HcalForward, HcalForward, HcalForward,
                            HcalForward, HcalForward, HcalForward, HcalForward, HcalForward, HcalForward};
  int ietas[n] = {1, 5, 9, 12, 14, 16, -2, -4, -8, -11, -13, -16, 18, 21, 23, 25, 27, 29, -19, -20, -22, -24, -26, -28,
                  1, 3, 5, 9,  11, 13, -2, -4, -6, -10, -12, -14, 30, 32, 34, 36, 38, 40, -31, -33, -35, -37, -39, -41};
  int iphis[n] = {11, 21, 31, 41, 51, 61, 5, 15, 25, 35, 45, 65, 11, 23, 35, 47, 59, 71, 3, 15, 27, 39, 51, 63,
                  11, 21, 31, 41, 51, 61, 5, 15, 25, 35, 45, 65, 11, 23, 35, 47, 59, 71, 3, 15, 27, 39, 51, 63};
  int depth[n] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};

  // Check on Dense Index
  std::cout << "\nCheck on Dense Index for DetId's" << std::endl << "==================================" << std::endl;
  int total(0), good(0), bad(0);
  std::string error;
  for (int i = 0; i < n; ++i) {
    HcalDetId cell(sds[i], ietas[i], iphis[i], depth[i]);
    if (topology.valid(cell)) {
      ++total;
      unsigned int dense = topology.detId2denseId(DetId(cell));
      DetId id = topology.denseId2detId(dense);
      if (cell == HcalDetId(id)) {
        ++good;
        error = "";
      } else {
        ++bad;
        error = "**** ERROR *****";
      }
      std::cout << "[" << total << "] " << cell << " Dense " << dense << " o/p " << HcalDetId(id) << " " << error
                << std::endl;
    }
  }

  std::cout << "Analyzes total of " << n << ":" << total << " HcalDetIds with " << good << " good and " << bad
            << " bad DetId's" << std::endl;
}

void HcalDetId2DenseTester::doTestHcalCalibDetId(const HcalTopology& topology) {
  const int n = 48;
  HcalCalibDetId::CalibDetType dets[n] = {HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::CalibrationBox,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HOCrosstalk,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HBX,
                                          HcalCalibDetId::HEX,
                                          HcalCalibDetId::HEX,
                                          HcalCalibDetId::HEX,
                                          HcalCalibDetId::HEX,
                                          HcalCalibDetId::HEX,
                                          HcalCalibDetId::HEX,
                                          HcalCalibDetId::HEX,
                                          HcalCalibDetId::HEX};
  HcalSubdetector sds[n] = {HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,
                            HcalBarrel,  HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalOuter,   HcalOuter,
                            HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,   HcalForward, HcalForward, HcalForward,
                            HcalForward, HcalForward, HcalForward, HcalBarrel,  HcalBarrel,  HcalBarrel,  HcalBarrel,
                            HcalBarrel,  HcalBarrel,  HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalEndcap,  HcalEndcap,
                            HcalEndcap,  HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,   HcalOuter,
                            HcalForward, HcalForward, HcalForward, HcalForward, HcalForward, HcalForward};
  int ietas[n] = {1,   1,   1,   -1,  -1, -1, 1,  1,  1,  -1, -1, -1, 1,   -1,  0,   2,
                  -1,  -2,  1,   1,   -1, -1, 1,  -1, 4,  4,  -4, -4, 15,  15,  -15, -15,
                  -16, -16, -16, -16, 16, 16, 16, 16, 25, 25, 27, 27, -25, -25, -27, -27};
  int iphis[n] = {11, 23, 35, 47, 59, 71, 3, 15, 27, 39, 51, 65, 22, 11, 35, 71, 47, 59, 19, 37, 55, 1,  19, 37,
                  47, 36, 36, 41, 51, 61, 5, 15, 25, 35, 45, 65, 11, 1,  35, 47, 59, 71, 3,  15, 27, 39, 51, 63};
  int depth[n] = {0, 1, 2, 0, 1, 2, 0, 1, 3, 4, 5, 6, 0, 1, 0, 1, 7, 7, 0, 1, 0, 1, 0, 1,
                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};

  // Check on Dense Index
  std::cout << "\nCheck on Dense Index for CalibDetId's" << std::endl
            << "=======================================" << std::endl;
  int total(0), good(0), bad(0);
  std::string error;
  for (int i = 0; i < n; ++i) {
    HcalCalibDetId cell;
    if (dets[i] == HcalCalibDetId::CalibrationBox) {
      cell = HcalCalibDetId(sds[i], ietas[i], iphis[i], depth[i]);
    } else {
      cell = HcalCalibDetId(dets[i], ietas[i], iphis[i]);
    }
    if (topology.validCalib(cell)) {
      ++total;
      unsigned int dense = topology.detId2denseIdCALIB(DetId(cell));
      DetId id = topology.denseId2detIdCALIB(dense);
      if (cell == HcalCalibDetId(id)) {
        ++good;
        error = "";
      } else {
        ++bad;
        error = "**** ERROR *****";
      }
      std::cout << "[" << total << "] " << cell << " Dense " << dense << " o/p " << HcalCalibDetId(id) << " " << error
                << std::endl;
    }
  }
  std::cout << "Analyzes total of " << n << ":" << total << " CalibIds with " << good << " good and " << bad
            << " bad DetId's" << std::endl;
}

void HcalDetId2DenseTester::doTestOnlyHcalCalibDetId(const HcalTopology& topology) {
  // Check on Dense Index
  std::cout << "\nCheck on Dense Index for CalibDetId's" << std::endl
            << "=======================================" << std::endl;
  int allT(0), totalT(0), goodT(0), badT(0);
  int all(0), total(0), good(0), bad(0);
  // CalibrationBox (HB)
  for (int i1 = 0; i1 < 2; ++i1) {
    int ieta = 2 * i1 - 1;
    for (int i2 = 0; i2 < 3; ++i2) {
      int channel = i2;
      for (int i3 = 0; i3 < 18; ++i3) {
        int iphi = 4 * i3 + 3;
        HcalCalibDetId cell(HcalBarrel, ieta, iphi, channel);
        doTestOneCell(topology, "HcalBarrel", cell, all, total, good, bad);
      }
    }
  }
  std::cout << "CalibHB:" << all << ":" << total << " CalibIds with " << good << " good and " << bad
            << " bad DetId's\n\n";
  allT += all;
  totalT += total;
  goodT += good;
  badT += bad;
  all = 0;
  total = 0;
  good = 0;
  bad = 0;

  // CalibrationBox (HE)
  for (int i1 = 0; i1 < 2; ++i1) {
    int ieta = 2 * i1 - 1;
    for (int i2 = 0; i2 < 7; ++i2) {
      int channel = i2;
      for (int i3 = 0; i3 < 18; ++i3) {
        int iphi = 4 * i3 + 3;
        HcalCalibDetId cell(HcalEndcap, ieta, iphi, channel);
        doTestOneCell(topology, "HcalEndcap", cell, all, total, good, bad);
      }
    }
  }
  std::cout << "CalibHE:" << all << ":" << total << " CalibIds with " << good << " good and " << bad
            << " bad DetId's\n\n";
  allT += all;
  totalT += total;
  goodT += good;
  badT += bad;
  all = 0;
  total = 0;
  good = 0;
  bad = 0;

  // CalibrationBox (HF)
  int chanHF[4] = {0, 1, 8, 9};
  for (int i1 = 0; i1 < 2; ++i1) {
    int ieta = 2 * i1 - 1;
    for (int i2 = 0; i2 < 4; ++i2) {
      int channel = chanHF[i2];
      int phimax = (i2 == 3) ? 1 : 4;
      for (int i3 = 0; i3 < phimax; ++i3) {
        int iphi = 18 * i3 + 3;
        HcalCalibDetId cell(HcalForward, ieta, iphi, channel);
        doTestOneCell(topology, "HcalForward", cell, all, total, good, bad);
      }
    }
  }
  std::cout << "CalibHF:" << all << ":" << total << " CalibIds with " << good << " good and " << bad
            << " bad DetId's\n\n";
  allT += all;
  totalT += total;
  goodT += good;
  badT += bad;
  all = 0;
  total = 0;
  good = 0;
  bad = 0;

  // CalibrationBox (HO)
  int chanHO[3] = {0, 1, 7};
  int phiHO[5] = {59, 47, 53, 47, 47};
  for (int i1 = 0; i1 < 5; ++i1) {
    int ieta = i1 - 2;
    for (int i2 = 0; i2 < 3; ++i2) {
      int channel = chanHO[i2];
      int phimax = (i2 == 2) ? 1 : ((ieta == 0) ? 12 : 6);
      for (int i3 = 0; i3 < phimax; ++i3) {
        int iphi = (i2 == 2) ? phiHO[i1] : ((ieta == 0) ? 6 * i3 + 5 : 12 * i3 + 11);
        HcalCalibDetId cell(HcalOuter, ieta, iphi, channel);
        doTestOneCell(topology, "HcalOuter", cell, all, total, good, bad);
      }
    }
  }
  std::cout << "CalibHO:" << all << ":" << total << " CalibIds with " << good << " good and " << bad
            << " bad DetId's\n\n";
  allT += all;
  totalT += total;
  goodT += good;
  badT += bad;
  all = 0;
  total = 0;
  good = 0;
  bad = 0;

  // HOX
  int etaHOX[4] = {4, -4, 15, -15};
  for (int i1 = 0; i1 < 4; ++i1) {
    int ieta = etaHOX[i1];
    int phimax = (i1 < 2) ? 36 : 72;
    for (int i3 = 0; i3 < phimax; ++i3) {
      int iphi = (i1 >= 2) ? i3 + 1 : ((i3 + 4) % 12 < 6) ? 2 * i3 + 1 : 2 * i3 + 2;
      HcalCalibDetId cell(HcalCalibDetId::HOCrosstalk, ieta, iphi);
      doTestOneCell(topology, "HOX", cell, all, total, good, bad);
    }
  }
  std::cout << "HOX:" << all << ":" << total << " CalibIds with " << good << " good and " << bad << " bad DetId's\n\n";
  allT += all;
  totalT += total;
  goodT += good;
  badT += bad;
  all = 0;
  total = 0;
  good = 0;
  bad = 0;

  // HBX
  int etaHBX[2] = {16, -16};
  for (int i1 = 0; i1 < 2; ++i1) {
    int ieta = etaHBX[i1];
    for (int i3 = 0; i3 < 72; ++i3) {
      int iphi = i3 + 1;
      HcalCalibDetId cell(HcalCalibDetId::HBX, ieta, iphi);
      doTestOneCell(topology, "HBX", cell, all, total, good, bad);
    }
  }
  std::cout << "HBX:" << all << ":" << total << " CalibIds with " << good << " good and " << bad << " bad DetId's\n\n";
  allT += all;
  totalT += total;
  goodT += good;
  badT += bad;
  all = 0;
  total = 0;
  good = 0;
  bad = 0;

  // HEX
  int etaHEX[4] = {25, -25, 27, -27};
  for (int i1 = 0; i1 < 4; ++i1) {
    int ieta = etaHEX[i1];
    for (int i3 = 0; i3 < 36; ++i3) {
      int iphi = 2 * i3 + 1;
      HcalCalibDetId cell(HcalCalibDetId::HEX, ieta, iphi);
      doTestOneCell(topology, "HEX", cell, all, total, good, bad);
    }
  }
  std::cout << "HEX:" << all << ":" << total << " CalibIds with " << good << " good and " << bad << " bad DetId's\n\n";
  allT += all;
  totalT += total;
  goodT += good;
  badT += bad;

  std::cout << "\nAnalyzes total of " << allT << ":" << totalT << " CalibIds with " << goodT << " good and " << badT
            << " bad DetId's" << std::endl;
}

void HcalDetId2DenseTester::doTestOneCell(const HcalTopology& topology,
                                          const std::string& det,
                                          const HcalCalibDetId& cell,
                                          int& all,
                                          int& total,
                                          int& good,
                                          int& bad) {
  std::string error;
  ++all;
  if (topology.validCalib(cell)) {
    ++total;
    unsigned int dense = topology.detId2denseIdCALIB(DetId(cell));
    DetId id = topology.denseId2detIdCALIB(dense);
    if (cell == HcalCalibDetId(id)) {
      ++good;
      error = "";
    } else {
      ++bad;
      error = "***** ERROR *****";
    }
    std::cout << det << "[" << all << ":" << total << "] " << cell << " Dense " << dense << " o/p "
              << HcalCalibDetId(id) << " " << error << "\n";
  } else {
    std::cout << det << "[" << all << "] " << cell << "***** INVALID *****\n";
  }
}

std::vector<std::string> HcalDetId2DenseTester::splitString(const std::string& fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.push_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDetId2DenseTester);
