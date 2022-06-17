// -*- C++ -*-
//
// Package:    HGCalCellPartialCellTester
// Class:      HGCalCellPartialCellTester
//
/**\class HGCalCellPartialCellTester HGCalCellPartialCellTester.cc
 test/HGCalCellPartialCellTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee, Pruthvi Suryadevara
//         Created:  Mon 2022/06/10
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stdlib.h>
#include <cmath>
//#include <chrono>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellUV.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"

class HGCalPartialCellTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalPartialCellTester(const edm::ParameterSet&);
  ~HGCalPartialCellTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const double waferSize_;
  const int waferType_;
  const int placeIndex_;
  const int partialType_;
  const int nTrials_;
  const int modeUV_;
};

HGCalPartialCellTester::HGCalPartialCellTester(const edm::ParameterSet& iC)
    : waferSize_(iC.getParameter<double>("waferSize")),
      waferType_(iC.getParameter<int>("waferType")),
      placeIndex_(iC.getParameter<int>("cellPlacementIndex")),
      partialType_(iC.getParameter<int>("partialType")),
      nTrials_(iC.getParameter<int>("numbberOfTrials")),
      modeUV_(iC.getParameter<int>("modeUV")) {
  edm::LogVerbatim("HGCalGeom") << "Test positions for wafer of size " << waferSize_ << " Type " << waferType_
                                << " Partial Type " << partialType_ << " Placement Index " << placeIndex_ << " mode "
                                << modeUV_ << " with " << nTrials_ << " trials";
}

void HGCalPartialCellTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("waferSize", 166.4408);
  desc.add<int>("waferType", 0);
  desc.add<int>("cellPlacementIndex", 3);
  desc.add<int>("partialType", 25);
  desc.add<int>("numbberOfTrials", 1000);
  desc.add<int>("modeUV", 0);
  descriptions.add("hgcalPartialCellTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalPartialCellTester::analyze(const edm::Event&, const edm::EventSetup&) {
  const int nFine(12), nCoarse(8);
  double r2 = 0.5 * waferSize_;
  double R2 = 2 * r2 / sqrt(3);
  int nCells = (waferType_ == 0) ? nFine : nCoarse;
  HGCalCellUV wafer(waferSize_, 0.0, nFine, nCoarse);
  HGCalCell wafer2(waferSize_, nFine, nCoarse);
  edm::LogVerbatim("HGCalGeom") << "\nHGCalPartialCellTester:: nCells " << nCells << " and placement index "
                                << placeIndex_ << "\n\n";
  auto start_t = std::chrono::high_resolution_clock::now();

  if (modeUV_ <= 0) {
    for (int i = 0; i < nTrials_; i++) {
      double xi = (2 * r2 * static_cast<double>(rand()) / RAND_MAX) - r2;
      double yi = (2 * R2 * static_cast<double>(rand()) / RAND_MAX) - R2;
      bool goodPoint = true;
      int ug = 0;
      int vg = 0;
      if (partialType_ == 11 || partialType_ == 13 || partialType_ == 15 || partialType_ == 21 || partialType_ == 23 ||
          partialType_ == 25) {
        ug = 0;
        vg = 0;
      } else if (partialType_ == 12 || partialType_ == 14 || partialType_ == 16 || partialType_ == 22 ||
                 partialType_ == 24) {
        ug = nCells + 1;
        vg = 2 * (nCells - 1);
        //std::cout << ug << " " << vg << std::endl;
      }
      std::pair<double, double> xyg = wafer2.cellUV2XY2(ug, vg, placeIndex_, waferType_);
      //std::cout << xyg.first << ":" << xyg.second << std::endl;
      std::vector<std::pair<double, double> > wxy =
          HGCalWaferMask::waferXY(partialType_, placeIndex_, r2, R2, 0.0, 0.0);
      for (unsigned int i = 0; i < (wxy.size() - 1); ++i) {
        double xp1 = wxy[i].first;
        double yp1 = wxy[i].second;
        double xp2 = wxy[i + 1].first;
        double yp2 = wxy[i + 1].second;
        if ((((xi - xp1) / (xp2 - xp1)) - ((yi - yp1) / (yp2 - yp1))) *
                (((xyg.first - xp1) / (xp2 - xp1)) - ((xyg.second - yp1) / (yp2 - yp1))) <=
            0) {
          goodPoint = false;
        }
      }
      if (goodPoint) {  //Only allowing (x, y) inside a partial wafer 11, placement index 2
        std::pair<int32_t, int32_t> uv1 = wafer.cellUVFromXY1(xi, yi, placeIndex_, waferType_, true, false);
        std::pair<int32_t, int32_t> uv5 =
            wafer.cellUVFromXY1(xi, yi, placeIndex_, waferType_, partialType_, true, false);
        std::pair<double, double> xy1 = wafer2.cellUV2XY2(uv5.first, uv5.second, placeIndex_, waferType_);
        std::string cellType = (HGCalWaferMask::goodCell(uv5.first, uv5.second, partialType_)) ? "Goodcell" : "Badcell";
        std::string pointType = goodPoint ? "GoodPoint" : "BadPoint";
        std::string comment =
            ((uv1.first != uv5.first) || (uv1.second != uv5.second) || (xy1.first - xi > 6) || (xy1.second - yi > 6))
                ? " ***** ERROR *****"
                : "";
        edm::LogVerbatim("HGCalGeom") << pointType << " " << cellType << " x = " << xi << ":" << xy1.first
                                      << " y = " << yi << ":" << xy1.second << " type = " << waferType_
                                      << " placement index " << placeIndex_ << " u " << uv1.first << ":" << uv5.first
                                      << ":" << uv5.first << " v " << uv1.second << ":" << uv5.second << ":"
                                      << uv5.second << ":" << comment;
      }
    }
  } else {
    for (int i = 0; i < nTrials_; i++) {
      int ui = std::floor(2 * nCells * rand() / RAND_MAX);
      int vi = std::floor(2 * nCells * rand() / RAND_MAX);
      if ((ui < 2 * nCells) && (vi < 2 * nCells) && ((vi - ui) < nCells) && ((ui - vi) <= nCells) &&
          HGCalWaferMask::goodCell(ui, vi, partialType_)) {
        //Only allowing (U, V) inside a wafer
        std::pair<double, double> xy1 = wafer2.cellUV2XY2(ui, vi, placeIndex_, waferType_);
        std::pair<int32_t, int32_t> uv1 =
            wafer.cellUVFromXY1(xy1.first, xy1.second, placeIndex_, waferType_, true, false);
        std::pair<int32_t, int32_t> uv5 =
            wafer.cellUVFromXY1(xy1.first, xy1.second, placeIndex_, waferType_, partialType_, true, false);
        std::string comment = ((uv1.first != ui) || (uv1.second != vi) || (uv5.first != ui) || (uv5.second != vi))
                                  ? " ***** ERROR *****"
                                  : "";
        edm::LogVerbatim("HGCalGeom") << "u = " << ui << " v = " << vi << " type = " << waferType_
                                      << " placement index " << placeIndex_ << " x " << xy1.first << " ,y "
                                      << xy1.second << " u " << uv5.first << " v " << uv5.second << comment;
      }
    }
  }
  auto end_t = std::chrono::high_resolution_clock::now();
  auto diff_t = end_t - start_t;
  edm::LogVerbatim("HGCalGeom") << "Execution time for " << nTrials_
                                << " events = " << std::chrono::duration<double, std::milli>(diff_t).count() << " ms";
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalPartialCellTester);
