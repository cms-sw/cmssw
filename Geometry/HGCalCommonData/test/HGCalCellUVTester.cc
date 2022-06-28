// -*- C++ -*-
//
// Package:    HGCalCellUVTester
// Class:      HGCalCellUVTester
//
/**\class HGCalCellUVTester HGCalCellUVTester.cc
 test/HGCalCellUVTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee, Pruthvi Suryadevara
//         Created:  Mon 2022/01/15
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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellUV.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"

class HGCalCellUVTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalCellUVTester(const edm::ParameterSet&);
  ~HGCalCellUVTester() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const double waferSize_;
  const int waferType_;
  const int placeIndex_;
};

HGCalCellUVTester::HGCalCellUVTester(const edm::ParameterSet& iC)
    : waferSize_(iC.getParameter<double>("waferSize")),
      waferType_(iC.getParameter<int>("waferType")),
      placeIndex_(iC.getParameter<int>("cellPlacementIndex")) {
  edm::LogVerbatim("HGCalGeom") << "Test positions for wafer of size " << waferSize_ << " Type " << waferType_
                                << " Placement Index " << placeIndex_;
}

void HGCalCellUVTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("waferSize", 166.4408);
  desc.add<int>("waferType", 1);
  desc.add<int>("cellPlacementIndex", 6);
  descriptions.add("hgcalCellUVTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalCellUVTester::analyze(const edm::Event&, const edm::EventSetup&) {
  const int nFine(12), nCoarse(8);
  HGCalCellUV wafer(waferSize_, 0.0, nFine, nCoarse);
  HGCalCell wafer2(waferSize_, nFine, nCoarse);
  double r2 = 0.5 * waferSize_;
  double R2 = 2 * r2 / sqrt(3);
  int nCells = (waferType_ == 0) ? nFine : nCoarse;
  int indexMin = (placeIndex_ >= 0) ? placeIndex_ : 0;
  int indexMax = (placeIndex_ >= 0) ? placeIndex_ : 11;
  edm::LogVerbatim("HGCalGeom") << "\nHGCalCellUVTester:: nCells " << nCells << " and placement index between "
                                << indexMin << " and " << indexMax << "\n\n";
  auto start_t = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 100000; i++) {
    double xi = (2 * r2 * (float)rand() / RAND_MAX) - r2;
    double yi = (2 * R2 * (float)rand() / RAND_MAX) - R2;
    double c1 = yi + xi / sqrt(3);
    double c2 = yi - (xi / sqrt(3));
    if ((xi < r2) && (xi > -1 * r2) && (c1 < R2) && (c1 > -1 * R2) && (c2 < R2) &&
        (c2 > -1 * R2)) {  //Only allowing (x, y) inside a wafer
      std::pair<int32_t, int32_t> uv1 = wafer.cellUVFromXY1(xi, yi, placeIndex_, waferType_, true, false);
      std::pair<int32_t, int32_t> uv2 = wafer.cellUVFromXY2(xi, yi, placeIndex_, waferType_, true, false);
      std::pair<int32_t, int32_t> uv3 = wafer.cellUVFromXY3(xi, yi, placeIndex_, waferType_, true, false);
      //std::pair<int32_t, int32_t> uv2 = wafer.HGCalCellUVFromXY2(xi, yi, placeIndex_, waferType_, true, false);
      std::string comment = ((uv1.first != uv3.first) || (uv2.first != uv3.first) || (uv1.second != uv3.second) ||
                             (uv2.second != uv3.second))
                                ? " ***** ERROR *****"
                                : "";
      edm::LogVerbatim("HGCalGeom") << "x = " << xi << " y = " << yi << " type = " << waferType_ << " placement index "
                                    << placeIndex_ << " u " << uv1.first << ":" << uv2.first << ":" << uv3.first
                                    << " v " << uv1.second << ":" << uv2.second << ":" << uv3.second << ":" << comment;
    }
  }
  auto end_t = std::chrono::high_resolution_clock::now();
  auto diff_t = end_t - start_t;
  edm::LogVerbatim("HGCalGeom") << "Execution time for 100000 events = "
                                << std::chrono::duration<double, std::milli>(diff_t).count() << " ms";
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalCellUVTester);
