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
#include "Geometry/HGCalCommonData/interface/HGCalCellOffset.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"

class HGCalCellOffsetTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalCellOffsetTester(const edm::ParameterSet&);
  ~HGCalCellOffsetTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const double waferSize_;
  const int waferType_;
  const int placeIndex_;
  const int partialType_;
  const double mouseBiteCut_;
  const double guardRingOffset_;
};

HGCalCellOffsetTester::HGCalCellOffsetTester(const edm::ParameterSet& iC)
    : waferSize_(iC.getParameter<double>("waferSize")),
      waferType_(iC.getParameter<int>("waferType")),
      placeIndex_(iC.getParameter<int>("cellPlacementIndex")),
      partialType_(iC.getParameter<int>("partialType")),
      mouseBiteCut_(iC.getParameter<double>("mouseBiteCut")),
      guardRingOffset_(iC.getParameter<double>("guardRingOffset")) {
  edm::LogVerbatim("HGCalGeom") << "Test positions for wafer of size " << waferSize_ << " Type " << waferType_
                                << " Placement Index " << placeIndex_ << " GuardRing offset " << guardRingOffset_
                                << " Mousebite cut " << mouseBiteCut_;
}

void HGCalCellOffsetTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("waferSize", 166.4408);
  desc.add<int>("waferType", 0);
  desc.add<int>("cellPlacementIndex", 3);
  desc.add<int>("partialType", 0);
  desc.add<double>("mouseBiteCut", 0.0);
  desc.add<double>("guardRingOffset", 0.0);
  descriptions.add("hgcalCellOffsetTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalCellOffsetTester::analyze(const edm::Event&, const edm::EventSetup&) {
  const int nFine(12), nCoarse(8);
  int nCells = (waferType_ == 0) ? nFine : nCoarse;
  HGCalCellUV wafer(waferSize_, 0.0, nFine, nCoarse);
  HGCalCell wafer2(waferSize_, nFine, nCoarse);
  HGCalCellOffset offset(waferSize_, nFine, nCoarse, guardRingOffset_, mouseBiteCut_);
  edm::LogVerbatim("HGCalGeom") << "\nHGCalPartialCellTester:: nCells " << nCells << " and placement index "
                                << placeIndex_ << "\n\n";
  for (int ui = 0; ui < 2 * nCells; ui++) {
    for (int vi = 0; vi < 2 * nCells; vi++) {
      if ((ui < 2 * nCells) && (vi < 2 * nCells) && ((vi - ui) < nCells) && ((ui - vi) <= nCells)) {
        //Only allowing (U, V) inside a wafer
        std::pair<double, double> xy1 = wafer2.cellUV2XY2(ui, vi, placeIndex_, waferType_);
        std::pair<double, double> xyOffset = offset.cellOffsetUV2XY1(ui, vi, placeIndex_, waferType_);
        outFile1 << xy1.first << "    " << xy1.second << std::endl;
        outFile2 << (xy1.first + xyOffset.first) << "    " << (xy1.second + xyOffset.second) << std::endl;
        std::pair<int32_t, int32_t> uv1 =
            wafer.cellUVFromXY1(xy1.first, xy1.second, placeIndex_, waferType_, true, false);
        std::pair<int32_t, int32_t> uv5 =
            wafer.cellUVFromXY1(xy1.first, xy1.second, placeIndex_, waferType_, partialType_, true, false);
        std::string comment = ((uv1.first != ui) || (uv1.second != vi) || (uv5.first != ui) || (uv5.second != vi))
                                  ? " ***** ERROR *****"
                                  : "";
        edm::LogVerbatim("HGCalGeom") << "u = " << ui << " v = " << vi << " type = " << waferType_
                                      << " placement index " << placeIndex_ << " u " << uv5.first << " v " << uv5.second
                                      << " x " << xy1.first << " ,y " << xy1.second << " xoff " << xyOffset.first
                                      << " ,yoff " << xyOffset.second;
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalCellOffsetTester);
