// -*- C++ -*-
//
// Package:    HGCalCellPositionTester
// Class:      HGCalCellPositionTester
//
/**\class HGCalCellPositionTester HGCalCellPositionTester.cc
 test/HGCalCellPositionTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2022/01/15
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"

class HGCalCellPositionTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalCellPositionTester(const edm::ParameterSet&);
  ~HGCalCellPositionTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const double waferSize_;
  const int waferType_;
  const int placeIndex_;
};

HGCalCellPositionTester::HGCalCellPositionTester(const edm::ParameterSet& iC)
    : waferSize_(iC.getParameter<double>("waferSize")),
      waferType_(iC.getParameter<int>("waferType")),
      placeIndex_(iC.getParameter<int>("cellPlacementIndex")) {
  edm::LogVerbatim("HGCalGeom") << "Test positions for wafer of size " << waferSize_ << " Type " << waferType_
                                << " Placement Index " << placeIndex_;
}

void HGCalCellPositionTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("waferSize", 166.4408);
  desc.add<int>("waferType", 0);
  desc.add<int>("cellPlacementIndex", 7);
  descriptions.add("hgcalCellPositionTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalCellPositionTester::analyze(const edm::Event&, const edm::EventSetup&) {
  const int nFine(12), nCoarse(8);
  const double tol(0.00001);
  HGCalCell wafer(waferSize_, nFine, nCoarse);
  int nCells = (waferType_ == 0) ? nFine : nCoarse;
  int indexMin = (placeIndex_ >= 0) ? placeIndex_ : 0;
  int indexMax = (placeIndex_ >= 0) ? placeIndex_ : 11;
  edm::LogVerbatim("HGCalGeom") << "\nHGCalCellPositionTester:: nCells " << nCells << " and placement index between "
                                << indexMin << " and " << indexMax << "\n\n";
  for (int placeIndex = indexMin; placeIndex <= indexMax; ++placeIndex) {
    for (int iu = 0; iu < 2 * nCells; ++iu) {
      for (int iv = 0; iv < 2 * nCells; ++iv) {
        int u(iu), v(iv);
        if (placeIndex < HGCalCell::cellPlacementExtra) {
          u = iv;
          v = iu;
        }
        if (((v - u) < nCells) && ((u - v) <= nCells)) {
          std::pair<double, double> xy1 = wafer.cellUV2XY1(u, v, placeIndex, waferType_);
          std::pair<double, double> xy2 = wafer.cellUV2XY2(u, v, placeIndex, waferType_);
          double dx = xy1.first - xy2.first;
          double dy = xy1.second - xy2.second;
          std::string comment = ((std::abs(dx) > tol) || (std::abs(dy) > tol)) ? " ***** ERROR *****" : "";
          edm::LogVerbatim("HGCalGeom") << "u = " << u << " v = " << v << " type = " << waferType_
                                        << " placement index " << placeIndex << " x " << xy1.first << ":" << xy2.first
                                        << ":" << dx << " y " << xy1.second << ":" << xy2.second << ":" << dy
                                        << comment;
        }
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalCellPositionTester);
