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
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellUV.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellOffset.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

class HGCalCellUVTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalCellUVTester(const edm::ParameterSet &);
  ~HGCalCellUVTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}
  //void analyze(edm::Run const &iRun, edm::EventSetup const &iSetup) override;

private:
  const std::vector<std::string> nameDetectors_;
  const std::string fileName_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> tok_hgcal_;

  //const HGCalDDDConstants * hgcCons_;
  const double waferSize_;
  const int waferType_;
  const int placeIndex_;
  const int partial_;
  std::ofstream outputFile;
  std::ofstream outputFile2;
};

HGCalCellUVTester::HGCalCellUVTester(const edm::ParameterSet &iC)
    : nameDetectors_(iC.getParameter<std::vector<std::string>>("nameDetectors")),
      tok_hgcal_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", "HGCalEESensitive"})),
      waferSize_(iC.getParameter<double>("waferSize")),
      waferType_(iC.getParameter<int>("waferType")),
      placeIndex_(iC.getParameter<int>("cellPlacementIndex")),
      partial_(iC.getParameter<int>("cellPartial")) {
  edm::LogVerbatim("HGCalGeom") << "Test positions for wafer of size " << waferSize_ << " Type " << waferType_
                                << " Placement Index " << placeIndex_;
  outputFile.open("nand.csv");
  if (!outputFile.is_open()) {
    edm::LogError("HGCalGeom") << "Could not open output file.";
  } else {
    outputFile << "lay, wu, wv, wx, wy, x, y, cox, coy, cu, cv, cx, cy, cix, ciy, area, ctype, cpos, place, cwx, cwy\n";
  }
  outputFile2.open("DetIDs.csv");
}

void HGCalCellUVTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("nameDetectors", {"HGCalEESensitive"});
  desc.add<double>("waferSize", 167.4408);
  desc.add<int>("waferType", 1);
  desc.add<int>("cellPlacementIndex", 6);
  desc.add<int>("cellPartial", 0);
  descriptions.add("hgcalCellUVTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalCellUVTester::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  const edm::ESHandle<HGCalDDDConstants> &hgcCons_ = iSetup.getHandle(tok_hgcal_);
  const HGCalDDDConstants *cons = hgcCons_.product();
  auto hgpar_ = cons->getParameter();
  const int nFine(12), nCoarse(8);
  HGCalCellUV wafer(waferSize_, 0.0, nFine, nCoarse);
  HGCalCell wafer2(waferSize_, nFine, nCoarse);

  HGCalCellOffset offset(waferSize_, nFine, nCoarse, 0.8985, 5.834, 0.87);
  double r2 = 0.5 * waferSize_;
  double R2 = 2 * r2 / sqrt(3);
  int nCells = (waferType_ == 0) ? nFine : nCoarse;
  int indexMin = (placeIndex_ >= 0) ? placeIndex_ : 0;
  int indexMax = (placeIndex_ >= 0) ? placeIndex_ : 11;
  edm::LogVerbatim("HGCalGeom") << "\nHGCalCellUVTester:: nCells " << nCells << " and placement index between "
                                << indexMin << " and " << indexMax << "\n\n";
  auto start_t = std::chrono::high_resolution_clock::now();
  unsigned int kk(0);
  std::unordered_map<int32_t, HGCalParameters::waferInfo>::const_iterator itr = hgpar_->waferInfoMap_.begin();
  for (; itr != hgpar_->waferInfoMap_.end(); ++itr, ++kk) {
    //for (auto itr = hgpar_->waferInfoMap_.begin(); itr != hgpar_->waferInfoMap_.end(); ++itr){
    if ((itr->second).part != HGCalTypes::WaferFull) {
      int indx = itr->first;
      //if ((HGCalWaferIndex::waferU(indx) == 7) && (HGCalWaferIndex::waferV(indx) == 5)){
      int partial_ = HGCalWaferType::getPartial(indx, hgpar_->waferInfoMap_);
      int orient = HGCalWaferType::getOrient(indx, hgpar_->waferInfoMap_);
      int layer = HGCalWaferIndex::waferLayer(indx);
      if (layer >= 1) {
        int waferU = HGCalWaferIndex::waferU(indx);
        int waferV = HGCalWaferIndex::waferV(indx);
        int waferType_ = (itr->second).type;
        nCells = (waferType_ == 0) ? nFine : nCoarse;
        int FrontBack = (1 + layer) % 2;
        int placeIndex_ = HGCalCell::cellPlacementIndex(-1, FrontBack, orient);
        auto waferxy = cons->waferPositionWithCshift(layer, waferU, waferV, true, true, false);
        for (int uu = 0; uu < 2 * nCells; uu++) {
          for (int vv = 0; vv < 2 * nCells; vv++) {
            if (((vv - uu) < nCells) && ((uu - vv) <= nCells)) {
              if (HGCalWaferMask::goodCell(uu, vv, partial_)) {
                uint32_t DetID =
                    static_cast<DetId>(HGCSiliconDetId(DetId::HGCalEE, 1, waferType_, layer, waferU, waferV, uu, vv));
                HGCalDetId iD(DetID);
                //if (iD.isValid());
                //  std::cout << "true" << std::endl;
                //} else {
                //  std::cout << "false" << std::endl;
                //  }
                //}
                outputFile2 << DetID << std::endl;
              }
            }
          }
        }
        for (int i = 0; i < 1000; i++) {
          double xi = (2 * r2 * (float)rand() / RAND_MAX) - r2;
          double yi = (2 * R2 * (float)rand() / RAND_MAX) - R2;
          double c1 = yi + xi / sqrt(3);
          double c2 = yi - (xi / sqrt(3));
          if ((xi < r2) && (xi > -1 * r2) && (c1 < R2) && (c1 > -1 * R2) && (c2 < R2) &&
              (c2 > -1 * R2)) {  //Only allowing (x, y) inside a wafer
            std::pair<int32_t, int32_t> uv1 = wafer.cellUVFromXY1(xi, yi, placeIndex_, waferType_, true, false);
            std::pair<int32_t, int32_t> uv2 = wafer.cellUVFromXY2(xi, yi, placeIndex_, waferType_, true, false);
            std::pair<int32_t, int32_t> uv3 = wafer.cellUVFromXY3(xi, yi, placeIndex_, waferType_, true, false);
            int ui = uv1.first;
            int vi = uv1.second;
            if (HGCalWaferMask::goodCell(ui, vi, partial_)) {
              auto area = offset.cellAreaUV(ui, vi, placeIndex_, waferType_, partial_, true);
              std::pair<double, double> xy1 = wafer2.cellUV2XY2(ui, vi, placeIndex_, waferType_);
              std::pair<double, double> xyOffsetLD = offset.cellOffsetUV2XY1(ui, vi, placeIndex_, waferType_, partial_);
              auto cellType = HGCalCell::cellType(ui, vi, nCells, placeIndex_, partial_);
	      auto Totalxy = cons->locateCell(-1, layer, waferU, waferV, ui, vi, true, true, false, true, true);
              //std::pair<int32_t, int32_t> uv2 = wafer.HGCalCellUVFromXY2(xi, yi, placeIndex_, waferType_, true, false);
              outputFile << layer << "," << waferU << "," << waferV << "," << waferxy.first << "," << waferxy.second
                         << "," << xi << "," << yi << "," << -10 * waferxy.first - xyOffsetLD.first - xy1.first << ","
                         << 10 * waferxy.second + xyOffsetLD.second + xy1.second << "," << uv1.first << ","
                         << uv1.second << "," << xy1.first << "," << xy1.second << "," << xyOffsetLD.first << ","
                         << xyOffsetLD.second << "," << area << "," << cellType.second << "," << cellType.first << ","
                         << placeIndex_ << "," << Totalxy.first << "," << Totalxy.second << std::endl;
              std::string comment = ((uv1.first != uv3.first) || (uv2.first != uv3.first) ||
                                     (uv1.second != uv3.second) || (uv2.second != uv3.second))
                                        ? " ***** ERROR *****"
                                        : "";
              edm::LogVerbatim("HGCalGeom")
                  //<< "x = " << xi << " y = " << yi << " type = " << waferType_ << " placement index " << placeIndex_
                  //<< " u " << uv1.first << ":" << uv2.first << ":" << uv3.first << " v " << uv1.second << ":"
                  //<< uv2.second << ":" << uv3.second << ":"
                  << layer << "," << waferU << ":" << waferV << "," << waferxy.first << ":" << waferxy.second << ","
                  << xi << ":" << yi << "," << -10 * waferxy.first - xyOffsetLD.first - xy1.first << ":"
                  << 10 * waferxy.second + xyOffsetLD.second + xy1.second << "," << uv1.first << ":" << uv1.second
                  << "," << xy1.first << ":" << xy1.second << " cell cen " << xy1.first << ":" << xy1.second << " Cell cog " << xyOffsetLD.first << ":" << xyOffsetLD.second << comment;
            }
          }
        }
      }
    }
  }
  auto end_t = std::chrono::high_resolution_clock::now();
  auto diff_t = end_t - start_t;
  edm::LogVerbatim("HGCalGeom") << "Execution time for 100000 events = "
                                << std::chrono::duration<double, std::milli>(diff_t).count() << " ms";
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalCellUVTester);
