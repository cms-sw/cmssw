// -*- C++ -*-
//
// Package:    HGCalWaferCell
// Class:      HGCalWaferCell
//
/**\class HGCalWaferCell HGCalWaferCell.cc
 test/HGCalWaferCell.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2020/09/15
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferCell : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferCell(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_, nameDetector_;
  const bool debug_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalWaferCell::HGCalWaferCell(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("NameSense")),
      nameDetector_(iC.getParameter<std::string>("NameDevice")),
      debug_(iC.getParameter<bool>("Verbosity")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  std::cout << "Check # of cells for " << nameDetector_ << " using constants of " << nameSense_ << std::endl;
}

void HGCalWaferCell::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("NameSense", "HGCalEESensitive");
  desc.add<std::string>("NameDevice", "HGCal EE");
  desc.add<bool>("Verbosity", false);
  descriptions.add("hgcalEEWaferCell", desc);
}

// ------------ method called to produce the data  ------------
void HGCalWaferCell::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  const auto& hgdc = geom->topology().dddConstants();

  if (hgdc.waferHexagon8()) {
    // Find all valid wafers
    std::vector<DetId> ids = geom->getValidGeomDetIds();
    std::cout << "\n\nCheck Wafers for " << nameDetector_ << " with " << ids.size() << " wafers\n\n";
    DetId::Detector det = (nameSense_ == "HGCalHESiliconSensitive") ? DetId::HGCalHSi : DetId::HGCalEE;
    int bad(0);
    std::map<int, int> waferMap;
    for (unsigned int k = 0; k < hgdc.waferFileSize(); ++k) {
      int indx = hgdc.waferFileIndex(k);
      int layer = HGCalWaferIndex::waferLayer(indx);
      int waferU = HGCalWaferIndex::waferU(indx);
      int waferV = HGCalWaferIndex::waferV(indx);
      int type = std::get<0>(hgdc.waferFileInfo(k));
      int part = std::get<1>(hgdc.waferFileInfo(k));
      int rotn = std::get<2>(hgdc.waferFileInfo(k));
      HGCSiliconDetId id(det, 1, type, layer, waferU, waferV, 0, 0);
      int kndx = ((rotn * 10 + part) * 10 + type);
      if (debug_)
        std::cout << std::hex << id.rawId() << std::dec << " " << id << " Index:" << kndx << std::endl;
      if (waferMap.find(kndx) == waferMap.end()) {
        waferMap[kndx] = 1;
      } else {
        ++waferMap[kndx];
      }
      if ((type < 0) || (type > 2) || (part < 0) || (part > 7) || (rotn < 0) || (rotn > 5))
        ++bad;
    }
    std::cout << bad << " wafers of unknown types among " << hgdc.waferFileSize() << " wafers\n\n";

    // Now print out the summary
    static const std::vector<int> itype = {0, 1, 2};
    static const std::vector<int> itypc = {0, 1, 2, 3, 4, 5};
    static const std::vector<int> itypp = {0, 1, 2, 3, 4, 5, 6, 7};
    static const std::vector<std::string> typep = {"F", "b", "g", "gm", "a", "d", "dm", "c"};
    for (const auto& type : itype) {
      for (unsigned int k = 0; k < itypp.size(); ++k) {
        int part = itypp[k];
        for (const auto& rotn : itypc) {
          int kndx = ((rotn * 10 + part) * 10 + type);
          auto itr = waferMap.find(kndx);
          if (itr != waferMap.end())
            std::cout << "Type:" << type << " Partial:" << typep[k] << " Orientation:" << rotn << " with "
                      << (itr->second) << " wafers\n";
        }
      }
    }

    std::cout << "\n\nSummary of Cells\n================\n";
    for (const auto& type : itype) {
      int N = (type == 0) ? hgdc.getParameter()->nCellsFine_ : hgdc.getParameter()->nCellsCoarse_;
      for (unsigned int k = 0; k < itypp.size(); ++k) {
        int part = itypp[k];
        for (const auto& rotn : itypc) {
          int num(0);
          for (int cellU = 0; cellU < 2 * N; ++cellU) {
            for (int cellV = 0; cellV < 2 * N; ++cellV) {
              if (((cellV - cellU) < N) && (cellU - cellV) <= N) {
                if (HGCalWaferMask::goodCell(cellU, cellV, N, part, rotn))
                  ++num;
              }
            }
          }
          std::cout << "Type:" << type << " Partial:" << typep[k] << " Orientation:" << rotn << " with " << num
                    << " cells\n";
        }
      }
    }
    std::cout << "\n\n\n";
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferCell);
