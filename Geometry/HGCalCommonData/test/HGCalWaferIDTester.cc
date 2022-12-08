// -*- C++ -*-
//
// Package:    HGCalWaferIDTester
// Class:      HGCalWaferIDTester
//
/**\class HGCalWaferIDTester HGCalWaferIDTester.cc
 test/HGCalWaferIDTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2022/12/01
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
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferIDTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferIDTester(const edm::ParameterSet&);
  ~HGCalWaferIDTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_, errorFile_;
  const int mode_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
  std::vector<HGCSiliconDetId> detIds_;
  std::vector<std::pair<double, double>> posXY_;
};

HGCalWaferIDTester::HGCalWaferIDTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("nameSense")),
      errorFile_(iC.getParameter<std::string>("fileName")),
      mode_(iC.getParameter<int>("mode")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeom") << "Test HGCSilicon DetID for " << nameSense_ << " of positions from the file "
                                << errorFile_ << " for the mode " << mode_;

  edm::FileInPath filetmp("Geometry/HGCalCommonData/data/" + errorFile_);
  std::string fileName = filetmp.fullPath();
  std::ifstream fInput(fileName.c_str());
  if (!fInput.good()) {
    edm::LogVerbatim("HGCalSim") << "Cannot open file " << fileName;
  } else {
    char buffer[80];
    DetId::Detector det = (nameSense_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi;
    int kount(0);
    while (fInput.getline(buffer, 80)) {
      std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
      ++kount;
      if (items.size() > 8) {
        int type = std::atoi(items[0].c_str());
        int zp = std::atoi(items[1].c_str());
        int layer = std::atoi(items[2].c_str());
        int waferU = std::atoi(items[3].c_str());
        int waferV = std::atoi(items[4].c_str());
        int cellU = std::atoi(items[5].c_str());
        int cellV = std::atoi(items[6].c_str());
        double xx = std::atof(items[7].c_str()) * CLHEP::cm;
        double yy = std::atof(items[8].c_str()) * CLHEP::cm;
        HGCSiliconDetId id(det, zp, type, layer, waferU, waferV, cellU, cellV);
        detIds_.emplace_back(id);
        posXY_.emplace_back(std::make_pair(xx, yy));
      }
    }
    fInput.close();
    edm::LogVerbatim("HGCalGeom") << "Reads a total of " << detIds_.size() << ":" << posXY_.size() << " entries out of "
                                  << kount << "\n";
    for (unsigned int k = 0; k < detIds_.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << detIds_[k] << " (" << posXY_[k].first << ", "
                                    << posXY_[k].second << ")";
  }
}

void HGCalWaferIDTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameSense", "HGCalHESiliconSensitive");
  desc.add<std::string>("fileName", "cellIDHEF.txt");
  desc.add<int>("mode", 1);
  descriptions.add("hgcalWaferIDTesterHEF", desc);
}

// ------------ method called to produce the data  ------------
void HGCalWaferIDTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  edm::LogVerbatim("HGCalGeom") << "\nStart testing " << nameSense_ << " for mode " << mode_ << std::endl;

  bool debug = (mode_ > 0) ? true : false;
  for (unsigned int k = 0; k < detIds_.size(); ++k) {
    int cellU(0), cellV(0), waferType(-1), waferU(0), waferV(0);
    double wt(0);
    double xx = posXY_[k].first;
    double yy = posXY_[k].second;
    int layer = detIds_[k].layer();
    hgdc.waferFromPosition(xx, yy, layer, waferU, waferV, cellU, cellV, waferType, wt, false, debug);
    HGCSiliconDetId id(detIds_[k].det(), detIds_[k].zside(), waferType, layer, waferU, waferV, cellU, cellV);
    if (id.rawId() != detIds_[k].rawId())
      edm::LogVerbatim("HGCalGeom") << "Non-matching DetId for [" << k << "] Original " << detIds_[k] << " New " << id;
    auto xy = hgdc.locateCell(id, false);
    double xx0 = (id.zside() > 0) ? xy.first : -xy.first;
    double yy0 = xy.second;
    double dx = xx0 - (xx / CLHEP::cm);
    double dy = yy0 - (yy / CLHEP::cm);
    double diff = std::sqrt(dx * dx + dy * dy);
    constexpr double tol = 1.0;
    if (diff > tol)
      edm::LogVerbatim("HGCalGeom") << "CheckID " << id << " input position: (" << xx / CLHEP::cm << ", "
                                    << yy / CLHEP::cm << "); position from ID (" << xx0 << ", " << yy0 << ") distance "
                                    << diff;
    bool valid1 = hgdc.isValidHex8(
        detIds_[k].layer(), detIds_[k].waferU(), detIds_[k].waferV(), detIds_[k].cellU(), detIds_[k].cellV(), true);
    bool valid2 = hgdc.isValidHex8(id.layer(), id.waferU(), id.waferV(), id.cellU(), id.cellV(), true);
    if ((!valid1) || (!valid2))
      edm::LogVerbatim("HGCalGeom") << "Original flag: " << valid1 << " New flag: " << valid2;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferIDTester);
