// -*- C++ -*-
//
// Package:    HGCalScintIDTester
// Class:      HGCalScintIDTester
//
/**\class HGCalScintIDTester HGCalScintIDTester.cc
 test/HGCalScintIDTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2023/02/01
//
//

// system include files
#include <fstream>
#include <iostream>
#include <sstream>
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
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalScintIDTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalScintIDTester(const edm::ParameterSet&);
  ~HGCalScintIDTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_, errorFile_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
  std::vector<HGCScintillatorDetId> detIds_;
  std::vector<std::pair<double, double>> posXY_;
};

HGCalScintIDTester::HGCalScintIDTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("nameSense")),
      errorFile_(iC.getParameter<std::string>("fileName")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeom") << "Test HGCScintillator DetID for " << nameSense_ << " of positions from the file "
                                << errorFile_;

  edm::FileInPath filetmp("Geometry/HGCalCommonData/data/" + errorFile_);
  std::string fileName = filetmp.fullPath();
  std::ifstream fInput(fileName.c_str());
  if (!fInput.good()) {
    edm::LogVerbatim("HGCalSim") << "Cannot open file " << fileName;
  } else {
    char buffer[80];
    int kount(0);
    while (fInput.getline(buffer, 80)) {
      std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
      ++kount;
      if (items.size() > 8) {
        bool trig = (std::atoi(items[1].c_str()) > 0);
        int zp = std::atoi(items[2].c_str());
        int type = std::atoi(items[3].c_str());
        int sipm = std::atoi(items[4].c_str());
        int layer = std::atoi(items[5].c_str());
        int ring = std::atoi(items[6].c_str());
        int iphi = std::atoi(items[7].c_str());
        double xx = std::atof(items[8].c_str());
        double yy = std::atof(items[9].c_str());
        HGCScintillatorDetId id(type, layer, zp * ring, iphi, trig, sipm);
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

void HGCalScintIDTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameSense", "HGCalHEScintillatorSensitive");
  desc.add<std::string>("fileName", "errorScintD88.txt");
  descriptions.add("hgcalScintIDTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalScintIDTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  edm::LogVerbatim("HGCalGeom") << "\nStart testing " << nameSense_ << std::endl;

  for (unsigned int k = 0; k < detIds_.size(); ++k) {
    std::ostringstream st1;
    st1 << "Hit[" << k << "] " << detIds_[k];
    double xx = posXY_[k].first;
    double yy = posXY_[k].second;
    int layer = detIds_[k].layer();
    int zside = detIds_[k].zside();
    double zz = zside * hgdc.waferZ(layer, false);
    std::array<int, 3> idx = hgdc.assignCellTrap(xx, yy, zz, layer, false);
    HGCScintillatorDetId id(idx[2], layer, (zside * idx[0]), idx[1], false, 0);
    std::pair<int, int> typm = hgdc.tileType(layer, idx[0], 0);
    if (typm.first >= 0) {
      id.setType(typm.first);
      id.setSiPM(typm.second);
    }
    if (id.rawId() != detIds_[k].rawId())
      st1 << " non-matching DetId: new ID " << id;
    auto xy = hgdc.locateCell(id, false);
    double xx0 = (id.zside() > 0) ? xy.first : -xy.first;
    double yy0 = xy.second;
    double dx = xx0 - xx;
    double dy = yy0 - yy;
    double diff = std::sqrt(dx * dx + dy * dy);
    st1 << " input position: (" << xx << ", " << yy << "); position from ID (" << xx0 << ", " << yy0 << ") distance "
        << diff;
    constexpr double tol = 3.0;
    if (diff > tol)
      st1 << " ***** CheckID *****";
    bool valid1 = hgdc.isValidTrap(detIds_[k].zside(), detIds_[k].layer(), detIds_[k].ring(), detIds_[k].iphi());
    bool valid2 = hgdc.isValidTrap(id.zside(), id.layer(), id.ring(), id.iphi());
    st1 << " Validity flag: " << valid1 << ":" << valid2;
    if ((!valid1) || (!valid2))
      st1 << " +++++ Validity Check +++++ ";
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalScintIDTester);
