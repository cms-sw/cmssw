// system include files
#include <array>
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

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalTBNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalTBNumberingTester(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ESGetToken<HGCalTBDDDConstants, IdealGeometryRecord> dddToken_;
  std::string nameSense_, nameDetector_;
  std::vector<double> positionX_, positionY_;
  int increment_;
};

HGCalTBNumberingTester::HGCalTBNumberingTester(const edm::ParameterSet& iC) {
  nameSense_ = iC.getParameter<std::string>("nameSense");
  nameDetector_ = iC.getParameter<std::string>("nameDevice");
  positionX_ = iC.getParameter<std::vector<double> >("localPositionX");
  positionY_ = iC.getParameter<std::vector<double> >("localPositionY");
  increment_ = iC.getParameter<int>("increment");

  dddToken_ = esConsumes<HGCalTBDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_});

  std::string unit("mm");
  for (unsigned int k = 0; k < positionX_.size(); ++k) {
    positionX_[k] /= CLHEP::mm;
    positionY_[k] /= CLHEP::mm;
  }
  edm::LogVerbatim("HGCalGeom") << "Test numbering for " << nameDetector_ << " using constants of " << nameSense_
                                << " at " << positionX_.size() << " local positions "
                                << "for every " << increment_ << " layers";
  for (unsigned int k = 0; k < positionX_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Position[" << k << "] " << positionX_[k] << " " << unit << ", " << positionY_[k]
                                  << " " << unit;
}

void HGCalTBNumberingTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::vector<double> vecxy;
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<std::string>("nameDevice", "HGCal EE");
  desc.add<std::vector<double> >("localPositionX", vecxy);
  desc.add<std::vector<double> >("localPositionY", vecxy);
  desc.add<int>("increment", 2);
  descriptions.add("hgcalTBNumberingTesterEE", desc);
}

// ------------ method called to produce the data  ------------
void HGCalTBNumberingTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HGCalTBDDDConstants& hgdc = iSetup.getData(dddToken_);
  edm::LogVerbatim("HGCalGeom") << nameDetector_ << " Layers = " << hgdc.layers(false)
                                << " Sectors = " << hgdc.sectors() << " Minimum Slope = " << hgdc.minSlope();

  edm::LogVerbatim("HGCalGeom") << "Minimum Wafer # " << hgdc.waferMin() << " Mamximum Wafer # " << hgdc.waferMax()
                                << " Wafer counts " << hgdc.waferCount(0) << ":" << hgdc.waferCount(1);
  for (unsigned int i = 0; i < hgdc.layers(true); ++i) {
    int lay = i + 1;
    double z = hgdc.waferZ(lay, false);
    edm::LogVerbatim("HGCalGeom") << "Layer " << lay << " Wafers " << hgdc.wafers(lay, 0) << ":" << hgdc.wafers(lay, 1)
                                  << ":" << hgdc.wafers(lay, 2) << " Z " << z << " R " << hgdc.rangeR(z, false).first
                                  << ":" << hgdc.rangeR(z, false).second;
  }

  edm::LogVerbatim("HGCalGeom") << std::endl;
  std::pair<float, float> xy;
  std::string flg;
  int subsec(0);
  int loff = hgdc.firstLayer();
  for (unsigned int k = 0; k < positionX_.size(); ++k) {
    float localx(positionX_[k]), localy(positionY_[k]);
    for (unsigned int i = 0; i < hgdc.layers(false); ++i) {
      std::pair<int, int> kxy, lxy;
      kxy = hgdc.assignCell(localx, localy, i + loff, subsec, false);
      xy = hgdc.locateCell(kxy.second, i + loff, kxy.first, false);
      lxy = hgdc.assignCell(xy.first, xy.second, i + loff, 0, false);
      flg = (kxy == lxy) ? " " : " ***** Error *****";
      edm::LogVerbatim("HGCalGeom") << "Input: (" << localx << "," << localy << "," << i + loff << ", " << subsec
                                    << "), assignCell o/p (" << kxy.first << ", " << kxy.second << ") locateCell o/p ("
                                    << xy.first << ", " << xy.second << "),"
                                    << " final (" << lxy.first << ", " << lxy.second << ")" << flg;
      kxy = hgdc.assignCell(-localx, -localy, i + loff, subsec, false);
      xy = hgdc.locateCell(kxy.second, i + loff, kxy.first, false);
      lxy = hgdc.assignCell(xy.first, xy.second, i + loff, 0, false);
      flg = (kxy == lxy) ? " " : " ***** Error *****";
      edm::LogVerbatim("HGCalGeom") << "Input: (" << -localx << "," << -localy << "," << i + loff << ", " << subsec
                                    << "), assignCell o/p (" << kxy.first << ", " << kxy.second << ") locateCell o/p ("
                                    << xy.first << ", " << xy.second << "), final (" << lxy.first << ", " << lxy.second
                                    << ")" << flg;

      if (k == 0 && i == 0) {
        std::vector<int> ncells = hgdc.numberCells(i + 1, false);
        edm::LogVerbatim("HGCalGeom") << "Layer " << i + 1 << " with " << ncells.size() << " rows";
        int ntot(0);
        for (unsigned int k = 0; k < ncells.size(); ++k) {
          ntot += ncells[k];
          edm::LogVerbatim("HGCalGeom") << "Row " << k << " with " << ncells[k] << " cells";
        }
        edm::LogVerbatim("HGCalGeom") << "Total Cells " << ntot << ":" << hgdc.maxCells(i + 1, false);
      }
      i += increment_;
    }
  }

  // Test the range variables
  unsigned int kk(0);
  for (auto const& zz : hgdc.getParameter()->zLayerHex_) {
    std::pair<double, double> rr = hgdc.rangeR(zz, true);
    edm::LogVerbatim("HGCalGeom") << "[" << kk << "]\t z = " << zz << "\t rMin = " << rr.first
                                  << "\t rMax = " << rr.second;
    ++kk;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalTBNumberingTester);
