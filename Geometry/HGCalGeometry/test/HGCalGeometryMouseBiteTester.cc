#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalMouseBite.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

class HGCalGeometryMouseBiteTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeometryMouseBiteTester(const edm::ParameterSet&);
  ~HGCalGeometryMouseBiteTester() override = default;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  const std::string nameSense_, nameDetector_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
};

HGCalGeometryMouseBiteTester::HGCalGeometryMouseBiteTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("NameSense")),
      nameDetector_(iC.getParameter<std::string>("NameDevice")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeomX") << "Test mousebite for " << nameDetector_ << " using constants of " << nameSense_
                                 << std::endl;
}

void HGCalGeometryMouseBiteTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  edm::LogVerbatim("HGCalGeomX") << nameDetector_ << " Layers = " << hgdc.layers(true)
                                 << " Sectors = " << hgdc.sectors();

  const auto bite = std::make_unique<HGCalMouseBite>(hgdc, false);
  ForwardSubdetector subdet(ForwardEmpty);
  DetId::Detector det;
  if (nameSense_ == "HGCalHESiliconSensitive") {
    det = DetId::HGCalHSi;
  } else if (nameSense_ == "HFNoseSensitive") {
    det = DetId::Forward;
    subdet = HFNose;
  } else {
    det = DetId::HGCalEE;
  }
  edm::LogVerbatim("HGCalGeomX") << "Perform test for " << nameSense_ << " Detector " << det << " SubDetector "
                                 << subdet;

  int zside(1), layer(1), waferU(1), waferV(1);
  int types[] = {0, 1};
  for (int type : types) {
    int ncell = (type == 0) ? HGCSiliconDetId::HGCalHighDensityN : HGCSiliconDetId::HGCalLowDensityN;
    edm::LogVerbatim("HGCalGeomX") << "zside " << zside << " layer " << layer << " wafer " << waferU << ":" << waferV
                                   << " type " << type << " cells " << ncell;
    for (int u = 0; u < 2 * ncell; ++u) {
      for (int v = 0; v < 2 * ncell; ++v) {
        if (((v - u) < ncell) && ((u - v) <= ncell)) {
          if (det == DetId::Forward) {
            HFNoseDetId id(zside, type, layer, waferU, waferV, u, v);
            edm::LogVerbatim("HGCalGeomX") << "ID: " << id << " with exclude flag " << bite->exclude(id);
          } else {
            HGCSiliconDetId id(det, zside, type, layer, waferU, waferV, u, v);
            edm::LogVerbatim("HGCalGeomX") << "ID: " << id << " with exclude flag " << bite->exclude(id);
          }
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryMouseBiteTester);
