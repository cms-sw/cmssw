// -*- C++ -*-
//
// Package:    HGCalWaferInFileCheck
// Class:      HGCalWaferInFileCheck
//
/**\class HGCalWaferInFileCheck HGCalWaferInFileCheck.cc
 test/HGCalWaferInFileCheck.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2020/06/24
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
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferInFileCheck : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferInFileCheck(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_, nameDetector_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalWaferInFileCheck::HGCalWaferInFileCheck(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("NameSense")),
      nameDetector_(iC.getParameter<std::string>("NameDevice")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeom") << "Test numbering for " << nameDetector_ << " using constants of " << nameSense_;
}

void HGCalWaferInFileCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("NameSense", "HGCalEESensitive");
  desc.add<std::string>("NameDevice", "HGCal EE");
  descriptions.add("hgcalEEWaferInFileCheck", desc);
}

// ------------ method called to produce the data  ------------
void HGCalWaferInFileCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  const auto& hgdc = geom->topology().dddConstants();

  edm::LogVerbatim("HGCalGeom") << nameDetector_ << "\nCheck Wafers in file are all valid for " << nameDetector_
                                << "\n";
  if (hgdc.waferHexagon8()) {
    DetId::Detector det = (nameSense_ == "HGCalHESiliconSensitive") ? DetId::HGCalHSi : DetId::HGCalEE;
    edm::LogVerbatim("HGCalGeom")
        << "Old Partial types: 1|Five, 2|ChopTwo, 3|ChopTwoM, 4|Half, 5|Semi, 6|Semi2, 7|Three, 8|Half2, 9|Five2\nNew "
           "Partial types: 11|LDTop, 12|LDBottom, 13|LDLeft, 14|LDRight, 15|LDFive, 16|LDThree, 21|HDTop, 22|HDBottom, "
           "23|HDLeft, 24|HDRight, 25|HDFive\nCommon Partial types: 0|Full, 99|Out";
    // See if all entries in the file are valid
    int bad1(0);
    for (unsigned int k = 0; k < hgdc.waferFileSize(); ++k) {
      int indx = hgdc.waferFileIndex(k);
      int layer = HGCalWaferIndex::waferLayer(indx);
      int waferU = HGCalWaferIndex::waferU(indx);
      int waferV = HGCalWaferIndex::waferV(indx);
      int type = std::get<0>(hgdc.waferFileInfo(k));
      HGCSiliconDetId id(det, 1, type, layer, waferU, waferV, 0, 0);
      if (!geom->topology().validModule(id, 3)) {
        int part = std::get<1>(hgdc.waferFileInfoFromIndex(indx));
        const auto& xy = hgdc.waferPosition(layer, waferU, waferV, true, false);
        edm::LogVerbatim("HGCalGeom") << "ID[" << k << "]: (" << (hgdc.getLayerOffset() + layer) << ", " << waferU
                                      << ", " << waferV << ", " << part << ") at (" << std::setprecision(4) << xy.first
                                      << ", " << xy.second << ", " << hgdc.waferZ(layer, true) << ") not valid";
        ++bad1;
      }
    }
    edm::LogVerbatim("HGCalGeom") << "\n\nFinds " << bad1 << " invalid wafers among " << hgdc.waferFileSize()
                                  << " wafers in the list\n";

    // See if some of the valid wafers are missing
    auto const& ids = geom->getValidGeomDetIds();
    int all(0), bad2(0), xtra(0);
    for (unsigned int k = 0; k < ids.size(); ++k) {
      HGCSiliconDetId id(ids[k]);
      if ((ids[k].rawId() != 0) && (id.zside() == 1)) {
        ++all;
        int indx = HGCalWaferIndex::waferIndex(id.layer(), id.waferU(), id.waferV());
        if (!hgdc.waferFileInfoExist(indx)) {
          int part = hgdc.waferTypeRotation(id.layer(), id.waferU(), id.waferV(), false, false).first;
          if (part != HGCalTypes::WaferOut) {
            const auto& xy = hgdc.waferPosition(id.layer(), id.waferU(), id.waferV(), true, false);
            edm::LogVerbatim("HGCalGeom")
                << "ID[" << k << "]: (" << (hgdc.getLayerOffset() + id.layer()) << ", " << id.waferU() << ", "
                << id.waferV() << ", " << part << ")  at (" << std::setprecision(4) << xy.first << ", " << xy.second
                << ", " << hgdc.waferZ(id.layer(), true) << ") not in wafer-list";
            ++bad2;
          } else {
            ++xtra;
          }
        }
      }
    }
    edm::LogVerbatim("HGCalGeom") << "\n\nFinds " << bad2 << " missing wafers among " << all << " valid wafers and "
                                  << xtra << " extra ones\n";

    // Now cross check the content
    int allG(0), badT(0), badP(0), badP2(0), badR(0), badG(0), badT1(0), badT2(0);
    for (unsigned int k = 0; k < hgdc.waferFileSize(); ++k) {
      int indx = hgdc.waferFileIndex(k);
      int type1 = std::get<0>(hgdc.waferFileInfo(k));
      int part1 = std::get<1>(hgdc.waferFileInfo(k));
      int rotn1 = std::get<2>(hgdc.waferFileInfo(k));
      int layer = HGCalWaferIndex::waferLayer(indx);
      int waferU = HGCalWaferIndex::waferU(indx);
      int waferV = HGCalWaferIndex::waferV(indx);
      int type2 = hgdc.waferType(layer, waferU, waferV, false);
      HGCSiliconDetId id(det, 1, type2, layer, waferU, waferV, 0, 0);
      if (geom->topology().validModule(id, 3)) {
        ++allG;
        int part2 = hgdc.waferTypeRotation(id.layer(), id.waferU(), id.waferV(), false, false).first;
        int rotn2 = hgdc.waferTypeRotation(id.layer(), id.waferU(), id.waferV(), false, false).second;
        bool typeOK = (type1 == type2);
        bool partOK = ((part1 == part2) || ((part1 == HGCalTypes::WaferFull) && (part2 == HGCalTypes::WaferOut)) ||
                       ((part2 == HGCalTypes::WaferFull) && (part1 != HGCalTypes::WaferOut)));
        bool rotnOK = ((rotn1 == rotn2) || (part1 == HGCalTypes::WaferFull) || (part2 == HGCalTypes::WaferFull));
        if (part1 < part2)
          ++badP2;
        if (!typeOK) {
          ++badT;
          if (type1 == 0)
            ++badT1;
          else if (type2 == 0)
            ++badT2;
        }
        if (!partOK)
          ++badP;
        if (!rotnOK)
          ++badR;
        if ((!typeOK) || (!partOK) || (!rotnOK)) {
          ++badG;
          const auto& xy = hgdc.waferPosition(layer, waferU, waferV, true, false);
          edm::LogVerbatim("HGCalGeom") << "ID[" << k << "]: (" << (hgdc.getLayerOffset() + layer) << ", " << waferU
                                        << ", " << waferV << ", " << type1 << ":" << type2 << ", " << part1 << ":"
                                        << part2 << ", " << rotn1 << ":" << rotn2 << ") at (" << std::setprecision(4)
                                        << xy.first << ", " << xy.second << ", " << hgdc.waferZ(layer, true)
                                        << ") failure flag " << typeOK << ":" << partOK << ":" << rotnOK << ":"
                                        << (part1 >= part2);
        }
      }
    }
    edm::LogVerbatim("HGCalGeom") << "\n\nFinds " << badG << " (" << badT << "[" << badT1 << ":" << badT2
                                  << "]:" << badP << ":" << badP2 << ":" << badR << ") mismatch among " << allG
                                  << " wafers with the same indices";
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferInFileCheck);
