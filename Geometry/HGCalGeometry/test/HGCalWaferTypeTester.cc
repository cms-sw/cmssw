// -*- C++ -*-
//
// Package:    HGCalWaferTypeTester
// Class:      HGCalWaferTypeTester
//
/**\class HGCalWaferTypeTester HGCalWaferTypeTester.cc
 test/HGCalWaferTypeTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2020/06/08
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

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferTypeTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferTypeTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_, nameDetector_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalWaferTypeTester::HGCalWaferTypeTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("NameSense")),
      nameDetector_(iC.getParameter<std::string>("NameDevice")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeomX") << "Test wafer types for " << nameDetector_ << " using constants of " << nameSense_
                                 << " for  RecoFlag true";
}

void HGCalWaferTypeTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("NameSense", "HGCalEESensitive");
  desc.add<std::string>("NameDevice", "HGCal EE");
  descriptions.add("hgcalEEWaferTypeTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalWaferTypeTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  const HGCalDDDConstants& hgdc = geom->topology().dddConstants();
  HGCalGeometryMode::GeometryMode mode = hgdc.geomMode();
  edm::LogVerbatim("HGCalGeomX") << nameDetector_ << "\n Mode = " << mode;
  if (hgdc.waferHexagon8()) {
    double r = hgdc.waferParameters(true).first;
    double R = hgdc.waferParameters(true).second;
    edm::LogVerbatim("HGCalGeomX") << "Wafer Parameters " << r << ":" << R << std::endl;
    // Determine if the 24 points on the perphery of the wafer is within range
    // These are the 6 corners; the middle of the edges and the positions
    // which determines the positions of choptwoMinus and semiMinus
    // Offset of these points wrt the center of the wafer is given in dx, dy
    static const unsigned int nc = 24;
    double dx[nc] = {0.0, 0.25 * r,  0.50 * r,  0.75 * r,  r,  r,  r,  r,  r,  0.75 * r,  0.50 * r,  0.25 * r,
                     0.0, -0.25 * r, -0.50 * r, -0.75 * r, -r, -r, -r, -r, -r, -0.75 * r, -0.50 * r, -0.25 * r};
    double dy[nc] = {-R,       -0.875 * R, -0.75 * R, -0.625 * R, -0.50 * R, -0.25 * R,  0.0,       0.25 * R,
                     0.50 * R, 0.625 * R,  0.75 * R,  0.875 * R,  R,         0.875 * R,  0.75 * R,  0.625 * R,
                     0.50 * R, 0.25 * R,   0.0,       -0.25 * R,  -0.50 * R, -0.625 * R, -0.75 * R, -0.875 * R};
    // There are 43 valid patterns corresponding to 7 types of partial wafers
    // in 6 orientation and one full wafer. These are are minmum requirerement
    // for these patterns
    static const unsigned int np = 43;
    unsigned int pat[np] = {0xFFFF01, 0xFFF01F, 0xFF01FF, 0xF01FFF, 0x01FFFF, 0x1FFFF0, 0xFFFC07, 0xFFC07F, 0xFC07FF,
                            0xC07FFF, 0x07FFFC, 0x7FFFC0, 0xFFF803, 0xFF803F, 0xF803FF, 0x803FFF, 0x03FFF8, 0x3FFF80,
                            0xFFF001, 0xFF001F, 0xF001FF, 0x001FFF, 0x01FFF0, 0x1FFF00, 0xFFC007, 0xFC007F, 0xC007FF,
                            0x007FFC, 0x07FFC0, 0x7FFC00, 0xFF8003, 0xF8003F, 0x8003FF, 0x003FF8, 0x03FF80, 0x3FF800,
                            0xFF0001, 0xF0001F, 0x0001FF, 0x001FF0, 0x01FF00, 0x1FF000, 0xFFFFFF};
    const std::vector<int> partTypeNew = {HGCalTypes::WaferLDBottom,
                                          HGCalTypes::WaferLDLeft,
                                          HGCalTypes::WaferLDRight,
                                          HGCalTypes::WaferLDFive,
                                          HGCalTypes::WaferLDThree,
                                          HGCalTypes::WaferHDTop,
                                          HGCalTypes::WaferHDBottom,
                                          HGCalTypes::WaferHDLeft,
                                          HGCalTypes::WaferHDRight,
                                          HGCalTypes::WaferHDFive};
    const std::vector<int> partTypeOld = {HGCalTypes::WaferHalf,
                                          HGCalTypes::WaferHalf,
                                          HGCalTypes::WaferSemi,
                                          HGCalTypes::WaferSemi,
                                          HGCalTypes::WaferFive,
                                          HGCalTypes::WaferThree,
                                          HGCalTypes::WaferHalf2,
                                          HGCalTypes::WaferChopTwoM,
                                          HGCalTypes::WaferSemi2,
                                          HGCalTypes::WaferSemi2,
                                          HGCalTypes::WaferFive2};
    const std::vector<DetId>& ids = geom->getValidGeomDetIds();
    int all(0), total(0), good(0), bad(0);
    for (auto id : ids) {
      HGCSiliconDetId hid(id);
      auto type = hgdc.waferTypeRotation(hid.layer(), hid.waferU(), hid.waferV(), false, false);
      if (hid.zside() > 0)
        ++all;
      // Not a full wafer
      int part = type.first;
      if (part > 10) {
        auto itr = std::find(partTypeNew.begin(), partTypeNew.end(), part);
        if (itr != partTypeNew.end()) {
          unsigned int indx = static_cast<unsigned int>(itr - partTypeNew.begin());
          part = partTypeOld[indx];
        }
      }
      if (part > 0 && part < 10 && hid.zside() > 0) {
        ++total;
        int wtype = hgdc.waferType(hid.layer(), hid.waferU(), hid.waferV(), false);
        int indx = (part - 1) * 6 + type.second;
        GlobalPoint xyz = geom->getWaferPosition(id);
        auto range = hgdc.rangeRLayer(hid.layer(), true);
        unsigned int ipat(0), ii(1);
        for (unsigned int i = 0; i < nc; ++i) {
          double rp = std::sqrt((xyz.x() + dx[i]) * (xyz.x() + dx[i]) + (xyz.y() + dy[i]) * (xyz.y() + dy[i]));
          if ((rp >= range.first) && (rp <= range.second))
            ipat += ii;
          ii *= 2;
        }
        bool match = (ipat == pat[indx]);
        if (!match) {
          // Make sure the minimum requirement is satisfied
          ii = 1;
          match = true;
          for (unsigned int i = 0; i < nc; ++i) {
            if ((((pat[indx] / ii) & 1) != 0) && (((ipat / ii) & 1) == 0)) {
              edm::LogVerbatim("HGCalGeomX") << "Fail at " << i << ":" << ii << " Expect " << ((pat[indx] / ii) & 1)
                                             << " Found " << ((ipat / ii) & 1);
              match = false;
              break;
            }
            ii *= 2;
          }
          if (match) {
            // and it doe not satify the higher ups
            if (wtype == 0) {
              match = (static_cast<unsigned int>(std::find(pat, pat + np, ipat) - pat) >= np);
            } else {
              // for coarse wafers the "minus" types are not allowed
              for (unsigned int i = 0; i < np; ++i) {
                if (i < 12 || (i >= 18 && i < 30) || (i >= 36)) {
                  if (ipat == pat[i]) {
                    match = false;
                    break;
                  }
                }
              }
            }
          }
        }
        std::string cherr = (!match) ? " ***** ERROR *****" : "";
        edm::LogVerbatim("HGCalGeomX") << "Wafer[" << wtype << ", " << hid.layer() << ", " << hid.waferU() << ", "
                                       << hid.waferV() << "]  with type: rotation " << type.first << "(" << part
                                       << "):" << type.second << " Pattern " << std::hex << pat[indx] << ":" << ipat
                                       << std::dec << cherr;
        if (!match) {
          ++bad;
          // Need debug information here
          hgdc.waferTypeRotation(hid.layer(), hid.waferU(), hid.waferV(), true, false);
          HGCalWaferMask::getTypeMode(xyz.x(), xyz.y(), r, R, range.first, range.second, wtype, 0, true);
          for (unsigned int i = 0; i < 24; ++i) {
            double rp = std::sqrt((xyz.x() + dx[i]) * (xyz.x() + dx[i]) + (xyz.y() + dy[i]) * (xyz.y() + dy[i]));
            edm::LogVerbatim("HGCalGeomX")
                << "Corner[" << i << "] (" << xyz.x() << ":" << xyz.y() << ") (" << (xyz.x() + dx[i]) << ":"
                << (xyz.y() + dy[i]) << " ) R " << rp << " Limit " << range.first << ":" << range.second;
          }
        } else {
          ++good;
        }
        ++total;
      }
    }
    edm::LogVerbatim("HGCalGeomX") << "\n\nExamined " << ids.size() << ":" << all << " wafers " << total
                                   << " partial wafers of which " << good << " are good and " << bad << " are bad\n";
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferTypeTester);
