// system include files
#include <memory>
#include <string>
#include <vector>
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
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalPartialWaferTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalPartialWaferTester(const edm::ParameterSet&);
  ~HGCalPartialWaferTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_;
  const std::vector<int> orientations_, partialTypes_;
  const int nTrials_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
};

HGCalPartialWaferTester::HGCalPartialWaferTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("nameSense")),
      orientations_(iC.getParameter<std::vector<int>>("waferOrientations")),
      partialTypes_(iC.getParameter<std::vector<int>>("partialTypes")),
      nTrials_(iC.getParameter<int>("numberOfTrials")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeom") << "Test positions for " << partialTypes_.size() << " partial wafer types "
                                << " and " << orientations_.size() << " Orientations for " << nameSense_;
}

void HGCalPartialWaferTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<int> orients = {HGCalTypes::WaferOrient0,
                              HGCalTypes::WaferOrient1,
                              HGCalTypes::WaferOrient2,
                              HGCalTypes::WaferOrient3,
                              HGCalTypes::WaferOrient4,
                              HGCalTypes::WaferOrient5};
  std::vector<int> types = {HGCalTypes::WaferLDTop,
                            HGCalTypes::WaferLDBottom,
                            HGCalTypes::WaferLDLeft,
                            HGCalTypes::WaferLDRight,
                            HGCalTypes::WaferLDFive,
                            HGCalTypes::WaferLDThree,
                            HGCalTypes::WaferHDTop,
                            HGCalTypes::WaferHDBottom,
                            HGCalTypes::WaferHDLeft,
                            HGCalTypes::WaferHDRight,
                            HGCalTypes::WaferHDFive};
  desc.add<std::string>("nameSense", "HGCalHESiliconSensitive");
  desc.add<std::vector<int>>("waferOrientations", orients);
  desc.add<std::vector<int>>("partialTypes", types);
  desc.add<int>("numberOfTrials", 1000);
  descriptions.add("hgcalPartialWaferTester", desc);
}

void HGCalPartialWaferTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  const bool reco(true), all(true), norot(false), debug1(false), debug2(false), extend(true);
  for (const auto& partialType : partialTypes_) {
    for (const auto& orientation : orientations_) {
      int indx(-1), type(-1);
      for (auto itr = hgdc.getParameter()->waferInfoMap_.begin(); itr != hgdc.getParameter()->waferInfoMap_.end();
           ++itr) {
        if (((itr->second).part == partialType) && ((itr->second).orient == orientation)) {
          indx = itr->first;
          type = (itr->second).type;
          break;
        }
      }

      if (indx > 0) {
        int alltry(0), error(0);
        int layer = HGCalWaferIndex::waferLayer(indx);
        int waferU = HGCalWaferIndex::waferU(indx);
        int waferV = HGCalWaferIndex::waferV(indx);
        auto xy = hgdc.waferPosition(layer, waferU, waferV, true, false);
        edm::LogVerbatim("HGCalGeom") << "\n\nPartial Type " << partialType << " Orientation " << orientation
                                      << " Wafer " << waferU << ":" << waferV << " in layer " << layer << " at "
                                      << xy.first << ":" << xy.second << "\n\n";
        int nCells = (type == 0) ? HGCSiliconDetId::HGCalFineN : HGCSiliconDetId::HGCalCoarseN;
        for (int i = 0; i < nTrials_; i++) {
          int ui = std::floor(nCells * 0.0002 * (rand() % 10000));
          int vi = std::floor(nCells * 0.0002 * (rand() % 10000));
          if ((ui < 2 * nCells) && (vi < 2 * nCells) && ((vi - ui) < nCells) && ((ui - vi) <= nCells) &&
              HGCalWaferMask::goodCell(ui, vi, partialType)) {
            ++alltry;
            double zpos = hgdc.waferZ(layer, reco);
            int zside = (zpos > 0) ? 1 : -1;
            auto xy = hgdc.locateCell(zside, layer, waferU, waferV, ui, vi, reco, all, norot, debug1);
            int lay(layer), cU(0), cV(0), wType(-1), wU(0), wV(0);
            double wt(0);
            hgdc.waferFromPosition(HGCalParameters::k_ScaleToDDD * xy.first,
                                   HGCalParameters::k_ScaleToDDD * xy.second,
                                   zside,
                                   lay,
                                   wU,
                                   wV,
                                   cU,
                                   cV,
                                   wType,
                                   wt,
                                   extend,
                                   debug2);
            bool ok =
                ((wType == type) && (layer == lay) && (waferU == wU) && (waferV == wV) && (ui == cU) && (vi == cV));
            std::string comment = (ok) ? "" : " ***** ERROR *****";
            edm::LogVerbatim("HGCalGeom")
                << "Layer " << layer << ":" << lay << " waferU " << waferU << ":" << wU << " waferV " << waferV << ":"
                << wV << " Type " << type << ":" << wType << " cellU " << ui << ":" << cU << " cellV " << vi << ":"
                << cV << " position " << xy.first << ":" << xy.second << comment;
            if (!ok)
              ++error;
          }
        }
        edm::LogVerbatim("HGCalGeom") << "\n\nFound " << error << " errors among " << alltry << ":" << nTrials_
                                      << " trials for partial type " << partialType << " orientation " << orientation
                                      << " of " << nameSense_;
      } else {
        edm::LogVerbatim("HGCalGeom") << "\n\nCannot find a wafer of type " << partialType << " and orientation "
                                      << orientation << " for " << nameSense_;
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalPartialWaferTester);
