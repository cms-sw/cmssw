// Checks the cell quantities of HGCalDDDConstants for each silicon wafer type:
// the high-density types get the fine cells, the low-density types the coarse ones.

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/transform.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

class HGCalCellSizeHexTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalCellSizeHexTester(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const std::vector<std::string> nameDetectors_;
  const std::vector<edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord>> tokens_;
};

HGCalCellSizeHexTester::HGCalCellSizeHexTester(const edm::ParameterSet& iC)
    : nameDetectors_(iC.getParameter<std::vector<std::string>>("nameDetectors")),
      tokens_{edm::vector_transform(nameDetectors_, [this](const std::string& name) {
        return esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", name});
      })} {}

void HGCalCellSizeHexTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("nameDetectors", {"HGCalEESensitive", "HGCalHESiliconSensitive"});
  descriptions.add("hgcalCellSizeHexTester", desc);
}

void HGCalCellSizeHexTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  constexpr double kTolerance = 1.e-6;      // cm
  constexpr double kToleranceArea = 1.e-3;  // cm2
  const std::vector<int> types = {HGCSiliconDetId::HGCalHD120,
                                  HGCSiliconDetId::HGCalLD200,
                                  HGCSiliconDetId::HGCalLD300,
                                  HGCSiliconDetId::HGCalHD200};
  unsigned int nFail(0);
  auto check = [&nFail](const std::string& name,
                        const char* what,
                        int type,
                        bool highDensity,
                        double value,
                        double expected,
                        double tolerance) {
    const bool ok = (std::abs(value - expected) < tolerance);
    if (!ok)
      ++nFail;
    edm::LogVerbatim("HGCalGeom") << name << " wafer type " << type
                                  << (highDensity ? " (high density) " : " (low density) ") << what << " " << value
                                  << " expected " << expected << (ok ? " OK" : " FAIL");
  };
  for (unsigned int k = 0; k < nameDetectors_.size(); ++k) {
    const HGCalDDDConstants& cons = iSetup.getData(tokens_[k]);
    if (!cons.waferHexagon8()) {
      edm::LogVerbatim("HGCalGeom") << nameDetectors_[k] << " does not use hexagon8 wafers: no check";
      continue;
    }
    const HGCalParameters* par = cons.getParameter();
    const double fine = 0.5 * par->waferSize_ / par->nCellsFine_;
    const double coarse = 0.5 * par->waferSize_ / par->nCellsCoarse_;
    for (const int type : types) {
      const bool highDensity = HGCSiliconDetId(DetId::HGCalEE, 1, type, 1, 0, 0, 0, 0).highDensity();
      const int nCells = highDensity ? par->nCellsFine_ : par->nCellsCoarse_;
      check(nameDetectors_[k],
            "cellSizeHex",
            type,
            highDensity,
            cons.cellSizeHex(type),
            highDensity ? fine : coarse,
            kTolerance);
      check(nameDetectors_[k], "getUVMax", type, highDensity, cons.getUVMax(type), nCells, kTolerance);
      // Central cell of a full wafer of this type. The wafer file describes the z = -1 wafers.
      auto itr = std::find_if(par->waferInfoMap_.begin(), par->waferInfoMap_.end(), [type](const auto& wafer) {
        return ((wafer.second.type == type) && (wafer.second.part == HGCalTypes::WaferFull));
      });
      if (itr != par->waferInfoMap_.end()) {
        const HGCSiliconDetId id((par->detectorType_ == 2) ? DetId::HGCalHSi : DetId::HGCalEE,
                                 -1,
                                 type,
                                 HGCalWaferIndex::waferLayer(itr->first),
                                 HGCalWaferIndex::waferU(itr->first),
                                 HGCalWaferIndex::waferV(itr->first),
                                 nCells,
                                 nCells);
        const double area = par->waferSize_ * par->waferSize_ / (2.0 * std::sqrt(3.0) * nCells * nCells);
        check(nameDetectors_[k], "cellArea", type, highDensity, cons.cellArea(id, true), area, kToleranceArea);
      }
    }
  }
  if (nFail > 0)
    throw cms::Exception("HGCalCellSizeHex") << nFail << " checks fail";
}

DEFINE_FWK_MODULE(HGCalCellSizeHexTester);
