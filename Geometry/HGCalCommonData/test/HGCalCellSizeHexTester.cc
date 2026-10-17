// Checks HGCalDDDConstants::cellSizeHex for each silicon wafer type: the
// high-density types get the fine cell size, the low-density types the coarse one.

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
#include "Geometry/Records/interface/IdealGeometryRecord.h"

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
  constexpr double kTolerance = 1.e-6;  // cm
  const std::vector<int> types = {HGCSiliconDetId::HGCalHD120,
                                  HGCSiliconDetId::HGCalLD200,
                                  HGCSiliconDetId::HGCalLD300,
                                  HGCSiliconDetId::HGCalHD200};
  unsigned int nFail(0);
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
      const double expected = highDensity ? fine : coarse;
      const double size = cons.cellSizeHex(type);
      const bool ok = (std::abs(size - expected) < kTolerance);
      if (!ok)
        ++nFail;
      edm::LogVerbatim("HGCalGeom") << nameDetectors_[k] << " wafer type " << type
                                    << (highDensity ? " (high density)" : " (low density)") << " cellSizeHex " << size
                                    << " expected " << expected << (ok ? " OK" : " FAIL");
    }
  }
  if (nFail > 0)
    throw cms::Exception("HGCalCellSizeHex") << nFail << " wafer types get a wrong cellSizeHex";
}

DEFINE_FWK_MODULE(HGCalCellSizeHexTester);
