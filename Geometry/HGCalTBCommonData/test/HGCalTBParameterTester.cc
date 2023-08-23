#include <array>
#include <iostream>
#include <sstream>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/HGCalTBCommonData/interface/HGCalTBParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalTBParameterTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalTBParameterTester(const edm::ParameterSet&);
  ~HGCalTBParameterTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  template <typename T>
  void myPrint(std::string const& s, std::vector<T> const& obj, int n) const;
  void myPrint(std::string const& s, std::vector<double> const& obj1, std::vector<double> const& obj2, int n) const;
  void printTrform(HGCalTBParameters const*) const;
  void printWaferType(HGCalTBParameters const* phgp) const;

  const std::string name_;
  edm::ESGetToken<HGCalTBParameters, IdealGeometryRecord> token_;
  const int mode_;
};

HGCalTBParameterTester::HGCalTBParameterTester(const edm::ParameterSet& ic)
    : name_(ic.getParameter<std::string>("name")),
      token_(esConsumes<HGCalTBParameters, IdealGeometryRecord>(edm::ESInputTag{"", name_})),
      mode_(ic.getParameter<int>("mode")) {}

void HGCalTBParameterTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "HGCalEESensitive");
  desc.add<int>("mode", 0);
  descriptions.add("hgcTBParameterTesterEE", desc);
}

void HGCalTBParameterTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("HGCalGeomr") << "HGCalTBParameter::Here I am";
  auto start = std::chrono::high_resolution_clock::now();

  const auto& hgp = iSetup.getData(token_);
  const auto* phgp = &hgp;

  edm::LogVerbatim("HGCalGeom") << phgp->name_;

  // Wafers of 6-inch format
  edm::LogVerbatim("HGCalGeom") << "DetectorType: " << phgp->detectorType_;
  edm::LogVerbatim("HGCalGeom") << "WaferR_: " << phgp->waferR_;
  edm::LogVerbatim("HGCalGeom") << "nCells_: " << phgp->nCells_;
  edm::LogVerbatim("HGCalGeom") << "nSectors_: " << phgp->nSectors_;
  edm::LogVerbatim("HGCalGeom") << "FirstLayer: " << phgp->firstLayer_;
  edm::LogVerbatim("HGCalGeom") << "mode_: " << phgp->mode_;

  myPrint("CellSize", phgp->cellSize_, 10);
  myPrint("slopeMin", phgp->slopeMin_, 10);
  myPrint("slopeTop", phgp->slopeTop_, 10);
  myPrint("zFrontTop", phgp->zFrontTop_, 10);
  myPrint("rMaxFront", phgp->rMaxFront_, 10);
  myPrint("zRanges", phgp->zRanges_, 10);
  myPrint("moduleBlS", phgp->moduleBlS_, 10);
  myPrint("moduleTlS", phgp->moduleTlS_, 10);
  myPrint("moduleHS", phgp->moduleHS_, 10);
  myPrint("moduleDzS", phgp->moduleDzS_, 10);
  myPrint("moduleAlphaS", phgp->moduleAlphaS_, 10);
  myPrint("moduleCellS", phgp->moduleCellS_, 10);
  myPrint("moduleBlR", phgp->moduleBlR_, 10);
  myPrint("moduleTlR", phgp->moduleTlR_, 10);
  myPrint("moduleHR", phgp->moduleHR_, 10);
  myPrint("moduleDzR", phgp->moduleDzR_, 10);
  myPrint("moduleAlphaR", phgp->moduleAlphaR_, 10);
  myPrint("moduleCellR", phgp->moduleCellR_, 10);
  myPrint("trformTranX", phgp->trformTranX_, 10);
  myPrint("trformTranY", phgp->trformTranY_, 10);
  myPrint("trformTranZ", phgp->trformTranZ_, 10);
  myPrint("trformRotXX", phgp->trformRotXX_, 10);
  myPrint("trformRotYX", phgp->trformRotYX_, 10);
  myPrint("trformRotZX", phgp->trformRotZX_, 10);
  myPrint("trformRotXY", phgp->trformRotXY_, 10);
  myPrint("trformRotYY", phgp->trformRotYY_, 10);
  myPrint("trformRotZY", phgp->trformRotZY_, 10);
  myPrint("trformRotXZ", phgp->trformRotXZ_, 10);
  myPrint("trformRotYZ", phgp->trformRotYZ_, 10);
  myPrint("trformRotZZ", phgp->trformRotZZ_, 10);
  myPrint("zLayerHex", phgp->zLayerHex_, 10);
  myPrint("rMinLayHex", phgp->rMinLayHex_, 10);
  myPrint("rMaxLayHex", phgp->rMaxLayHex_, 10);
  myPrint("waferPos", phgp->waferPosX_, phgp->waferPosY_, 4);
  myPrint("cellFine", phgp->cellFineX_, phgp->cellFineY_, 4);
  myPrint("cellFineHalf", phgp->cellFineHalf_, 10);
  myPrint("cellCoarse", phgp->cellCoarseX_, phgp->cellCoarseY_, 4);
  myPrint("cellCoarseHalf", phgp->cellCoarseHalf_, 10);
  myPrint("boundR", phgp->boundR_, 10);
  myPrint("moduleLayS", phgp->moduleLayS_, 10);
  myPrint("moduleLayR", phgp->moduleLayR_, 10);
  myPrint("layer", phgp->layer_, 18);
  myPrint("layerIndex", phgp->layerIndex_, 18);
  myPrint("layerGroup", phgp->layerGroup_, 18);
  myPrint("cellFactor", phgp->cellFactor_, 10);
  myPrint("depth", phgp->depth_, 18);
  myPrint("depthIndex", phgp->depthIndex_, 18);
  myPrint("depthLayerF", phgp->depthLayerF_, 18);
  myPrint("waferCopy", phgp->waferCopy_, 10);
  myPrint("waferTypeL", phgp->waferTypeL_, 25);
  myPrint("waferTypeT", phgp->waferTypeT_, 25);
  myPrint("layerGroupM", phgp->layerGroupM_, 18);
  myPrint("layerGroupO", phgp->layerGroupO_, 18);
  printTrform(phgp);
  myPrint("levelTop", phgp->levelT_, 10);
  printWaferType(phgp);

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  edm::LogVerbatim("HGCalGeom") << "Elapsed time: " << elapsed.count() << " s";
}

template <typename T>
void HGCalTBParameterTester::myPrint(std::string const& s, std::vector<T> const& obj, int n) const {
  int k(0), kk(0);
  edm::LogVerbatim("HGCalGeom") << s << " with " << obj.size() << " elements with n " << n << ": 1000";
  std::ostringstream st1[1000];
  for (auto const& it : obj) {
    st1[kk] << it << ", ";
    ++k;
    if (k == n) {
      edm::LogVerbatim("HGCalGeom") << st1[kk].str();
      ++kk;
      k = 0;
    }
  }
  if (k > 0)
    edm::LogVerbatim("HGCalGeom") << st1[kk].str();
}

void HGCalTBParameterTester::myPrint(std::string const& s,
                                     std::vector<double> const& obj1,
                                     std::vector<double> const& obj2,
                                     int n) const {
  int k(0), kk(0);
  std::ostringstream st1[250];
  edm::LogVerbatim("HGCalGeom") << s << " with " << obj1.size() << " elements with n " << n << ": 250";
  for (unsigned int k1 = 0; k1 < obj1.size(); ++k1) {
    st1[kk] << "(" << obj1[k1] << ", " << obj2[k1] << ") ";
    ++k;
    if (k == n) {
      edm::LogVerbatim("HGCalGeom") << st1[kk].str();
      ++kk;
      k = 0;
    }
  }
  if (k > 0)
    edm::LogVerbatim("HGCalGeom") << st1[kk].str();
}

void HGCalTBParameterTester::printTrform(HGCalTBParameters const* phgp) const {
  int k(0), kk(0);
  std::ostringstream st1[20];
  edm::LogVerbatim("HGCalGeom") << "TrformIndex with " << phgp->trformIndex_.size() << " elements with n 7:20";
  for (unsigned int i = 0; i < phgp->trformIndex_.size(); ++i) {
    std::array<int, 4> id = phgp->getID(i);
    st1[kk] << id[0] << ":" << id[1] << ":" << id[2] << ":" << id[3] << ", ";
    ++k;
    if (k == 7) {
      edm::LogVerbatim("HGCalGeom") << st1[kk].str();
      ++kk;
      k = 0;
    }
  }
  if (k > 0)
    edm::LogVerbatim("HGCalGeom") << st1[kk].str();
}

void HGCalTBParameterTester::printWaferType(HGCalTBParameters const* phgp) const {
  int k(0);
  edm::LogVerbatim("HGCalGeom") << "waferTypes with " << phgp->waferTypes_.size() << " elements";
  std::map<std::pair<int, int>, int> kounts;
  std::map<std::pair<int, int>, int>::iterator itr;
  for (auto const& it : phgp->waferTypes_) {
    std::ostringstream st1;
    st1 << " [" << k << "] " << HGCalWaferIndex::waferLayer(it.first);
    if (HGCalWaferIndex::waferFormat(it.first)) {
      st1 << ":" << HGCalWaferIndex::waferU(it.first) << ":" << HGCalWaferIndex::waferV(it.first);
    } else {
      st1 << ":" << HGCalWaferIndex::waferCopy(it.first);
    }
    edm::LogVerbatim("HGCalGeom") << st1.str() << " ==> (" << (it.second).first << ":" << (it.second).second << ")";
    itr = kounts.find(it.second);
    if (itr == kounts.end())
      kounts[it.second] = 1;
    else
      ++(itr->second);
    ++k;
  }
  if (!kounts.empty()) {
    edm::LogVerbatim("HGCalGeom") << "Summary of waferTypes ==========================";
    for (itr = kounts.begin(); itr != kounts.end(); ++itr)
      edm::LogVerbatim("HGCalGeom") << "Type (" << (itr->first).first << ":" << (itr->first).second << ") Kount "
                                    << itr->second;
  }
}

DEFINE_FWK_MODULE(HGCalTBParameterTester);
