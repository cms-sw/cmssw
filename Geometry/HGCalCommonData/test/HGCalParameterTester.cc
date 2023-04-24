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

#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalParameterTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalParameterTester(const edm::ParameterSet&);
  ~HGCalParameterTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  template <typename T>
  void myPrint(std::string const& s, std::vector<T> const& obj, int n) const;
  template <typename T>
  void myPrint(std::string const& s, std::vector<std::pair<T, T> > const& obj, int n) const;
  void myPrint(std::string const& s, std::vector<double> const& obj1, std::vector<double> const& obj2, int n) const;
  void myPrint(std::string const& s, HGCalParameters::wafer_map const& obj, int n) const;
  void printTrform(HGCalParameters const*) const;
  void printWaferType(HGCalParameters const* phgp) const;

  const std::string name_;
  edm::ESGetToken<HGCalParameters, IdealGeometryRecord> token_;
  const int mode_;
};

HGCalParameterTester::HGCalParameterTester(const edm::ParameterSet& ic)
    : name_(ic.getParameter<std::string>("Name")),
      token_(esConsumes<HGCalParameters, IdealGeometryRecord>(edm::ESInputTag{"", name_})),
      mode_(ic.getParameter<int>("Mode")) {}

void HGCalParameterTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("Name", "HGCalEESensitive");
  desc.add<int>("Mode", 1);
  descriptions.add("hgcParameterTesterEE", desc);
}

void HGCalParameterTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("HGCalGeomr") << "HGCalParameter::Here I am";
  auto start = std::chrono::high_resolution_clock::now();

  const auto& hgp = iSetup.getData(token_);
  const auto* phgp = &hgp;

  edm::LogVerbatim("HGCalGeom") << phgp->name_;
  if (mode_ == 0) {
    // Wafers of 6-inch format
    edm::LogVerbatim("HGCalGeom") << "DetectorType: " << phgp->detectorType_;
    edm::LogVerbatim("HGCalGeom") << "WaferR_: " << phgp->waferR_;
    edm::LogVerbatim("HGCalGeom") << "nCells_: " << phgp->nCells_;
    edm::LogVerbatim("HGCalGeom") << "nSectors_: " << phgp->nSectors_;
    edm::LogVerbatim("HGCalGeom") << "FirstLayer: " << phgp->firstLayer_;
    edm::LogVerbatim("HGCalGeom") << "FirstMixedLayer: " << phgp->firstMixedLayer_;
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

  } else if (mode_ == 1) {
    // Wafers of 8-inch format
    edm::LogVerbatim("HGCalGeom") << "DetectorType: " << phgp->detectorType_;
    edm::LogVerbatim("HGCalGeom") << "UseSimWt: " << phgp->useSimWt_;
    edm::LogVerbatim("HGCalGeom") << "Wafer Parameters: " << phgp->waferSize_ << ":" << phgp->waferR_ << ":"
                                  << phgp->waferThick_ << ":" << phgp->sensorSeparation_ << ":"
                                  << phgp->sensorSizeOffset_ << ":" << phgp->guardRingOffset_ << ":" << phgp->mouseBite_
                                  << ":" << phgp->useOffset_;
    myPrint("waferThickness", phgp->waferThickness_, 10);
    edm::LogVerbatim("HGCalGeom") << "nCells_: " << phgp->nCellsFine_ << ":" << phgp->nCellsCoarse_;
    edm::LogVerbatim("HGCalGeom") << "nSectors_: " << phgp->nSectors_;
    edm::LogVerbatim("HGCalGeom") << "FirstLayer: " << phgp->firstLayer_;
    edm::LogVerbatim("HGCalGeom") << "FirstMixedLayer: " << phgp->firstMixedLayer_;
    edm::LogVerbatim("HGCalGeom") << "LayerOffset: " << phgp->layerOffset_;
    edm::LogVerbatim("HGCalGeom") << "mode_: " << phgp->mode_;
    edm::LogVerbatim("HGCalGeom") << "cassettes_: " << phgp->cassettes_;

    edm::LogVerbatim("HGCalGeom") << "waferUVMax: " << phgp->waferUVMax_;
    myPrint("waferUVMaxLayer", phgp->waferUVMaxLayer_, 20);
    myPrint("CellThickness", phgp->cellThickness_, 10);
    myPrint("radius100to200", phgp->radius100to200_, 10);
    myPrint("radius200to300", phgp->radius200to300_, 10);
    edm::LogVerbatim("HGCalGeom") << "choiceType " << phgp->choiceType_ << "   nCornerCut " << phgp->nCornerCut_
                                  << "  fracAreaMin " << phgp->fracAreaMin_ << "  zMinForRad " << phgp->zMinForRad_;

    myPrint("CellSize", phgp->cellSize_, 10);
    myPrint("radiusMixBoundary", phgp->radiusMixBoundary_, 10);
    myPrint("LayerCenter", phgp->layerCenter_, 20);
    edm::LogVerbatim("HGCalGeom") << "Layer Rotation " << phgp->layerRotation_ << "   with " << phgp->layerRotV_.size()
                                  << "  parameters";
    for (unsigned int k = 0; k < phgp->layerRotV_.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "Element[" << k << "] " << phgp->layerRotV_[k].first << ":"
                                    << phgp->layerRotV_[k].second;
    myPrint("slopeMin", phgp->slopeMin_, 10);
    myPrint("zFrontMin", phgp->zFrontMin_, 10);
    myPrint("rMinFront", phgp->rMinFront_, 10);
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
    myPrint("trformTranX", phgp->trformTranX_, 8);
    myPrint("trformTranY", phgp->trformTranY_, 8);
    myPrint("trformTranZ", phgp->trformTranZ_, 8);
    myPrint("trformRotXX", phgp->trformRotXX_, 10);
    myPrint("trformRotYX", phgp->trformRotYX_, 10);
    myPrint("trformRotZX", phgp->trformRotZX_, 10);
    myPrint("trformRotXY", phgp->trformRotXY_, 10);
    myPrint("trformRotYY", phgp->trformRotYY_, 10);
    myPrint("trformRotZY", phgp->trformRotZY_, 10);
    myPrint("trformRotXZ", phgp->trformRotXZ_, 10);
    myPrint("trformRotYZ", phgp->trformRotYZ_, 10);
    myPrint("trformRotZZ", phgp->trformRotZZ_, 10);
    myPrint("xLayerHex", phgp->xLayerHex_, 8);
    myPrint("yLayerHex", phgp->yLayerHex_, 8);
    myPrint("zLayerHex", phgp->zLayerHex_, 8);
    myPrint("rMinLayHex", phgp->rMinLayHex_, 8);
    myPrint("rMaxLayHex", phgp->rMaxLayHex_, 8);
    myPrint("waferPos", phgp->waferPosX_, phgp->waferPosY_, 4);
    myPrint("cellFineIndex", phgp->cellFineIndex_, 8);
    myPrint("cellFine", phgp->cellFineX_, phgp->cellFineY_, 4);
    myPrint("cellCoarseIndex", phgp->cellCoarseIndex_, 8);
    myPrint("cellCoarse", phgp->cellCoarseX_, phgp->cellCoarseY_, 4);
    myPrint("layer", phgp->layer_, 18);
    myPrint("layerIndex", phgp->layerIndex_, 18);
    myPrint("depth", phgp->depth_, 18);
    myPrint("depthIndex", phgp->depthIndex_, 18);
    myPrint("depthLayerF", phgp->depthLayerF_, 18);
    myPrint("waferCopy", phgp->waferCopy_, 10);
    myPrint("waferTypeL", phgp->waferTypeL_, 25);
    printTrform(phgp);
    myPrint("levelTop", phgp->levelT_, 10);
    printWaferType(phgp);
    myPrint("cassetteShift", phgp->cassetteShift_, 8);

    edm::LogVerbatim("HGCalGeom") << "MaskMode: " << phgp->waferMaskMode_;
    if (phgp->waferMaskMode_ > 1) {
      edm::LogVerbatim("HGCalGeom") << "WaferInfo with " << phgp->waferInfoMap_.size() << " elements";
      unsigned int kk(0);
      std::unordered_map<int32_t, HGCalParameters::waferInfo>::const_iterator itr = phgp->waferInfoMap_.begin();
      for (; itr != phgp->waferInfoMap_.end(); ++itr, ++kk)
        edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << itr->first << "["
                                      << HGCalWaferIndex::waferLayer(itr->first) << ", "
                                      << HGCalWaferIndex::waferU(itr->first) << ", "
                                      << HGCalWaferIndex::waferV(itr->first) << "] (" << (itr->second).type << ", "
                                      << (itr->second).part << ", " << (itr->second).orient << ")";
    }
  } else {
    // Tpaezoid (scintillator) type
    edm::LogVerbatim("HGCalGeom") << "DetectorType: " << phgp->detectorType_;
    edm::LogVerbatim("HGCalGeom") << "UseSimWt: " << phgp->useSimWt_;
    edm::LogVerbatim("HGCalGeom") << "nCells_: " << phgp->nCellsFine_ << ":" << phgp->nCellsCoarse_;
    edm::LogVerbatim("HGCalGeom") << "MinTileZize: " << phgp->minTileSize_;
    edm::LogVerbatim("HGCalGeom") << "FirstLayer: " << phgp->firstLayer_;
    edm::LogVerbatim("HGCalGeom") << "FirstMixedLayer: " << phgp->firstMixedLayer_;
    edm::LogVerbatim("HGCalGeom") << "LayerOffset: " << phgp->layerOffset_;
    edm::LogVerbatim("HGCalGeom") << "mode_: " << phgp->mode_;
    edm::LogVerbatim("HGCalGeom") << "waferUVMax: " << phgp->waferUVMax_;
    edm::LogVerbatim("HGCalGeom") << "nSectors_: " << phgp->nSectors_;
    edm::LogVerbatim("HGCalGeom") << "nCells_: " << phgp->nCellsFine_ << ":" << phgp->nCellsCoarse_;

    myPrint("CellSize", phgp->cellSize_, 10);
    myPrint("radiusMixBoundary", phgp->radiusMixBoundary_, 10);
    myPrint("nPhiBinBH", phgp->nPhiBinBH_, 18);
    myPrint("layerFrontBH", phgp->layerFrontBH_, 10);
    myPrint("LayerCenter", phgp->layerCenter_, 20);
    myPrint("rMinLayerBH", phgp->rMinLayerBH_, 10);
    myPrint("slopeMin", phgp->slopeMin_, 10);
    myPrint("zFrontMin", phgp->zFrontMin_, 10);
    myPrint("rMinFront", phgp->rMinFront_, 10);
    myPrint("radiusLayer[0]", phgp->radiusLayer_[0], 10);
    myPrint("radiusLayer[1]", phgp->radiusLayer_[1], 10);
    myPrint("iradMinBH", phgp->iradMinBH_, 20);
    myPrint("iradMaxBH", phgp->iradMaxBH_, 20);
    myPrint("slopeTop", phgp->slopeTop_, 10);
    myPrint("zFrontTop", phgp->zFrontTop_, 10);
    myPrint("rMaxFront", phgp->rMaxFront_, 10);
    myPrint("zRanges", phgp->zRanges_, 10);
    myPrint("firstModule", phgp->firstModule_, 10);
    myPrint("lastModule", phgp->lastModule_, 10);
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
    myPrint("moduleCellR", phgp->moduleCellR_, 9);
    myPrint("trformTranX", phgp->trformTranY_, 9);
    myPrint("trformTranY", phgp->trformTranY_, 9);
    myPrint("trformTranZ", phgp->trformTranZ_, 9);
    myPrint("trformRotXX", phgp->trformRotXX_, 10);
    myPrint("trformRotYX", phgp->trformRotYX_, 10);
    myPrint("trformRotZX", phgp->trformRotZX_, 10);
    myPrint("trformRotXY", phgp->trformRotXY_, 10);
    myPrint("trformRotYY", phgp->trformRotYY_, 10);
    myPrint("trformRotZY", phgp->trformRotZY_, 10);
    myPrint("trformRotXZ", phgp->trformRotXZ_, 10);
    myPrint("trformRotYZ", phgp->trformRotYZ_, 10);
    myPrint("trformRotZZ", phgp->trformRotZZ_, 10);
    myPrint("xLayerHex", phgp->xLayerHex_, 10);
    myPrint("yLayerHex", phgp->yLayerHex_, 10);
    myPrint("zLayerHex", phgp->zLayerHex_, 10);
    myPrint("rMinLayHex", phgp->rMinLayHex_, 9);
    myPrint("rMaxLayHex", phgp->rMaxLayHex_, 9);
    myPrint("layer", phgp->layer_, 18);
    myPrint("layerIndex", phgp->layerIndex_, 18);
    myPrint("depth", phgp->depth_, 18);
    myPrint("depthIndex", phgp->depthIndex_, 18);
    myPrint("depthLayerF", phgp->depthLayerF_, 18);
    printTrform(phgp);
    myPrint("levelTop", phgp->levelT_, 10);
    printWaferType(phgp);

    edm::LogVerbatim("HGCalGeom") << "MaskMode: " << phgp->waferMaskMode_;
    if (phgp->waferMaskMode_ > 1) {
      myPrint("tileRingR", phgp->tileRingR_, 4);
      myPrint("tileRingRange", phgp->tileRingRange_, 8);
      edm::LogVerbatim("HGCalGeom") << "TileInfo with " << phgp->tileInfoMap_.size() << " elements";
      unsigned int kk(0);
      std::unordered_map<int32_t, HGCalParameters::tileInfo>::const_iterator itr = phgp->tileInfoMap_.begin();
      for (; itr != phgp->tileInfoMap_.end(); ++itr, ++kk)
        edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << itr->first << "[" << HGCalTileIndex::tileLayer(itr->first)
                                      << ", " << HGCalTileIndex::tileRing(itr->first) << ", "
                                      << HGCalTileIndex::tilePhi(itr->first) << "] (" << (itr->second).type << ", "
                                      << (itr->second).sipm << std::hex << ", " << (itr->second).hex[0] << ", "
                                      << (itr->second).hex[1] << ", " << (itr->second).hex[2] << ", "
                                      << (itr->second).hex[3] << ")" << std::dec;
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  edm::LogVerbatim("HGCalGeom") << "Elapsed time: " << elapsed.count() << " s";
}

template <typename T>
void HGCalParameterTester::myPrint(std::string const& s, std::vector<T> const& obj, int n) const {
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

template <typename T>
void HGCalParameterTester::myPrint(std::string const& s, std::vector<std::pair<T, T> > const& obj, int n) const {
  int k(0), kk(0);
  edm::LogVerbatim("HGCalGeom") << s << " with " << obj.size() << " elements with n " << n << ":200";
  std::ostringstream st1[200];
  for (auto const& it : obj) {
    st1[kk] << "(" << it.first << ", " << it.second << ") ";
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

void HGCalParameterTester::myPrint(std::string const& s,
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

void HGCalParameterTester::myPrint(std::string const& s, HGCalParameters::wafer_map const& obj, int n) const {
  int k(0), kk(0);
  std::ostringstream st1[100];
  edm::LogVerbatim("HGCalGeom") << s << " with " << obj.size() << " elements with n " << n << ": 100";
  for (auto const& it : obj) {
    st1[kk] << it.first << ":" << it.second << ", ";
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

void HGCalParameterTester::printTrform(HGCalParameters const* phgp) const {
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

void HGCalParameterTester::printWaferType(HGCalParameters const* phgp) const {
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

DEFINE_FWK_MODULE(HGCalParameterTester);
