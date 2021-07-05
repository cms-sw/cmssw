#include <iostream>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalParameterTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalParameterTester(const edm::ParameterSet&);
  ~HGCalParameterTester() override {}
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

  std::cout << phgp->name_ << "\n";
  if (mode_ == 0) {
    // Wafers of 6-inch format
    std::cout << "DetectorType: " << phgp->detectorType_ << "\n";
    std::cout << "WaferR_: " << phgp->waferR_ << "\n";
    std::cout << "nCells_: " << phgp->nCells_ << "\n";
    std::cout << "nSectors_: " << phgp->nSectors_ << "\n";
    std::cout << "FirstLayer: " << phgp->firstLayer_ << "\n";
    std::cout << "FirstMixedLayer: " << phgp->firstMixedLayer_ << "\n";
    std::cout << "mode_: " << phgp->mode_ << "\n";

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
    myPrint("waferCopy", phgp->waferCopy_, 8);
    myPrint("waferTypeL", phgp->waferTypeL_, 20);
    myPrint("waferTypeT", phgp->waferTypeT_, 20);
    myPrint("layerGroupM", phgp->layerGroupM_, 18);
    myPrint("layerGroupO", phgp->layerGroupO_, 18);
    printTrform(phgp);
    myPrint("levelTop", phgp->levelT_, 10);
    printWaferType(phgp);

  } else if (mode_ == 1) {
    // Wafers of 8-inch format
    std::cout << "DetectorType: " << phgp->detectorType_ << "\n";
    std::cout << "Wafer Parameters: " << phgp->waferSize_ << ":" << phgp->waferR_ << ":" << phgp->waferThick_ << ":"
              << phgp->sensorSeparation_ << ":" << phgp->mouseBite_ << "\n";
    myPrint("waferThickness", phgp->waferThickness_, 10);
    std::cout << "nCells_: " << phgp->nCellsFine_ << ":" << phgp->nCellsCoarse_ << "\n";
    std::cout << "nSectors_: " << phgp->nSectors_ << "\n";
    std::cout << "FirstLayer: " << phgp->firstLayer_ << "\n";
    std::cout << "FirstMixedLayer: " << phgp->firstMixedLayer_ << "\n";
    std::cout << "LayerOffset: " << phgp->layerOffset_ << "\n";
    std::cout << "mode_: " << phgp->mode_ << "\n";

    myPrint("waferUVMaxLayer", phgp->waferUVMaxLayer_, 20);
    myPrint("CellThickness", phgp->cellThickness_, 10);
    myPrint("radius100to200", phgp->radius100to200_, 10);
    myPrint("radius200to300", phgp->radius200to300_, 10);
    std::cout << "choiceType " << phgp->choiceType_ << "   nCornerCut " << phgp->nCornerCut_ << "  fracAreaMin "
              << phgp->fracAreaMin_ << "  zMinForRad " << phgp->zMinForRad_ << "\n";

    myPrint("CellSize", phgp->cellSize_, 10);
    myPrint("radiusMixBoundary", phgp->radiusMixBoundary_, 10);
    myPrint("LayerCenter", phgp->layerCenter_, 20);
    std::cout << "Layer Rotation " << phgp->layerRotation_ << "   with " << phgp->layerRotV_.size() << "  parameters\n";
    for (unsigned int k = 0; k < phgp->layerRotV_.size(); ++k)
      std::cout << "Element[" << k << "] " << phgp->layerRotV_[k].first << ":" << phgp->layerRotV_[k].second << "\n";
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
    myPrint("waferCopy", phgp->waferCopy_, 8);
    myPrint("waferTypeL", phgp->waferTypeL_, 20);
    printTrform(phgp);
    myPrint("levelTop", phgp->levelT_, 10);
    printWaferType(phgp);

    std::cout << "MaskMode: " << phgp->waferMaskMode_ << "\n";
    if (phgp->waferMaskMode_ > 1) {
      std::cout << "WaferInfo with " << phgp->waferInfoMap_.size() << " elements\n";
      unsigned int kk(0);
      std::unordered_map<int32_t, HGCalParameters::waferInfo>::const_iterator itr = phgp->waferInfoMap_.begin();
      for (; itr != phgp->waferInfoMap_.end(); ++itr, ++kk)
        std::cout << "[" << kk << "] " << itr->first << "[" << HGCalWaferIndex::waferLayer(itr->first) << ", "
                  << HGCalWaferIndex::waferU(itr->first) << ", " << HGCalWaferIndex::waferV(itr->first) << "] ("
                  << (itr->second).type << ", " << (itr->second).part << ", " << (itr->second).orient << ")"
                  << std::endl;
    }
  } else {
    // Tpaezoid (scintillator) type
    std::cout << "DetectorType: " << phgp->detectorType_ << "\n";
    std::cout << "nCells_: " << phgp->nCellsFine_ << ":" << phgp->nCellsCoarse_ << "\n";
    std::cout << "MinTileZize: " << phgp->minTileSize_ << "\n";
    std::cout << "FirstLayer: " << phgp->firstLayer_ << "\n";
    std::cout << "FirstMixedLayer: " << phgp->firstMixedLayer_ << "\n";
    std::cout << "LayerOffset: " << phgp->layerOffset_ << "\n";
    std::cout << "mode_: " << phgp->mode_ << "\n";
    std::cout << "waferUVMax: " << phgp->waferUVMax_ << "\n";
    std::cout << "nSectors_: " << phgp->nSectors_ << "\n";
    std::cout << "nCells_: " << phgp->nCellsFine_ << ":" << phgp->nCellsCoarse_ << "\n";

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

    std::cout << "MaskMode: " << phgp->waferMaskMode_ << "\n";
    if (phgp->waferMaskMode_ > 1) {
      myPrint("tileRingR", phgp->tileRingR_, 4);
      myPrint("tileRingRange", phgp->tileRingRange_, 8);
      std::cout << "TileInfo with " << phgp->tileInfoMap_.size() << " elements\n";
      unsigned int kk(0);
      std::unordered_map<int32_t, HGCalParameters::tileInfo>::const_iterator itr = phgp->tileInfoMap_.begin();
      for (; itr != phgp->tileInfoMap_.end(); ++itr, ++kk)
        std::cout << "[" << kk << "] " << itr->first << "[" << HGCalTileIndex::tileLayer(itr->first) << ", "
                  << HGCalTileIndex::tileRing(itr->first) << ", " << HGCalTileIndex::tilePhi(itr->first) << "] ("
                  << (itr->second).type << ", " << (itr->second).sipm << std::hex << ", " << (itr->second).hex[0]
                  << ", " << (itr->second).hex[1] << ", " << (itr->second).hex[2] << ", " << (itr->second).hex[3] << ")"
                  << std::dec << std::endl;
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}

template <typename T>
void HGCalParameterTester::myPrint(std::string const& s, std::vector<T> const& obj, int n) const {
  int k(0);
  std::cout << s << " with " << obj.size() << " elements\n";
  for (auto const& it : obj) {
    std::cout << it << ", ";
    ++k;
    if (k == n) {
      std::cout << "\n";
      k = 0;
    }
  }
  if (k > 0)
    std::cout << "\n";
}

template <typename T>
void HGCalParameterTester::myPrint(std::string const& s, std::vector<std::pair<T, T> > const& obj, int n) const {
  int k(0);
  std::cout << s << " with " << obj.size() << " elements\n";
  for (auto const& it : obj) {
    std::cout << "(" << it.first << ", " << it.second << ") ";
    ++k;
    if (k == n) {
      std::cout << "\n";
      k = 0;
    }
  }
  if (k > 0)
    std::cout << "\n";
}

void HGCalParameterTester::myPrint(std::string const& s,
                                   std::vector<double> const& obj1,
                                   std::vector<double> const& obj2,
                                   int n) const {
  int k(0);
  std::cout << s << " with " << obj1.size() << " elements\n";
  for (unsigned int k1 = 0; k1 < obj1.size(); ++k1) {
    std::cout << "(" << obj1[k1] << ", " << obj2[k1] << ") ";
    ++k;
    if (k == n) {
      std::cout << "\n";
      k = 0;
    }
  }
  if (k > 0)
    std::cout << "\n";
}

void HGCalParameterTester::myPrint(std::string const& s, HGCalParameters::wafer_map const& obj, int n) const {
  int k(0);
  std::cout << s << " with " << obj.size() << " elements\n";
  for (auto const& it : obj) {
    std::cout << it.first << ":" << it.second << ", ";
    ++k;
    if (k == n) {
      std::cout << "\n";
      k = 0;
    }
  }
  if (k > 0)
    std::cout << "\n";
}

void HGCalParameterTester::printTrform(HGCalParameters const* phgp) const {
  int k(0);
  std::cout << "TrformIndex with " << phgp->trformIndex_.size() << " elements\n";
  for (unsigned int i = 0; i < phgp->trformIndex_.size(); ++i) {
    std::array<int, 4> id = phgp->getID(i);
    std::cout << id[0] << ":" << id[1] << ":" << id[2] << ":" << id[3] << ", ";
    ++k;
    if (k == 7) {
      std::cout << "\n";
      k = 0;
    }
  }
  if (k > 0)
    std::cout << "\n";
}

void HGCalParameterTester::printWaferType(HGCalParameters const* phgp) const {
  int k(0);
  std::cout << "waferTypes with " << phgp->waferTypes_.size() << " elements\n";
  std::map<std::pair<int, int>, int> kounts;
  std::map<std::pair<int, int>, int>::iterator itr;
  for (auto const& it : phgp->waferTypes_) {
    std::cout << " [" << k << "] " << HGCalWaferIndex::waferLayer(it.first);
    if (HGCalWaferIndex::waferFormat(it.first)) {
      std::cout << ":" << HGCalWaferIndex::waferU(it.first) << ":" << HGCalWaferIndex::waferV(it.first);
    } else {
      std::cout << ":" << HGCalWaferIndex::waferCopy(it.first);
    }
    std::cout << " ==> (" << (it.second).first << ":" << (it.second).second << ")" << std::endl;
    itr = kounts.find(it.second);
    if (itr == kounts.end())
      kounts[it.second] = 1;
    else
      ++(itr->second);
    ++k;
  }
  if (!kounts.empty()) {
    std::cout << "Summary of waferTypes ==========================\n";
    for (itr = kounts.begin(); itr != kounts.end(); ++itr)
      std::cout << "Type (" << (itr->first).first << ":" << (itr->first).second << ") Kount " << itr->second
                << std::endl;
  }
}

DEFINE_FWK_MODULE(HGCalParameterTester);
