#include <iostream>
#include <sstream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/GeometryObjects/interface/PHGCalParameters.h"
#include "Geometry/Records/interface/PHGCalParametersRcd.h"

class HGCalParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalParametersAnalyzer(const edm::ParameterSet&)
      : token_{esConsumes<PHGCalParameters, PHGCalParametersRcd>(edm::ESInputTag{})} {}
  ~HGCalParametersAnalyzer() override = default;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  const edm::ESGetToken<PHGCalParameters, PHGCalParametersRcd> token_;
};

void HGCalParametersAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("HGCalParametersAnalyzer") << "Here I am";

  const auto& hgp = iSetup.getData(token_);
  const auto* phgp = &hgp;

  edm::LogVerbatim("HGCalGeom") << "\nname " << phgp->name_;
  edm::LogVerbatim("HGCalGeom") << "detectorType " << phgp->detectorType_;
  edm::LogVerbatim("HGCalGeom") << "useSimWt " << phgp->useSimWt_;
  edm::LogVerbatim("HGCalGeom") << "nCells " << phgp->nCells_;
  edm::LogVerbatim("HGCalGeom") << "nSectors " << phgp->nSectors_;
  edm::LogVerbatim("HGCalGeom") << "firstLayer " << phgp->firstLayer_;
  edm::LogVerbatim("HGCalGeom") << "firstMixedLayer " << phgp->firstMixedLayer_;
  edm::LogVerbatim("HGCalGeom") << "mode " << phgp->mode_;
  std::ostringstream st01;
  st01 << "cellsize ";
  for (auto it : phgp->cellSize_)
    st01 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st01.str();

  std::ostringstream st02;
  st02 << "slopeMin ";
  for (auto it : phgp->slopeMin_)
    st02 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st02.str();

  std::ostringstream st03;
  st03 << "zFrontMin ";
  for (auto it : phgp->zFrontMin_)
    st03 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st03.str();

  std::ostringstream st04;
  st04 << "rMinFront ";
  for (auto it : phgp->rMinFront_)
    st04 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st04.str();

  std::ostringstream st05;
  st05 << "slopeTop ";
  for (auto it : phgp->slopeTop_)
    st05 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st05.str();

  std::ostringstream st06;
  st06 << "zFrontTop ";
  for (auto it : phgp->zFrontTop_)
    st06 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st06.str();

  std::ostringstream st07;
  st07 << "rMaxFront ";
  for (auto it : phgp->rMaxFront_)
    st07 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st07.str();

  std::ostringstream st08;
  st08 << "zRanges ";
  for (auto it : phgp->zRanges_)
    st08 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st08.str();

  std::ostringstream st09;
  st09 << "moduleLayS ";
  for (auto it : phgp->moduleLayS_)
    st09 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st09.str();

  std::ostringstream st10;
  st10 << "ModuleBlS ";
  for (auto it : phgp->moduleBlS_)
    st10 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st10.str();

  std::ostringstream st11;
  st11 << "ModuleTlS ";
  for (auto it : phgp->moduleTlS_)
    st11 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st11.str();

  std::ostringstream st12;
  st12 << "moduleHS ";
  for (auto it : phgp->moduleHS_)
    st12 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st12.str();

  std::ostringstream st13;
  st13 << "moduleDzS ";
  for (auto it : phgp->moduleDzS_)
    st13 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st13.str();

  std::ostringstream st14;
  st14 << "moduleAlphaS ";
  for (auto it : phgp->moduleAlphaS_)
    st14 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st14.str();

  std::ostringstream st15;
  st15 << "moduleCellS ";
  for (auto it : phgp->moduleCellS_)
    st15 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st15.str();

  std::ostringstream st16;
  st16 << "moduleLayR ";
  for (auto it : phgp->moduleLayR_)
    st16 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st16.str();

  std::ostringstream st17;
  st17 << "moduleBlR ";
  for (auto it : phgp->moduleBlR_)
    st17 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st17.str();

  std::ostringstream st18;
  st18 << "moduleTlR ";
  for (auto it : phgp->moduleTlR_)
    st18 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st18.str();

  std::ostringstream st19;
  st19 << "moduleHR ";
  for (auto it : phgp->moduleHR_)
    st19 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st19.str();

  std::ostringstream st20;
  st20 << "moduleDzR ";
  for (auto it : phgp->moduleDzR_)
    st20 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st20.str();

  std::ostringstream st21;
  st21 << "moduleAlphaR ";
  for (auto it : phgp->moduleAlphaR_)
    st21 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st21.str();

  std::ostringstream st22;
  st22 << "moduleCellR ";
  for (auto it : phgp->moduleCellR_)
    st22 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st22.str();

  std::ostringstream st23;
  st23 << "trformIndex ";
  for (auto it : phgp->trformIndex_)
    st23 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st23.str();

  std::ostringstream st24;
  st24 << "trformTranX ";
  for (auto it : phgp->trformTranX_)
    st24 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st24.str();

  std::ostringstream st25;
  st25 << "trformTranY ";
  for (auto it : phgp->trformTranY_)
    st25 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st25.str();

  std::ostringstream st26;
  st26 << "trformTranZ ";
  for (auto it : phgp->trformTranZ_)
    st26 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st26.str();

  std::ostringstream st27;
  st27 << "trformRotXX ";
  for (auto it : phgp->trformRotXX_)
    st27 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st27.str();

  std::ostringstream st28;
  st28 << "trformRotYX ";
  for (auto it : phgp->trformRotYX_)
    st28 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st28.str();

  std::ostringstream st29;
  st29 << "trformRotZX ";
  for (auto it : phgp->trformRotZX_)
    st29 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st29.str();

  std::ostringstream st30;
  st30 << "trformRotXY ";
  for (auto it : phgp->trformRotXY_)
    st30 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st30.str();

  std::ostringstream st31;
  st31 << "trformRotYY ";
  for (auto it : phgp->trformRotYY_)
    st31 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st31.str();

  std::ostringstream st32;
  st32 << "trformRotZY ";
  for (auto it : phgp->trformRotZY_)
    st32 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st32.str();

  std::ostringstream st33;
  st33 << "trformRotXZ ";
  for (auto it : phgp->trformRotXZ_)
    st33 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st33.str();

  std::ostringstream st34;
  st34 << "trformRotYZ ";
  for (auto it : phgp->trformRotYZ_)
    st34 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st34.str();

  std::ostringstream st35;
  st35 << "trformRotZZ ";
  for (auto it : phgp->trformRotZZ_)
    st35 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st35.str();

  std::ostringstream st36;
  st36 << "xLayerHex ";
  for (auto it : phgp->xLayerHex_)
    st36 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st36.str();

  std::ostringstream st37;
  st37 << "yLayerHex ";
  for (auto it : phgp->yLayerHex_)
    st37 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st37.str();

  std::ostringstream st38;
  st38 << "zLayerHex ";
  for (auto it : phgp->zLayerHex_)
    st38 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st38.str();

  std::ostringstream st39;
  st39 << "rMinLayHex ";
  for (auto it : phgp->rMinLayHex_)
    st39 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st39.str();

  std::ostringstream st40;
  st40 << "rMaxLayHex ";
  for (auto it : phgp->rMaxLayHex_)
    st40 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st40.str();

  std::ostringstream st41;
  st41 << "waferPos ";
  for (unsigned int k = 0; k < phgp->waferPosX_.size(); ++k)
    st41 << "(" << phgp->waferPosX_[k] << ", " << phgp->waferPosY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st41.str();

  std::ostringstream st42;
  st42 << "cellFine ";
  for (unsigned int k = 0; k < phgp->cellFineX_.size(); ++k)
    st42 << "(" << phgp->cellFineX_[k] << ", " << phgp->cellFineY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st42.str();

  std::ostringstream st43;
  st43 << "cellFineHalf ";
  for (auto it : phgp->cellFineHalf_)
    st43 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st43.str();

  std::ostringstream st44;
  st44 << "cellCoarse ";
  for (unsigned int k = 0; k < phgp->cellCoarseX_.size(); ++k)
    st44 << "(" << phgp->cellCoarseX_[k] << ", " << phgp->cellCoarseY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st44.str();

  std::ostringstream st45;
  st45 << "cellCoarseHalf ";
  for (auto it : phgp->cellCoarseHalf_)
    st45 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st45.str();

  std::ostringstream st46;
  st46 << "boundR ";
  for (auto it : phgp->boundR_)
    st46 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st46.str();

  std::ostringstream st47;
  st47 << "layer ";
  for (auto it : phgp->layer_)
    st47 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st47.str();

  std::ostringstream st48;
  st48 << "layerIndex ";
  for (auto it : phgp->layerIndex_)
    st48 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st48.str();

  std::ostringstream st49;
  st49 << "layerGroup ";
  for (auto it : phgp->layerGroup_)
    st49 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st49.str();

  std::ostringstream st50;
  st50 << "cellFactor ";
  for (auto it : phgp->cellFactor_)
    st50 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st50.str();

  std::ostringstream st51;
  st51 << "depth ";
  for (auto it : phgp->depth_)
    st51 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st51.str();

  std::ostringstream st52;
  st52 << "depthIndex ";
  for (auto it : phgp->depthIndex_)
    st52 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st52.str();

  std::ostringstream st53;
  st53 << "depthLayerF ";
  for (auto it : phgp->depthLayerF_)
    st53 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st53.str();

  std::ostringstream st54;
  st54 << "waferCopy ";
  for (auto it : phgp->waferCopy_)
    st54 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st54.str();

  std::ostringstream st55;
  st55 << "waferTypeL ";
  for (auto it : phgp->waferTypeL_)
    st55 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st55.str();

  std::ostringstream st56;
  st56 << "waferTypeT ";
  for (auto it : phgp->waferTypeT_)
    st56 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st56.str();

  std::ostringstream st57;
  st57 << "layerGroupM ";
  for (auto it : phgp->layerGroupM_)
    st57 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st57.str();

  std::ostringstream st58;
  st58 << "layerGroupO ";
  for (auto it : phgp->layerGroupO_)
    st58 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st58.str();

  std::ostringstream st59;
  st59 << "rLimit ";
  for (auto it : phgp->rLimit_)
    st59 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st59.str();

  std::ostringstream st60;
  st60 << "cellFine ";
  for (auto it : phgp->cellFine_)
    st60 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st60.str();

  std::ostringstream st61;
  st61 << "cellCoarse ";
  for (auto it : phgp->cellCoarse_)
    st61 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st61.str();

  edm::LogVerbatim("HGCalGeom") << "waferR " << phgp->waferR_;

  std::ostringstream st62;
  st62 << "levelT ";
  for (auto it : phgp->levelT_)
    st62 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st62.str();

  edm::LogVerbatim("HGCalGeom") << "levelSide " << phgp->levelZSide_;
  edm::LogVerbatim("HGCalGeom") << "nCellsFine " << phgp->nCellsFine_;
  edm::LogVerbatim("HGCalGeom") << "nCellsCoarse " << phgp->nCellsCoarse_;
  edm::LogVerbatim("HGCalGeom") << "waferSize " << phgp->waferSize_;
  edm::LogVerbatim("HGCalGeom") << "waferThick " << phgp->waferThick_;
  edm::LogVerbatim("HGCalGeom") << "sensorSeparation " << phgp->sensorSeparation_;
  edm::LogVerbatim("HGCalGeom") << "sensorSizeOffset " << phgp->sensorSizeOffset_;
  edm::LogVerbatim("HGCalGeom") << "guardRingOffset " << phgp->guardRingOffset_;
  edm::LogVerbatim("HGCalGeom") << "mouseBite " << phgp->mouseBite_;
  edm::LogVerbatim("HGCalGeom") << "useOffset " << phgp->useOffset_;
  edm::LogVerbatim("HGCalGeom") << "waferUVMax " << phgp->waferUVMax_;
  edm::LogVerbatim("HGCalGeom") << "defineFull " << phgp->defineFull_;

  std::ostringstream st63;
  st63 << "waferUVMaxLayer ";
  for (auto it : phgp->waferUVMaxLayer_)
    st63 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st63.str();

  std::ostringstream st64;
  st64 << "waferThickness ";
  for (auto it : phgp->waferThickness_)
    st64 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st64.str();

  std::ostringstream st65;
  st65 << "cellThickness ";
  for (auto it : phgp->cellThickness_)
    st65 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st65.str();

  std::ostringstream st66;
  st66 << "radius100to200 ";
  for (auto it : phgp->radius100to200_)
    st66 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st66.str();

  std::ostringstream st67;
  st67 << "radius200to300 ";
  for (auto it : phgp->radius200to300_)
    st67 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st67.str();

  edm::LogVerbatim("HGCalGeom") << "choiceType " << phgp->choiceType_;
  edm::LogVerbatim("HGCalGeom") << "nCornercut " << phgp->nCornerCut_;
  edm::LogVerbatim("HGCalGeom") << "fracAreaMin " << phgp->fracAreaMin_;
  edm::LogVerbatim("HGCalGeom") << "zMinForRad " << phgp->zMinForRad_;

  std::ostringstream st68;
  st68 << "radiusMixBoundary ";
  for (auto it : phgp->radiusMixBoundary_)
    st68 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st68.str();

  std::ostringstream st69;
  st69 << "nPhiBinBH ";
  for (auto it : phgp->nPhiBinBH_)
    st69 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st69.str();

  std::ostringstream st70;
  st70 << "layerFrontBH ";
  for (auto it : phgp->layerFrontBH_)
    st70 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st70.str();

  std::ostringstream st71;
  st71 << "rMinLayerBH ";
  for (auto it : phgp->rMinLayerBH_)
    st71 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st71.str();

  for (int k = 0; k < 2; ++k) {
    std::ostringstream st71;
    st71 << "radiusLayer[" << k << "] ";
    for (auto it : phgp->radiusLayer_[k])
      st71 << it << ", ";
    edm::LogVerbatim("HGCalGeom") << st71.str();
  }

  std::ostringstream st72;
  st72 << "iradBH ";
  for (unsigned int k = 0; k < phgp->iradMinBH_.size(); ++k)
    st72 << phgp->iradMinBH_[k] << ":" << phgp->iradMaxBH_[k] << " ";
  edm::LogVerbatim("HGCalGeom") << st72.str();

  std::ostringstream st73;
  st73 << "iradBHFine ";
  for (unsigned int k = 0; k < phgp->iradMinBHFine_.size(); ++k)
    st73 << phgp->iradMinBHFine_[k] << ":" << phgp->iradMaxBHFine_[k] << " ";
  edm::LogVerbatim("HGCalGeom") << st73.str();

  edm::LogVerbatim("HGCalGeom") << "minTileSize " << phgp->minTileSize_;

  std::ostringstream st74;
  st74 << "firstModule ";
  for (auto it : phgp->firstModule_)
    st74 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st74.str();

  std::ostringstream st75;
  st75 << "lastModule ";
  for (auto it : phgp->lastModule_)
    st75 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st75.str();

  edm::LogVerbatim("HGCalGeom") << "layerOffset " << phgp->layerOffset_;
  edm::LogVerbatim("HGCalGeom") << "layerRotation " << phgp->layerRotation_;
  edm::LogVerbatim("HGCalGeom") << "waferMaskMode " << phgp->waferMaskMode_;
  edm::LogVerbatim("HGCalGeom") << "waferZSide " << phgp->waferZSide_;

  std::ostringstream st76;
  st76 << "layerType ";
  for (auto it : phgp->layerType_)
    st76 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st76.str();

  std::ostringstream st77;
  st77 << "layerCenter ";
  for (auto it : phgp->layerCenter_)
    st77 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st77.str();

  std::ostringstream st78;
  st78 << "nPhiLayer ";
  for (auto it : phgp->nPhiLayer_)
    st78 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st78.str();

  edm::LogVerbatim("HGCalGeom") << "cassettes " << phgp->cassettes_;
  edm::LogVerbatim("HGCalGeom") << "nPhiCassette " << phgp->nphiCassette_;
  edm::LogVerbatim("HGCalGeom") << "nPhiFineCassette " << phgp->nphiFineCassette_;
  edm::LogVerbatim("HGCalGeom") << "phiOffset " << phgp->phiOffset_;

  std::ostringstream st79;
  st79 << "cassetteShift ";
  for (auto it : phgp->cassetteShift_)
    st79 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st79.str();

  std::ostringstream st80;
  st80 << "cassetteShiftTile ";
  for (auto it : phgp->cassetteShiftTile_)
    st80 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st80.str();

  std::ostringstream st81;
  st81 << "cassetteRetractTile ";
  for (auto it : phgp->cassetteRetractTile_)
    st81 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st81.str();

  edm::LogVerbatim("HGCalGeom") << "calibCellRHD " << phgp->calibCellRHD_;
  edm::LogVerbatim("HGCalGeom") << "calibCellRLD " << phgp->calibCellRLD_;
  edm::LogVerbatim("HGCalGeom") << "tileUVMax " << phgp->tileUVMax_;
  edm::LogVerbatim("HGCalGeom") << "tileUVMaxFine " << phgp->tileUVMaxFine_;

  std::ostringstream st82;
  st82 << "calibCellFullHD ";
  for (auto it : phgp->calibCellFullHD_)
    st82 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st82.str();

  std::ostringstream st83;
  st83 << "calibCellPartHD ";
  for (auto it : phgp->calibCellPartHD_)
    st83 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st83.str();

  std::ostringstream st84;
  st84 << "calibCellFullHD ";
  for (auto it : phgp->calibCellFullLD_)
    st84 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st84.str();

  std::ostringstream st85;
  st85 << "calibCellPartLD ";
  for (auto it : phgp->calibCellPartLD_)
    st85 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st85.str();
}

DEFINE_FWK_MODULE(HGCalParametersAnalyzer);
