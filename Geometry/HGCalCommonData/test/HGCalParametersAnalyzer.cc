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
  edm::LogVerbatim("HGCalGeom") << "nCells " << phgp->nCells_;
  edm::LogVerbatim("HGCalGeom") << "nSectors " << phgp->nSectors_;
  std::ostringstream st1;
  st1 << "cellsize ";
  for (auto it : phgp->cellSize_)
    st1 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st1.str();

  std::ostringstream st2;
  st2 << "moduleLayS ";
  for (auto it : phgp->moduleLayS_)
    st2 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st2.str();

  std::ostringstream st3;
  st3 << "ModuleBlS ";
  for (auto it : phgp->moduleBlS_)
    st3 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st3.str();

  std::ostringstream st4;
  st4 << "ModuleTlS ";
  for (auto it : phgp->moduleTlS_)
    st4 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st4.str();

  std::ostringstream st5;
  st5 << "moduleHS ";
  for (auto it : phgp->moduleHS_)
    st5 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st5.str();

  std::ostringstream st6;
  st6 << "moduleDzS ";
  for (auto it : phgp->moduleDzS_)
    st6 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st6.str();

  std::ostringstream st7;
  st7 << "moduleAlphaS ";
  for (auto it : phgp->moduleAlphaS_)
    st7 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st7.str();

  std::ostringstream st8;
  st8 << "moduleCellS ";
  for (auto it : phgp->moduleCellS_)
    st8 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st8.str();

  std::ostringstream st9;
  st9 << "moduleLayR ";
  for (auto it : phgp->moduleLayR_)
    st9 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st9.str();

  std::ostringstream st10;
  st10 << "moduleBlR ";
  for (auto it : phgp->moduleBlR_)
    st10 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st10.str();

  std::ostringstream st11;
  st11 << "moduleTlR ";
  for (auto it : phgp->moduleTlR_)
    st11 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st11.str();

  std::ostringstream st12;
  st12 << "moduleHR ";
  for (auto it : phgp->moduleHR_)
    st12 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st12.str();

  std::ostringstream st13;
  st13 << "moduleDzR ";
  for (auto it : phgp->moduleDzR_)
    st13 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st13.str();

  std::ostringstream st14;
  st14 << "moduleAlphaR ";
  for (auto it : phgp->moduleAlphaR_)
    st14 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st14.str();

  std::ostringstream st15;
  st15 << "moduleCellR ";
  for (auto it : phgp->moduleCellR_)
    st15 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st15.str();

  std::ostringstream st16;
  st16 << "trformIndex ";
  for (auto it : phgp->trformIndex_)
    st16 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st16.str();

  std::ostringstream st17;
  st17 << "trformTranX ";
  for (auto it : phgp->trformTranX_)
    st17 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st17.str();

  std::ostringstream st18;
  st18 << "trformTranY ";
  for (auto it : phgp->trformTranY_)
    st18 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st18.str();

  std::ostringstream st19;
  st19 << "trformTranZ ";
  for (auto it : phgp->trformTranZ_)
    st19 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st19.str();

  std::ostringstream st20;
  st20 << "trformRotXX ";
  for (auto it : phgp->trformRotXX_)
    st20 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st20.str();

  std::ostringstream st21;
  st21 << "trformRotYX ";
  for (auto it : phgp->trformRotYX_)
    st21 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st21.str();

  std::ostringstream st22;
  st22 << "trformRotZX ";
  for (auto it : phgp->trformRotZX_)
    st22 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st22.str();

  std::ostringstream st23;
  st23 << "trformRotXY ";
  for (auto it : phgp->trformRotXY_)
    st23 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st23.str();

  std::ostringstream st24;
  st24 << "trformRotYY ";
  for (auto it : phgp->trformRotYY_)
    st24 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st24.str();

  std::ostringstream st25;
  st25 << "trformRotZY ";
  for (auto it : phgp->trformRotZY_)
    st25 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st25.str();

  std::ostringstream st26;
  st26 << "trformRotXZ ";
  for (auto it : phgp->trformRotXZ_)
    st26 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st26.str();

  std::ostringstream st27;
  st27 << "trformRotYZ ";
  for (auto it : phgp->trformRotYZ_)
    st27 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st27.str();

  std::ostringstream st28;
  st28 << "trformRotZZ ";
  for (auto it : phgp->trformRotZZ_)
    st28 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st28.str();

  std::ostringstream st29;
  st29 << "layer ";
  for (auto it : phgp->layer_)
    st29 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st29.str();

  std::ostringstream st30;
  st3 << "layerIndex ";
  for (auto it : phgp->layerIndex_)
    st30 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st30.str();

  std::ostringstream st31;
  st31 << "layerGroup ";
  for (auto it : phgp->layerGroup_)
    st31 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st31.str();

  std::ostringstream st32;
  st32 << "cellFactor ";
  for (auto it : phgp->cellFactor_)
    st32 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st32.str();

  std::ostringstream st33;
  st33 << "depth ";
  for (auto it : phgp->depth_)
    st33 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st33.str();

  std::ostringstream st34;
  st34 << "depthIndex ";
  for (auto it : phgp->depthIndex_)
    st34 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st34.str();

  std::ostringstream st35;
  st35 << "depthLayerF ";
  for (auto it : phgp->depthLayerF_)
    st35 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st35.str();

  std::ostringstream st36;
  st36 << "zLayerHex ";
  for (auto it : phgp->zLayerHex_)
    st36 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st36.str();

  std::ostringstream st37;
  st37 << "MinLayHex ";
  for (auto it : phgp->rMinLayHex_)
    st37 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st37.str();

  std::ostringstream st38;
  st38 << "rMaxLayHex ";
  for (auto it : phgp->rMaxLayHex_)
    st38 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st38.str();

  std::ostringstream st39;
  st39 << "waferCopy ";
  for (auto it : phgp->waferCopy_)
    st39 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st39.str();

  std::ostringstream st40;
  st40 << "waferTypeL ";
  for (auto it : phgp->waferTypeL_)
    st40 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st40.str();

  std::ostringstream st41;
  st41 << "waferTypeT ";
  for (auto it : phgp->waferTypeT_)
    st41 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st41.str();

  std::ostringstream st42;
  st42 << "waferPos ";
  for (unsigned int k = 0; k < phgp->waferPosX_.size(); ++k)
    st42 << "(" << phgp->waferPosX_[k] << ", " << phgp->waferPosY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st42.str();

  std::ostringstream st43;
  st43 << "cellFinePos ";
  for (unsigned int k = 0; k < phgp->cellFineX_.size(); ++k)
    st43 << "(" << phgp->cellFineX_[k] << ", " << phgp->cellFineY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st43.str();

  std::ostringstream st44;
  st44 << "cellCoarsePos ";
  for (unsigned int k = 0; k < phgp->cellCoarseX_.size(); ++k)
    st44 << "(" << phgp->cellCoarseX_[k] << ", " << phgp->cellCoarseY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st44.str();

  std::ostringstream st45;
  st45 << "layerGroupM ";
  for (auto it : phgp->layerGroupM_)
    st45 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st45.str();

  std::ostringstream st46;
  st46 << "layerGroupO ";
  for (auto it : phgp->layerGroupO_)
    st46 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st46.str();

  std::ostringstream st47;
  st47 << "boundR ";
  for (auto it : phgp->boundR_)
    st47 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st47.str();

  edm::LogVerbatim("HGCalGeom") << "waferR " << phgp->waferR_;
  edm::LogVerbatim("HGCalGeom") << "mode " << phgp->mode_;

  std::ostringstream st48;
  st48 << "slopeMin ";
  for (auto it : phgp->slopeMin_)
    st48 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st48.str();
}

DEFINE_FWK_MODULE(HGCalParametersAnalyzer);
