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

  edm::LogVerbatim("HGCalGeom") << phgp->name_ << "\n";
  std::ostringstream st1;
  for (auto it : phgp->cellSize_)
    st1 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st1.str();

  std::ostringstream st2;
  for (auto it : phgp->moduleBlS_)
    st2 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st2.str();

  std::ostringstream st3;
  for (auto it : phgp->moduleTlS_)
    st3 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st3.str();

  std::ostringstream st4;
  for (auto it : phgp->moduleHS_)
    st4 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st4.str();

  std::ostringstream st5;
  for (auto it : phgp->moduleDzS_)
    st5 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st5.str();

  std::ostringstream st6;
  for (auto it : phgp->moduleAlphaS_)
    st6 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st6.str();

  std::ostringstream st7;
  for (auto it : phgp->moduleCellS_)
    st7 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st7.str();

  std::ostringstream st8;
  for (auto it : phgp->moduleBlR_)
    st8 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st8.str();

  std::ostringstream st9;
  for (auto it : phgp->moduleTlR_)
    st9 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st9.str();

  std::ostringstream st10;
  for (auto it : phgp->moduleHR_)
    st10 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st10.str();

  std::ostringstream st11;
  for (auto it : phgp->moduleDzR_)
    st11 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st11.str();

  std::ostringstream st12;
  for (auto it : phgp->moduleAlphaR_)
    st12 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st12.str();

  std::ostringstream st13;
  for (auto it : phgp->moduleCellR_)
    st13 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st13.str();

  std::ostringstream st14;
  for (auto it : phgp->trformTranX_)
    st14 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st14.str();

  std::ostringstream st15;
  for (auto it : phgp->trformTranY_)
    st15 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st15.str();

  std::ostringstream st16;
  for (auto it : phgp->trformTranZ_)
    st16 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st16.str();

  std::ostringstream st17;
  for (auto it : phgp->trformRotXX_)
    st17 << it << ", ";

  std::ostringstream st18;
  for (auto it : phgp->trformRotYX_)
    st18 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st18.str();

  std::ostringstream st19;
  for (auto it : phgp->trformRotZX_)
    st19 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st19.str();

  std::ostringstream st20;
  for (auto it : phgp->trformRotXY_)
    st20 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st20.str();

  std::ostringstream st21;
  for (auto it : phgp->trformRotYY_)
    st21 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st21.str();

  std::ostringstream st22;
  for (auto it : phgp->trformRotZY_)
    st22 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st22.str();

  std::ostringstream st23;
  for (auto it : phgp->trformRotXZ_)
    st23 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st23.str();

  std::ostringstream st24;
  for (auto it : phgp->trformRotYZ_)
    st24 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st24.str();

  std::ostringstream st25;
  for (auto it : phgp->trformRotZZ_)
    st25 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st25.str();

  std::ostringstream st26;
  for (auto it : phgp->zLayerHex_)
    st26 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st26.str();

  std::ostringstream st27;
  for (auto it : phgp->rMinLayHex_)
    st27 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st27.str();

  std::ostringstream st28;
  for (auto it : phgp->rMaxLayHex_)
    st28 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st28.str();

  std::ostringstream st29;
  for (unsigned int k = 0; k < phgp->waferPosX_.size(); ++k)
    st29 << "(" << phgp->waferPosX_[k] << ", " << phgp->waferPosY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st29.str();

  std::ostringstream st30;
  for (unsigned int k = 0; k < phgp->cellFineX_.size(); ++k)
    st30 << "(" << phgp->cellFineX_[k] << ", " << phgp->cellFineY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st30.str();

  std::ostringstream st31;
  for (unsigned int k = 0; k < phgp->cellCoarseX_.size(); ++k)
    st31 << "(" << phgp->cellCoarseX_[k] << ", " << phgp->cellCoarseY_[k] << ") ";
  edm::LogVerbatim("HGCalGeom") << st31.str();

  std::ostringstream st32;
  for (auto it : phgp->boundR_)
    st32 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st32.str();

  std::ostringstream st33;
  for (auto it : phgp->moduleLayS_)
    st33 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st33.str();

  std::ostringstream st34;
  for (auto it : phgp->moduleLayR_)
    st34 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st34.str();

  std::ostringstream st35;
  for (auto it : phgp->layer_)
    st35 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st35.str();

  std::ostringstream st36;
  for (auto it : phgp->layerIndex_)
    st36 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st36.str();

  std::ostringstream st37;
  for (auto it : phgp->layerGroup_)
    st37 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st37.str();

  std::ostringstream st38;
  for (auto it : phgp->cellFactor_)
    st38 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st38.str();

  std::ostringstream st39;
  for (auto it : phgp->depth_)
    st39 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st39.str();

  std::ostringstream st40;
  for (auto it : phgp->depthIndex_)
    st40 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st40.str();

  std::ostringstream st41;
  for (auto it : phgp->depthLayerF_)
    st41 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st41.str();

  std::ostringstream st42;
  for (auto it : phgp->waferCopy_)
    st42 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st42.str();

  std::ostringstream st43;
  for (auto it : phgp->waferTypeL_)
    st43 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st43.str();

  std::ostringstream st44;
  for (auto it : phgp->waferTypeT_)
    st44 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st44.str();

  std::ostringstream st45;
  for (auto it : phgp->layerGroupM_)
    st45 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st45.str();

  std::ostringstream st46;
  for (auto it : phgp->layerGroupO_)
    st46 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st46.str();

  std::ostringstream st47;
  for (auto it : phgp->trformIndex_)
    st47 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st47.str();

  std::ostringstream st48;
  for (auto it : phgp->slopeMin_)
    st48 << it << ", ";
  edm::LogVerbatim("HGCalGeom") << st48.str();

  edm::LogVerbatim("HGCalGeom") << phgp->waferR_;
  edm::LogVerbatim("HGCalGeom") << phgp->nCells_;
  edm::LogVerbatim("HGCalGeom") << phgp->nSectors_;
  edm::LogVerbatim("HGCalGeom") << phgp->mode_;
}

DEFINE_FWK_MODULE(HGCalParametersAnalyzer);
