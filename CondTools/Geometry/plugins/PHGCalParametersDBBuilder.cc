#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/PHGCalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalParametersFromDD.h"

//#define EDM_ML_DEBUG

class PHGCalParametersDBBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  PHGCalParametersDBBuilder(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  void swapParameters(HGCalParameters*, PHGCalParameters*);

  std::string name_, name2_, namew_, namec_, namet_;
  bool fromDD4hep_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4HepCompactViewToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;
};

PHGCalParametersDBBuilder::PHGCalParametersDBBuilder(const edm::ParameterSet& iC) {
  name_ = iC.getParameter<std::string>("name");
  name2_ = iC.getParameter<std::string>("name2");
  namew_ = iC.getParameter<std::string>("nameW");
  namec_ = iC.getParameter<std::string>("nameC");
  namet_ = iC.getParameter<std::string>("nameT");
  fromDD4hep_ = iC.getParameter<bool>("fromDD4hep");
  dd4HepCompactViewToken_ = esConsumes<edm::Transition::BeginRun>();
  compactViewToken_ = esConsumes<edm::Transition::BeginRun>();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersESModule for " << name_ << ":" << name2_ << ":" << namew_ << ":"
                                << namec_ << ":" << namet_ << " and fromDD4hep flag " << fromDD4hep_;
#endif
}

void PHGCalParametersDBBuilder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "HGCalEESensitive");
  desc.add<std::string>("name2", "HGCalEE");
  desc.add<std::string>("nameW", "HGCalEEWafer");
  desc.add<std::string>("nameC", "HGCalEECell");
  desc.add<std::string>("nameT", "HGCal");
  desc.add<bool>("fromDD4hep", false);
  descriptions.add("HGCalEEParametersWriter", desc);
}

void PHGCalParametersDBBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  PHGCalParameters phgp;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("PHGCalParametersDBBuilder") << "PoolDBOutputService unavailable";
    return;
  }

  HGCalParameters* ptp = new HGCalParameters(name_);
  HGCalParametersFromDD builder;
  if (fromDD4hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "PHGCalParametersDBBuilder::Try to access cm::DDCompactView";
#endif
    auto cpv = es.getTransientHandle(dd4HepCompactViewToken_);
    builder.build(cpv.product(), *ptp, name_, namew_, namec_, namet_, name2_);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "PHGCalParametersDBBuilder::Try to access DDCompactView";
#endif
    auto cpv = es.getTransientHandle(compactViewToken_);
    builder.build(cpv.product(), *ptp, name_, namew_, namec_, namet_);
  }
  swapParameters(ptp, &phgp);
  delete ptp;

  if (mydbservice->isNewTagRequest("PHGCalParametersRcd")) {
    mydbservice->createOneIOV(phgp, mydbservice->beginOfTime(), "PHGCalParametersRcd");
  } else {
    edm::LogError("PHGCalParametersDBBuilder") << "PHGCalParameters and PHGCalParametersRcd Tag already present";
  }
}

void PHGCalParametersDBBuilder::swapParameters(HGCalParameters* ptp, PHGCalParameters* phgp) {
  phgp->name_ = ptp->name_;
  phgp->cellSize_.swap(ptp->cellSize_);
  phgp->slopeMin_.swap(ptp->slopeMin_);
  phgp->zFrontMin_.swap(ptp->zFrontMin_);
  phgp->rMinFront_.swap(ptp->rMinFront_);
  phgp->slopeTop_.swap(ptp->slopeTop_);
  phgp->zFrontTop_.swap(ptp->zFrontTop_);
  phgp->rMaxFront_.swap(ptp->rMaxFront_);
  phgp->zRanges_.swap(ptp->zRanges_);
  phgp->moduleBlS_.swap(ptp->moduleBlS_);
  phgp->moduleTlS_.swap(ptp->moduleTlS_);
  phgp->moduleHS_.swap(ptp->moduleHS_);
  phgp->moduleDzS_.swap(ptp->moduleDzS_);
  phgp->moduleAlphaS_.swap(ptp->moduleAlphaS_);
  phgp->moduleCellS_.swap(ptp->moduleCellS_);
  phgp->moduleBlR_.swap(ptp->moduleBlR_);
  phgp->moduleTlR_.swap(ptp->moduleTlR_);
  phgp->moduleHR_.swap(ptp->moduleHR_);
  phgp->moduleDzR_.swap(ptp->moduleDzR_);
  phgp->moduleAlphaR_.swap(ptp->moduleAlphaR_);
  phgp->moduleCellR_.swap(ptp->moduleCellR_);
  phgp->trformTranX_.swap(ptp->trformTranX_);
  phgp->trformTranY_.swap(ptp->trformTranY_);
  phgp->trformTranZ_.swap(ptp->trformTranZ_);
  phgp->trformRotXX_.swap(ptp->trformRotXX_);
  phgp->trformRotYX_.swap(ptp->trformRotYX_);
  phgp->trformRotZX_.swap(ptp->trformRotZX_);
  phgp->trformRotXY_.swap(ptp->trformRotXY_);
  phgp->trformRotYY_.swap(ptp->trformRotYY_);
  phgp->trformRotZY_.swap(ptp->trformRotZY_);
  phgp->trformRotXZ_.swap(ptp->trformRotXZ_);
  phgp->trformRotYZ_.swap(ptp->trformRotYZ_);
  phgp->trformRotZZ_.swap(ptp->trformRotZZ_);
  phgp->xLayerHex_.swap(ptp->xLayerHex_);
  phgp->yLayerHex_.swap(ptp->yLayerHex_);
  phgp->zLayerHex_.swap(ptp->zLayerHex_);
  phgp->rMinLayHex_.swap(ptp->rMinLayHex_);
  phgp->rMaxLayHex_.swap(ptp->rMaxLayHex_);
  phgp->waferPosX_.swap(ptp->waferPosX_);
  phgp->waferPosY_.swap(ptp->waferPosY_);
  phgp->cellFineX_.swap(ptp->cellFineX_);
  phgp->cellFineY_.swap(ptp->cellFineY_);
  phgp->cellCoarseX_.swap(ptp->cellCoarseX_);
  phgp->cellCoarseY_.swap(ptp->cellCoarseY_);
  phgp->boundR_.swap(ptp->boundR_);
  phgp->rLimit_.swap(ptp->rLimit_);
  phgp->waferThickness_.swap(ptp->waferThickness_);
  phgp->cellThickness_.swap(ptp->cellThickness_);
  phgp->radius100to200_.swap(ptp->radius100to200_);
  phgp->radius200to300_.swap(ptp->radius200to300_);
  phgp->radiusMixBoundary_.swap(ptp->radiusMixBoundary_);
  phgp->rMinLayerBH_.swap(ptp->rMinLayerBH_);
  phgp->radiusLayer_[0].swap(ptp->radiusLayer_[0]);
  phgp->radiusLayer_[1].swap(ptp->radiusLayer_[1]);
  phgp->cassetteShift_.swap(ptp->cassetteShift_);
  phgp->cassetteShiftTile_.swap(ptp->cassetteShiftTile_);
  phgp->cassetteRetractTile_.swap(ptp->cassetteRetractTile_);
  phgp->moduleLayS_.swap(ptp->moduleLayS_);
  phgp->moduleLayR_.swap(ptp->moduleLayR_);
  phgp->layer_.swap(ptp->layer_);
  phgp->layerIndex_.swap(ptp->layerIndex_);
  phgp->layerGroup_.swap(ptp->layerGroup_);
  phgp->cellFactor_.swap(ptp->cellFactor_);
  phgp->depth_.swap(ptp->depth_);
  phgp->depthIndex_.swap(ptp->depthIndex_);
  phgp->depthLayerF_.swap(ptp->depthLayerF_);
  phgp->waferCopy_.swap(ptp->waferCopy_);
  phgp->waferTypeL_.swap(ptp->waferTypeL_);
  phgp->waferTypeT_.swap(ptp->waferTypeT_);
  phgp->layerGroupM_.swap(ptp->layerGroupM_);
  phgp->layerGroupO_.swap(ptp->layerGroupO_);
  phgp->cellFine_.swap(ptp->cellFine_);
  phgp->cellCoarse_.swap(ptp->cellCoarse_);
  phgp->levelT_.swap(ptp->levelT_);
  phgp->waferUVMaxLayer_.swap(ptp->waferUVMaxLayer_);
  phgp->nPhiBinBH_.swap(ptp->nPhiBinBH_);
  phgp->layerFrontBH_.swap(ptp->layerFrontBH_);
  phgp->iradMinBH_.swap(ptp->iradMinBH_);
  phgp->iradMaxBH_.swap(ptp->iradMaxBH_);
  phgp->iradMinBHFine_.swap(ptp->iradMinBHFine_);
  phgp->iradMaxBHFine_.swap(ptp->iradMaxBHFine_);
  phgp->firstModule_.swap(ptp->firstModule_);
  phgp->lastModule_.swap(ptp->lastModule_);
  phgp->layerType_.swap(ptp->layerType_);
  phgp->layerCenter_.swap(ptp->layerCenter_);
  phgp->nPhiLayer_.swap(ptp->nPhiLayer_);
  phgp->calibCellFullHD_.swap(ptp->calibCellFullHD_);
  phgp->calibCellPartHD_.swap(ptp->calibCellPartHD_);
  phgp->calibCellFullLD_.swap(ptp->calibCellFullLD_);
  phgp->calibCellPartLD_.swap(ptp->calibCellPartLD_);
  phgp->trformIndex_.swap(ptp->trformIndex_);
  phgp->cellFineHalf_.swap(ptp->cellFineHalf_);
  phgp->cellCoarseHalf_.swap(ptp->cellCoarseHalf_);
  phgp->waferR_ = ptp->waferR_;
  phgp->waferSize_ = ptp->waferSize_;
  phgp->waferThick_ = ptp->waferThick_;
  phgp->sensorSeparation_ = ptp->sensorSeparation_;
  phgp->sensorSizeOffset_ = ptp->sensorSizeOffset_;
  phgp->guardRingOffset_ = ptp->guardRingOffset_;
  phgp->mouseBite_ = ptp->mouseBite_;
  phgp->fracAreaMin_ = ptp->fracAreaMin_;
  phgp->zMinForRad_ = ptp->zMinForRad_;
  phgp->minTileSize_ = ptp->minTileSize_;
  phgp->layerRotation_ = ptp->layerRotation_;
  phgp->calibCellRHD_ = ptp->calibCellRHD_;
  phgp->calibCellRLD_ = ptp->calibCellRLD_;
  phgp->detectorType_ = ptp->detectorType_;
  phgp->useSimWt_ = ptp->useSimWt_;
  phgp->nCells_ = ptp->nCells_;
  phgp->nSectors_ = ptp->nSectors_;
  phgp->mode_ = ptp->mode_;
  phgp->firstLayer_ = ptp->firstLayer_;
  phgp->firstMixedLayer_ = ptp->firstMixedLayer_;
  phgp->levelZSide_ = ptp->levelZSide_;
  phgp->nCellsFine_ = ptp->nCellsFine_;
  phgp->nCellsCoarse_ = ptp->nCellsCoarse_;
  phgp->useOffset_ = ptp->useOffset_;
  phgp->waferUVMax_ = ptp->waferUVMax_;
  phgp->choiceType_ = ptp->choiceType_;
  phgp->nCornerCut_ = ptp->nCornerCut_;
  phgp->layerOffset_ = ptp->layerOffset_;
  phgp->waferMaskMode_ = ptp->waferMaskMode_;
  phgp->waferZSide_ = ptp->waferZSide_;
  phgp->cassettes_ = ptp->cassettes_;
  phgp->nphiCassette_ = ptp->nphiCassette_;
  phgp->nphiFineCassette_ = ptp->nphiFineCassette_;
  phgp->phiOffset_ = ptp->phiOffset_;
  phgp->tileUVMax_ = ptp->tileUVMax_;
  phgp->tileUVMaxFine_ = ptp->tileUVMaxFine_;
  phgp->defineFull_ = ptp->defineFull_;
}

DEFINE_FWK_MODULE(PHGCalParametersDBBuilder);
