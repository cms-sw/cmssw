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
  bool fromDD4Hep_;
};

PHGCalParametersDBBuilder::PHGCalParametersDBBuilder(const edm::ParameterSet& iC) {
  name_ = iC.getParameter<std::string>("name");
  name2_ = iC.getParameter<std::string>("name2");
  namew_ = iC.getParameter<std::string>("nameW");
  namec_ = iC.getParameter<std::string>("nameC");
  namet_ = iC.getParameter<std::string>("nameT");
  fromDD4Hep_ = iC.getParameter<bool>("fromDD4Hep");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersESModule for " << name_ << ":" << name2_ << ":" << namew_ << ":"
                                << namec_ << ":" << namet_ << " and fromDD4Hep flag " << fromDD4Hep_;
#endif
}

void PHGCalParametersDBBuilder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "HGCalEESensitive");
  desc.add<std::string>("name2", "HGCalEE");
  desc.add<std::string>("nameW", "HGCalEEWafer");
  desc.add<std::string>("nameC", "HGCalEECell");
  desc.add<std::string>("nameT", "HGCal");
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("HGCalEEParametersWriter", desc);
}

void PHGCalParametersDBBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  PHGCalParameters* phgp = new PHGCalParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("PHGCalParametersDBBuilder") << "PoolDBOutputService unavailable";
    return;
  }

  HGCalParameters* ptp = new HGCalParameters(name_);
  HGCalParametersFromDD builder;
  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "PHGCalParametersDBBuilder::Try to access cm::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv;
    es.get<IdealGeometryRecord>().get(cpv);
    builder.build(cpv.product(), *ptp, name_, namew_, namec_, namet_, name2_);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "PHGCalParametersDBBuilder::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv;
    es.get<IdealGeometryRecord>().get(cpv);
    builder.build(cpv.product(), *ptp, name_, namew_, namec_, namet_);
  }
  swapParameters(ptp, phgp);
  delete ptp;

  if (mydbservice->isNewTagRequest("PHGCalParametersRcd")) {
    mydbservice->createNewIOV<PHGCalParameters>(
        phgp, mydbservice->beginOfTime(), mydbservice->endOfTime(), "PHGCalParametersRcd");
  } else {
    edm::LogError("PHGCalParametersDBBuilder") << "PHGCalParameters and PHGCalParametersRcd Tag already present";
  }
}

void PHGCalParametersDBBuilder::swapParameters(HGCalParameters* ptp, PHGCalParameters* phgp) {
  phgp->name_ = ptp->name_;
  phgp->cellSize_.swap(ptp->cellSize_);
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
  phgp->trformIndex_.swap(ptp->trformIndex_);
  phgp->slopeMin_.swap(ptp->slopeMin_);
  phgp->waferR_ = ptp->waferR_;
  phgp->nCells_ = ptp->nCells_;
  phgp->nSectors_ = ptp->nSectors_;
  phgp->mode_ = ptp->mode_;
}

DEFINE_FWK_MODULE(PHGCalParametersDBBuilder);
