/**\class Phase2OTMonitorVectorHits Phase2OTMonitorVectorHits.cc 
 Description:  DQM module to monitor Phase2 VectorHit
*/
//
// Author: Gourab Saha, Suvankar Roy Chowdhury
//
// system include files
#include <memory>
#include <map>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

class Phase2OTMonitorVectorHits : public DQMEDAnalyzer {
public:
  explicit Phase2OTMonitorVectorHits(const edm::ParameterSet&);
  ~Phase2OTMonitorVectorHits() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);

  edm::ParameterSet config_;
  const edm::EDGetTokenT<VectorHitCollection> tokenVecHitsOT_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;

  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  const MagneticField* magField_ = nullptr;

  MonitorElement* numberVecHits_ = nullptr;
  MonitorElement* globalXY_P_ = nullptr;
  MonitorElement* globalRZ_P_ = nullptr;
  MonitorElement* globalXY_S_ = nullptr;
  MonitorElement* globalRZ_S_ = nullptr;
  struct VecHitME {
    MonitorElement* numberVecHits_P = nullptr;
    MonitorElement* numberVecHits_S = nullptr;
    MonitorElement* localPosXY_P = nullptr;
    MonitorElement* localPosXY_S = nullptr;
    MonitorElement* curvature_P = nullptr;
    MonitorElement* curvature_S = nullptr;
    MonitorElement* curvErr_P = nullptr;
    MonitorElement* curvErr_S = nullptr;
    MonitorElement* curvatureVsEta_P = nullptr;
    MonitorElement* curvatureVsEta_S = nullptr;
    MonitorElement* phi_P = nullptr;
    MonitorElement* phi_S = nullptr;
    MonitorElement* eta_P = nullptr;
    MonitorElement* eta_S = nullptr;
    MonitorElement* pt_P = nullptr;
    MonitorElement* pt_S = nullptr;
    MonitorElement* chi2_P = nullptr;
    MonitorElement* chi2_S = nullptr;
  };
  std::map<std::string, VecHitME> layerMEs_;
};

//
// constructors
//
Phase2OTMonitorVectorHits::Phase2OTMonitorVectorHits(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      tokenVecHitsOT_(consumes<VectorHitCollection>(config_.getParameter<edm::InputTag>("vechitsSrc"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      magFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2OTMonitorVectorHits") << ">>> Construct Phase2OTMonitorVectorHits ";
}

//
// destructor
//
Phase2OTMonitorVectorHits::~Phase2OTMonitorVectorHits() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2OTMonitorVectorHits") << ">>> Destroy Phase2OTMonitorVectorHits ";
}
//
// -- DQM Begin Run
void Phase2OTMonitorVectorHits::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
  magField_ = &iSetup.getData(magFieldToken_);
}

//
// -- Analyze
//
void Phase2OTMonitorVectorHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the VecHits
  const auto& vechits = iEvent.getHandle(tokenVecHitsOT_);
  if (!vechits.isValid())
    return;
  std::map<std::string, unsigned int> nvechitLayerMapP;
  std::map<std::string, unsigned int> nvechitLayerMapS;
  unsigned long int nTotvechitsinevt = 0;
  // Loop over modules
  VectorHitCollection::const_iterator DSViter;
  for (DSViter = vechits->begin(); DSViter != vechits->end(); ++DSViter) {
    // Get the detector unit's id
    unsigned int rawid(DSViter->detId());
    DetId detId(rawid);
    TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
    if (mType != TrackerGeometry::ModuleType::Ph2PSP && mType != TrackerGeometry::ModuleType::Ph2SS)
      continue;
    std::string key = phase2tkutil::getOTHistoId(detId, tTopo_);
    nTotvechitsinevt += DSViter->size();
    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
      if (nvechitLayerMapP.find(key) == nvechitLayerMapP.end()) {
        nvechitLayerMapP.insert(std::make_pair(key, DSViter->size()));
      } else {
        nvechitLayerMapP[key] += DSViter->size();
      }
    } else if (mType == TrackerGeometry::ModuleType::Ph2SS) {
      if (nvechitLayerMapS.find(key) == nvechitLayerMapS.end()) {
        nvechitLayerMapS.insert(std::make_pair(key, DSViter->size()));
      } else {
        nvechitLayerMapS[key] += DSViter->size();
      }
    }

    edmNew::DetSet<VectorHit>::const_iterator vechitIt;
    for (vechitIt = DSViter->begin(); vechitIt != DSViter->end(); ++vechitIt) {
      const Global3DPoint globalPos = vechitIt->lowerGlobalPos();
      const LocalPoint lp = vechitIt->localPosition();
      const double gx = globalPos.x() * 10.;
      const double gy = globalPos.y() * 10.;
      const double gz = globalPos.z() * 10.;
      const double gr = globalPos.perp() * 10.;
      const Global3DVector globalVec = vechitIt->globalDirection();
      const float curvature = vechitIt->curvature();
      const float curverr = vechitIt->curvatureError();
      const float eta = globalVec.eta();
      const float phi = globalVec.phi();
      float QOverPT = vechitIt->transverseMomentum(magField_->inTesla(GlobalPoint(0., 0., 0.)).z());
      const int sign = QOverPT > 0. ? 1. : -1.;
      if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
        globalXY_P_->Fill(gx, gy);
        globalRZ_P_->Fill(gz, gr);
        //layer wise histo
        layerMEs_[key].localPosXY_P->Fill(lp.x(), lp.y());
        layerMEs_[key].curvature_P->Fill(curvature);
        if (curvature != 0.f)
          layerMEs_[key].curvErr_P->Fill(curverr / curvature);
        else
          edm::LogError("Phase2OTMonitorVectorHits") << "VectorHit with curvature zero found";
        layerMEs_[key].phi_P->Fill(phi);
        layerMEs_[key].eta_P->Fill(eta);
        layerMEs_[key].pt_P->Fill(sign * QOverPT);
        layerMEs_[key].chi2_P->Fill(vechitIt->chi2());
        layerMEs_[key].curvatureVsEta_P->Fill(eta, curvature);
      } else if (mType == TrackerGeometry::ModuleType::Ph2SS) {
        globalXY_S_->Fill(gx, gy);
        globalRZ_S_->Fill(gz, gr);
        //layer wise histo
        layerMEs_[key].localPosXY_S->Fill(lp.x(), lp.y());
        layerMEs_[key].curvature_S->Fill(curvature);
        if (curvature != 0.f)
          layerMEs_[key].curvErr_S->Fill(curverr / curvature);
        else
          edm::LogError("Phase2OTMonitorVectorHits") << "VectorHit with curvature zero found";
        layerMEs_[key].phi_S->Fill(phi);
        layerMEs_[key].eta_S->Fill(eta);
        layerMEs_[key].pt_S->Fill(sign * QOverPT);
        layerMEs_[key].chi2_S->Fill(vechitIt->chi2());
        layerMEs_[key].curvatureVsEta_S->Fill(eta, curvature);
      }
    }
  }
  //fill nVecHits per event
  numberVecHits_->Fill(nTotvechitsinevt);
  //fill nVecHit counter per layer
  for (auto& lme : nvechitLayerMapP) {
    layerMEs_[lme.first].numberVecHits_P->Fill(lme.second);
  }
  for (auto& lme : nvechitLayerMapS) {
    layerMEs_[lme.first].numberVecHits_S->Fill(lme.second);
  }
}
//
// -- Book Histograms
//
void Phase2OTMonitorVectorHits::bookHistograms(DQMStore::IBooker& ibooker,
                                               edm::Run const& iRun,
                                               edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  //std::stringstream folder_name;

  ibooker.cd();
  edm::LogInfo("Phase2OTMonitorVectorHits") << " Booking Histograms in : " << top_folder;
  ibooker.setCurrentFolder(top_folder);

  //Global histos for OT
  numberVecHits_ = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalNVecHits"), ibooker);

  globalXY_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_P"), ibooker);

  globalRZ_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_P"), ibooker);

  globalXY_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_S"), ibooker);

  globalRZ_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_S"), ibooker);

  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      unsigned int detId_raw = det_u->geographicalId().rawId();
      //we only need the layerwise histos for the lower layer (?)
      if (tkGeom_->getDetectorType(detId_raw) != TrackerGeometry::ModuleType::Ph2PSP &&
          tkGeom_->getDetectorType(detId_raw) != TrackerGeometry::ModuleType::Ph2SS)
        continue;
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}

//
// -- Book Layer Histograms
//
void Phase2OTMonitorVectorHits::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  std::string key = phase2tkutil::getOTHistoId(det_id, tTopo_);
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    VecHitME local_histos;
    ibooker.setCurrentFolder(subdir + "/" + key);
    edm::LogInfo("Phase2OTMonitorVectorHits") << " Booking Histograms in : " << key;
    //either PSP or SS - ensured by the call from bookHisto
    if (tkGeom_->getDetectorType(det_id) == TrackerGeometry::ModuleType::Ph2PSP) {
      local_histos.numberVecHits_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NVecHitsLayer_P"), ibooker);
      local_histos.localPosXY_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("LocalPositionXY_P"), ibooker);
      local_histos.curvature_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Curvature"), ibooker);
      local_histos.curvErr_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("CurvErr"), ibooker);
      local_histos.phi_P = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Phi"), ibooker);
      local_histos.eta_P = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Eta"), ibooker);
      local_histos.pt_P = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pt"), ibooker);
      local_histos.chi2_P = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Chi2"), ibooker);
      local_histos.curvatureVsEta_P =
          phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("CurvatureVsEta_P"), ibooker);
    } else {
      ibooker.setCurrentFolder(subdir + "/" + key);
      local_histos.numberVecHits_S =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NVecHitsLayer_P"), ibooker);
      local_histos.localPosXY_S =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("LocalPositionXY_S"), ibooker);
      local_histos.curvature_S =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Curvature"), ibooker);
      local_histos.curvErr_S =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("CurvErr"), ibooker);
      local_histos.phi_S = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Phi"), ibooker);
      local_histos.eta_S = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Eta"), ibooker);
      local_histos.pt_S = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Pt"), ibooker);
      local_histos.chi2_S = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Chi2"), ibooker);
      local_histos.curvatureVsEta_S =
          phase2tkutil::bookProfile1DFromPSet(config_.getParameter<edm::ParameterSet>("CurvatureVsEta_S"), ibooker);
    }
    layerMEs_.insert(std::make_pair(key, local_histos));
  }
}

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTMonitorVectorHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // vechitMonitorOT
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfVecHits");
    psd0.add<std::string>("title", ";Number of vechits per event;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 15000.0);
    psd0.add<int>("NxBins", 150);
    desc.add<edm::ParameterSetDescription>("GlobalNVecHits", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_VecHitPosition_XY_P");
    psd0.add<std::string>("title", "Global_VecHitPosition_XY_P;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_VecHitPosition_XY_S");
    psd0.add<std::string>("title", "Global_VecHitPosition_XY_S;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_VecHitPosition_RZ_P");
    psd0.add<std::string>("title", "Global_VecHitPosition_RZ_P;z [mm];r [mm]");
    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_VecHitPosition_RZ_S");
    psd0.add<std::string>("title", "Global_VecHitPosition_RZ_S;z [mm];r [mm]");

    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_S", psd0);
  }
  //Layer wise parameter
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfVecHitsLayerP");
    psd0.add<std::string>("title", ";Number of vector hits per event(macro pixel sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 5000.0);
    psd0.add<int>("NxBins", 100);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NVecHitsLayer_P", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfVecHitsLayerS");
    psd0.add<std::string>("title", ";Number of vector hits per event(strip sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 5000.0);
    psd0.add<int>("NxBins", 100);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NVecHitsLayer_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "LocalPositionXY_P");
    psd0.add<std::string>("title", "LocalPositionXY_P;x ;y ;");
    psd0.add<int>("NxBins", 50);
    psd0.add<double>("xmin", -10.0);
    psd0.add<double>("xmax", 10.0);
    psd0.add<int>("NyBins", 50);
    psd0.add<double>("ymin", -10.0);
    psd0.add<double>("ymax", 10.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "LocalPositionXY_S");
    psd0.add<std::string>("title", "LocalPositionXY_S;x ;y ;");
    psd0.add<int>("NxBins", 50);
    psd0.add<double>("xmin", -10.0);
    psd0.add<double>("xmax", 10.0);
    psd0.add<int>("NyBins", 50);
    psd0.add<double>("ymin", -10.0);
    psd0.add<double>("ymax", 10.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_S", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "CurvatureOfVecHits");
    psd0.add<std::string>("title", ";VectorHit curvature;");
    psd0.add<double>("xmin", -0.05);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.05);
    psd0.add<int>("NxBins", 200);
    desc.add<edm::ParameterSetDescription>("Curvature", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "CurvatureErrOCurvature");
    psd0.add<std::string>("title", ";VectorHit #delta#rho/#rho;");
    psd0.add<double>("xmin", -0.05);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.05);
    psd0.add<int>("NxBins", 500);
    desc.add<edm::ParameterSetDescription>("CurvErr", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "PhiOfVecHits");
    psd0.add<std::string>("title", ";VectorHit #phi;");
    psd0.add<double>("xmin", -M_PI);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", M_PI);
    psd0.add<int>("NxBins", 30);
    desc.add<edm::ParameterSetDescription>("Phi", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "EtaOfVecHits");
    psd0.add<std::string>("title", ";VectorHit #eta;");
    psd0.add<double>("xmin", -5.);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 5.);
    psd0.add<int>("NxBins", 50);
    desc.add<edm::ParameterSetDescription>("Eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "PtOfVecHits");
    psd0.add<std::string>("title", "VectorHit p_T;p_T ;");
    psd0.add<int>("NxBins", 100);
    psd0.add<double>("xmin", 0.);
    psd0.add<double>("xmax", 200.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("Pt", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Chi2OfVecHits");
    psd0.add<std::string>("title", "VectorHit chi squared; #chi^2;");
    psd0.add<int>("NxBins", 100);
    psd0.add<double>("xmin", 0.);
    psd0.add<double>("xmax", 0.000001);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("Chi2", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "CurvatureVsEtaProf_P");
    psd0.add<std::string>("title", "Curvature vs #eta (macro-pixel);#eta ;curvature ;");
    psd0.add<int>("NxBins", 50);
    psd0.add<double>("xmin", -5.0);
    psd0.add<double>("xmax", 5.0);
    psd0.add<double>("ymin", -0.05);
    psd0.add<double>("ymax", 0.05);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("CurvatureVsEta_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "CurvatureVsEtaProf_S");
    psd0.add<std::string>("title", "Curvature vs #eta (strip);#eta ;curvature ;");
    psd0.add<int>("NxBins", 25);
    psd0.add<double>("xmin", -5.0);
    psd0.add<double>("xmax", 5.0);
    psd0.add<double>("ymin", -0.05);
    psd0.add<double>("ymax", 0.05);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("CurvatureVsEta_S", psd0);
  }
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTVectorHits/Accepted");
  desc.add<bool>("Verbosity", false);
  desc.add<edm::InputTag>("vechitsSrc", edm::InputTag("siPhase2VectorHits", "accepted"));
  descriptions.add("Phase2OTMonitorVectorHits", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2OTMonitorVectorHits);
