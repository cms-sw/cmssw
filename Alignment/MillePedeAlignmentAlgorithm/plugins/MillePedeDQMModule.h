#ifndef Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h
#define Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h

/**
 * @package   Alignment/MillePedeAlignmentAlgorithm
 * @file      MillePedeDQMModule.h
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      Oct 26, 2015
 *
 * @brief     DQM Plotter for PCL-Alignment
 */

/*** system includes ***/
#include <array>
#include <memory>

/*** core framework functionality ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*** Geometry ***/
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

/*** Thresholds from DB ***/
#include "CondFormats/DataRecord/interface/AlignPCLThresholdsHGRcd.h"

/*** DQM ***/
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

/*** Records for ESWatcher ***/
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

/*** MillePede ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"

class MillePedeDQMModule : public DQMEDHarvester {
  //========================== PUBLIC METHODS ==================================
public:  //====================================================================
  MillePedeDQMModule(const edm::ParameterSet&);
  ~MillePedeDQMModule() override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  enum { SIZE_LG_STRUCTS = 6, SIZE_HG_STRUCTS = 820, SIZE_INDEX = 8 };

  //========================= PRIVATE METHODS ==================================
private:  //===================================================================
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void bookHistograms(DQMStore::IBooker&);

  void fillStatusHisto(MonitorElement* statusHisto);

  void fillStatusHistoHG(MonitorElement* statusHisto);

  void fillExpertHistos();

  void fillExpertHistos_HG();

  void fillExpertHisto(MonitorElement* histo,
                       const std::array<double, SIZE_INDEX>& cut,
                       const std::array<double, SIZE_INDEX>& sigCut,
                       const std::array<double, SIZE_INDEX>& maxMoveCut,
                       const std::array<double, SIZE_INDEX>& maxErrorCut,
                       const std::array<double, SIZE_LG_STRUCTS>& obs,
                       const std::array<double, SIZE_LG_STRUCTS>& obsErr);

  void fillExpertHisto_HG(std::map<std::string, MonitorElement*>& histo_map,
                          const std::array<double, SIZE_INDEX>& cut,
                          const std::array<double, SIZE_INDEX>& sigCut,
                          const std::array<double, SIZE_INDEX>& maxMoveCut,
                          const std::array<double, SIZE_INDEX>& maxErrorCut,
                          const std::array<double, SIZE_HG_STRUCTS>& obs,
                          const std::array<double, SIZE_HG_STRUCTS>& obsErr);

  bool setupChanged(const edm::EventSetup&);
  int getIndexFromString(const std::string& alignableId);

  //========================== PRIVATE DATA ====================================
  //============================================================================

  // esConsumes
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> gDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  const edm::ESGetToken<PTrackerAdditionalParametersPerDet, PTrackerAdditionalParametersPerDetRcd> ptitpToken_;
  const edm::ESGetToken<AlignPCLThresholdsHG, AlignPCLThresholdsHGRcd> aliThrToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;

  const edm::ParameterSet mpReaderConfig_;
  std::unique_ptr<AlignableTracker> tracker_;
  std::unique_ptr<MillePedeFileReader> mpReader_;
  std::shared_ptr<PixelTopologyMap> pixelTopologyMap_;

  std::vector<std::pair<std::string, int>> layerVec;

  edm::ESWatcher<TrackerTopologyRcd> watchTrackerTopologyRcd_;
  edm::ESWatcher<IdealGeometryRecord> watchIdealGeometryRcd_;
  edm::ESWatcher<PTrackerParametersRcd> watchPTrackerParametersRcd_;

  // Histograms
  MonitorElement* h_xPos;
  MonitorElement* h_xRot;
  MonitorElement* h_yPos;
  MonitorElement* h_yRot;
  MonitorElement* h_zPos;
  MonitorElement* h_zRot;

  std::map<std::string, MonitorElement*> h_xPos_HG;
  std::map<std::string, MonitorElement*> h_xRot_HG;
  std::map<std::string, MonitorElement*> h_yPos_HG;
  std::map<std::string, MonitorElement*> h_yRot_HG;
  std::map<std::string, MonitorElement*> h_zPos_HG;
  std::map<std::string, MonitorElement*> h_zRot_HG;

  MonitorElement* statusResults;
  MonitorElement* binariesAvalaible;
  MonitorElement* exitCode;

  bool isHG_;
};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeDQMModule);

#endif /* Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h */
