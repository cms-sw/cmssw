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

/*** DQM ***/
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

/*** Records for ESWatcher ***/
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

/*** MillePede ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"

class MillePedeDQMModule : public DQMEDHarvester {
  //========================== PUBLIC METHODS ==================================
public:  //====================================================================
  MillePedeDQMModule(const edm::ParameterSet&);
  ~MillePedeDQMModule() override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  //========================= PRIVATE METHODS ==================================
private:  //===================================================================
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void bookHistograms(DQMStore::IBooker&);

  void fillStatusHisto(MonitorElement* statusHisto);

  void fillExpertHistos();

  void fillExpertHisto(MonitorElement* histo,
                       const std::array<double, 6>& cut,
                       const std::array<double, 6>& sigCut,
                       const std::array<double, 6>& maxMoveCut,
                       const std::array<double, 6>& maxErrorCut,
                       const std::array<double, 6>& obs,
                       const std::array<double, 6>& obsErr);

  bool setupChanged(const edm::EventSetup&);
  int getIndexFromString(const std::string& alignableId);

  //========================== PRIVATE DATA ====================================
  //============================================================================

  const edm::ParameterSet mpReaderConfig_;
  std::unique_ptr<AlignableTracker> tracker_;
  std::unique_ptr<MillePedeFileReader> mpReader_;

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

  MonitorElement* statusResults;
  MonitorElement* binariesAvalaible;
  MonitorElement* exitCode;
};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeDQMModule);

#endif /* Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h */
