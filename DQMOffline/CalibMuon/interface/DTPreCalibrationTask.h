#ifndef DTPreCalibrationTask_H
#define DTPreCalibrationTask_H

/** \class DTPreCalibrationTask
 *  Analysis on DT digis (TB + occupancy) before the calibration step
 *
 *
 *  \author G. Mila - INFN Torino
 */

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include <FWCore/Framework/interface/ESHandle.h>

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include <DataFormats/DTDigi/interface/DTDigi.h>

#include <map>
#include <string>
#include <vector>


class DTPreCalibrationTask : public DQMEDAnalyzer {
public:
  /// Constructor
  DTPreCalibrationTask(const edm::ParameterSet &ps);

  /// Destructor
  ~DTPreCalibrationTask() override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  /// Book histos
  void bookTimeBoxes(DQMStore::IBooker &, int wheel, int sector);
  void bookOccupancyPlot(DQMStore::IBooker &, int wheel, int sector);

private:
  edm::EDGetTokenT<DTDigiCollection> digiLabel;
  int minTriggerWidth;
  int maxTriggerWidth;
  std::string folderName;

  // Time boxes map
  std::map<std::pair<int, int>, MonitorElement *> TimeBoxes;

  // Occupancy plot map
  std::map<std::pair<int, int>, MonitorElement *> OccupancyHistos;
};
#endif
