#ifndef DTPreCalibrationTask_H
#define DTPreCalibrationTask_H

/** \class DTPreCalibrationTask
 *  Analysis on DT digis (TB + occupancy) before the calibration step
 *
 *
 *  $Date: 2010/01/07 16:31:59 $
 *  $Revision: 1.4 $
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include <string>
#include <map>
#include <vector>


class DQMStore;
class MonitorElement;

class DTPreCalibrationTask: public edm::EDAnalyzer{

public:

  /// Constructor
  DTPreCalibrationTask(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTPreCalibrationTask();

  /// BeginJob
  void beginJob();
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Book histos
  void bookTimeBoxes(int wheel, int sector);
  void bookOccupancyPlot(int wheel, int sector);

  ///EndJob
  void endJob();

private:

  DQMStore* dbe;
  std::string digiLabel;
  int  minTriggerWidth;
  int  maxTriggerWidth;
  bool saveFile;
  std::string outputFileName;
  std::string folderName;

  // Time boxes map
  std::map<std::pair<int,int>, MonitorElement* > TimeBoxes;

  // Occupancy plot map
  std::map<std::pair<int,int>, MonitorElement* > OccupancyHistos;

};
#endif
