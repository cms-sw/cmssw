#ifndef DTPreCalibrationTask_H
#define DTPreCalibrationTask_H

/** \class DTPreCalibrationTask
 *  Analysis on DT digis (TB + occupancy) before the calibration step
 *
 *
 *  $Date: 2008/10/31 08:51:46 $
 *  $Revision: 1.5 $
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>

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
  void beginJob(const edm::EventSetup& c);
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Book histos
  void bookHistos(int wheel, int sector);

private:

  DQMStore* dbe;

  // Time boxes map
  std::map<std::pair<int,int>, MonitorElement* > TimeBoxes;

  // Occupancy plot map
  std::map<std::pair<int,int>, MonitorElement* > OccupancyHistos;

};
#endif
