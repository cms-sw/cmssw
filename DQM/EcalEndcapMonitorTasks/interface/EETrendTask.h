#ifndef EETrendTask_H
#define EETrendTask_H

/*
 * \file EETrendTask.h
 *
 * $Date: 2010/02/08 21:35:06 $
 * $Revision: 1.3 $
 * \author Dongwook Jang, Soon Yung Jun
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TProfile.h"

class MonitorElement;
class DQMStore;

class EETrendTask: public edm::EDAnalyzer{

 public:

  // Constructor
  EETrendTask(const edm::ParameterSet& ps);

  // Destructor
  virtual ~EETrendTask();

 protected:

  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  // BeginJob
  void beginJob(void);

  // EndJob
  void endJob(void);

  // BeginRun
  void beginRun(const edm::Run & r, const edm::EventSetup & c);

  // EndRun
  void endRun(const edm::Run & r, const edm::EventSetup & c);

  // Reset
  void reset(void);

  // Setup
  void setup(void);

  // Cleanup
  void cleanup(void);

  // Update time check
  void updateTime(void);



 private:

  int ievt_;

  DQMStore* dqmStore_;

  std::string prefixME_;

  bool enableCleanup_;

  bool mergeRuns_;

  bool verbose_;

  edm::InputTag EEDigiCollection_;
  edm::InputTag EcalRecHitCollection_;
  edm::InputTag BasicClusterCollection_;
  edm::InputTag SuperClusterCollection_;
  edm::InputTag FEDRawDataCollection_;

  MonitorElement* nEEDigiMinutely_;
  MonitorElement* nEcalRecHitMinutely_;
  MonitorElement* nBasicClusterMinutely_;
  MonitorElement* nBasicClusterSizeMinutely_;
  MonitorElement* nSuperClusterMinutely_;
  MonitorElement* nSuperClusterSizeMinutely_;
  MonitorElement* nFEDEEminusRawDataMinutely_;
  MonitorElement* nFEDEEplusRawDataMinutely_;

  MonitorElement* nEEDigiHourly_;
  MonitorElement* nEcalRecHitHourly_;
  MonitorElement* nBasicClusterHourly_;
  MonitorElement* nBasicClusterSizeHourly_;
  MonitorElement* nSuperClusterHourly_;
  MonitorElement* nSuperClusterSizeHourly_;
  MonitorElement* nFEDEEminusRawDataHourly_;
  MonitorElement* nFEDEEplusRawDataHourly_;

  bool init_;

  int start_time_;
  int current_time_;
  int last_time_;

};

#endif
