#ifndef EETrendTask_H
#define EETrendTask_H

/*
 * \file EETrendTask.h
 *
 * $Date: 2012/04/27 13:46:13 $
 * $Revision: 1.9 $
 * \author Dongwook Jang, Soon Yung Jun
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  edm::InputTag EcalPnDiodeDigiCollection_;
  edm::InputTag EcalRecHitCollection_;
  edm::InputTag EcalTrigPrimDigiCollection_;
  edm::InputTag BasicClusterCollection_;
  edm::InputTag SuperClusterCollection_;
  edm::InputTag EEDetIdCollection0_;
  edm::InputTag EEDetIdCollection1_;
  edm::InputTag EEDetIdCollection2_;
  edm::InputTag EEDetIdCollection3_;
  edm::InputTag EEDetIdCollection4_;
  edm::InputTag EcalElectronicsIdCollection1_;
  edm::InputTag EcalElectronicsIdCollection2_;
  edm::InputTag EcalElectronicsIdCollection3_;
  edm::InputTag EcalElectronicsIdCollection4_;
  edm::InputTag EcalElectronicsIdCollection5_;
  edm::InputTag EcalElectronicsIdCollection6_;
  edm::InputTag FEDRawDataCollection_;
  edm::InputTag EESRFlagCollection_;

  MonitorElement* nEEDigiMinutely_;
  MonitorElement* nEcalPnDiodeDigiMinutely_;
  MonitorElement* nEcalRecHitMinutely_;
  MonitorElement* nEcalTrigPrimDigiMinutely_;
  MonitorElement* nBasicClusterMinutely_;
  MonitorElement* nBasicClusterSizeMinutely_;
  MonitorElement* nSuperClusterMinutely_;
  MonitorElement* nSuperClusterSizeMinutely_;
  MonitorElement* nIntegrityErrorMinutely_;
  MonitorElement* nFEDEEminusRawDataMinutely_;
  MonitorElement* nFEDEEplusRawDataMinutely_;
  MonitorElement* nEESRFlagMinutely_;

  MonitorElement* nEEDigiHourly_;
  MonitorElement* nEcalPnDiodeDigiHourly_;
  MonitorElement* nEcalRecHitHourly_;
  MonitorElement* nEcalTrigPrimDigiHourly_;
  MonitorElement* nBasicClusterHourly_;
  MonitorElement* nBasicClusterSizeHourly_;
  MonitorElement* nSuperClusterHourly_;
  MonitorElement* nSuperClusterSizeHourly_;
  MonitorElement* nIntegrityErrorHourly_;
  MonitorElement* nFEDEEminusRawDataHourly_;
  MonitorElement* nFEDEEplusRawDataHourly_;
  MonitorElement* nEESRFlagHourly_;

  bool init_;

  int start_time_;
  int current_time_;
  int last_time_;

};

#endif
