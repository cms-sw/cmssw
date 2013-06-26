#ifndef EBTrendTask_H
#define EBTrendTask_H

/*
 * \file EBTrendTask.h
 *
 * $Date: 2010/03/28 09:13:48 $
 * $Revision: 1.4 $
 * \author Dongwook Jang, Soon Yung Jun
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBTrendTask: public edm::EDAnalyzer{

 public:

  // Constructor
  EBTrendTask(const edm::ParameterSet& ps);

  // Destructor
  virtual ~EBTrendTask();

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

  edm::InputTag EBDigiCollection_;
  edm::InputTag EcalPnDiodeDigiCollection_;
  edm::InputTag EcalRecHitCollection_;
  edm::InputTag EcalTrigPrimDigiCollection_;
  edm::InputTag BasicClusterCollection_;
  edm::InputTag SuperClusterCollection_;
  edm::InputTag EBDetIdCollection0_;
  edm::InputTag EBDetIdCollection1_;
  edm::InputTag EBDetIdCollection2_;
  edm::InputTag EBDetIdCollection3_;
  edm::InputTag EBDetIdCollection4_;
  edm::InputTag EcalElectronicsIdCollection1_;
  edm::InputTag EcalElectronicsIdCollection2_;
  edm::InputTag EcalElectronicsIdCollection3_;
  edm::InputTag EcalElectronicsIdCollection4_;
  edm::InputTag EcalElectronicsIdCollection5_;
  edm::InputTag EcalElectronicsIdCollection6_;
  edm::InputTag FEDRawDataCollection_;
  edm::InputTag EBSRFlagCollection_;

  MonitorElement* nEBDigiMinutely_;
  MonitorElement* nEcalPnDiodeDigiMinutely_;
  MonitorElement* nEcalRecHitMinutely_;
  MonitorElement* nEcalTrigPrimDigiMinutely_;
  MonitorElement* nBasicClusterMinutely_;
  MonitorElement* nBasicClusterSizeMinutely_;
  MonitorElement* nSuperClusterMinutely_;
  MonitorElement* nSuperClusterSizeMinutely_;
  MonitorElement* nIntegrityErrorMinutely_;
  MonitorElement* nFEDEBRawDataMinutely_;
  MonitorElement* nEBSRFlagMinutely_;

  MonitorElement* nEBDigiHourly_;
  MonitorElement* nEcalPnDiodeDigiHourly_;
  MonitorElement* nEcalRecHitHourly_;
  MonitorElement* nEcalTrigPrimDigiHourly_;
  MonitorElement* nBasicClusterHourly_;
  MonitorElement* nBasicClusterSizeHourly_;
  MonitorElement* nSuperClusterHourly_;
  MonitorElement* nSuperClusterSizeHourly_;
  MonitorElement* nIntegrityErrorHourly_;
  MonitorElement* nFEDEBRawDataHourly_;
  MonitorElement* nEBSRFlagHourly_;

  bool init_;

  int start_time_;
  int current_time_;
  int last_time_;

};

#endif
