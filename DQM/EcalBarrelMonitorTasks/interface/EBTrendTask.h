#ifndef EBTrendTask_H
#define EBTrendTask_H

/*
 * \file EBTrendTask.h
 *
 * \author Dongwook Jang, Soon Yung Jun
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

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

  edm::EDGetTokenT<EBDigiCollection> EBDigiCollection_;
  edm::EDGetTokenT<EcalPnDiodeDigiCollection> EcalPnDiodeDigiCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> EcalRecHitCollection_;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> EcalTrigPrimDigiCollection_;
  edm::EDGetTokenT<reco::BasicClusterCollection> BasicClusterCollection_;
  edm::EDGetTokenT<reco::SuperClusterCollection> SuperClusterCollection_;
  edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection0_;
  edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection1_;
  edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection2_;
  edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection3_;
  edm::EDGetTokenT<EBDetIdCollection> EBDetIdCollection4_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection1_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection2_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection3_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection4_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection5_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> EcalElectronicsIdCollection6_;
  edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_;
  edm::EDGetTokenT<EBSrFlagCollection> EBSRFlagCollection_;

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
