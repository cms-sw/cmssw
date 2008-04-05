#ifndef EEOccupancyTask_H
#define EEOccupancyTask_H

/*
 * \file EEOccupancyTask.h
 *
 * $Date: 2008/02/29 15:07:51 $
 * $Revision: 1.16 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EEOccupancyTask: public edm::EDAnalyzer{

public:

/// Constructor
EEOccupancyTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEOccupancyTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dbe_;

bool enableCleanup_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EEDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag EcalTrigPrimDigiCollection_;

MonitorElement* meEvent_[18];
MonitorElement* meOccupancy_[18];
MonitorElement* meOccupancyMem_[18];

MonitorElement* meEEDigiOccupancy_[2];
MonitorElement* meEEDigiOccupancyProR_[2];
MonitorElement* meEEDigiOccupancyProPhi_[2];
MonitorElement* meEERecHitOccupancy_[2];
MonitorElement* meEERecHitOccupancyProR_[2];
MonitorElement* meEERecHitOccupancyProPhi_[2];
MonitorElement* meEERecHitOccupancyThr_[2];
MonitorElement* meEERecHitOccupancyProRThr_[2];
MonitorElement* meEERecHitOccupancyProPhiThr_[2];
MonitorElement* meEETrigPrimDigiOccupancy_[2];
MonitorElement* meEETrigPrimDigiOccupancyProR_[2];
MonitorElement* meEETrigPrimDigiOccupancyProPhi_[2];
MonitorElement* meEETrigPrimDigiOccupancyThr_[2];
MonitorElement* meEETrigPrimDigiOccupancyProRThr_[2];
MonitorElement* meEETrigPrimDigiOccupancyProPhiThr_[2];
MonitorElement* meEETestPulseDigiOccupancy_[2];
MonitorElement* meEELaserDigiOccupancy_[2];
MonitorElement* meEELedDigiOccupancy_[2];
MonitorElement* meEEPedestalDigiOccupancy_[2];

float recHitEnergyMin_;
float trigPrimEtMin_;

bool init_;

};

#endif
