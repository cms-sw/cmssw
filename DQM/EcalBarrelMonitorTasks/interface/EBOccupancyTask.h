#ifndef EBOccupancyTask_H
#define EBOccupancyTask_H

/*
 * \file EBOccupancyTask.h
 *
 * $Date: 2010/03/12 11:36:29 $
 * $Revision: 1.25 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBOccupancyTask: public edm::EDAnalyzer{

public:

/// Constructor
EBOccupancyTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBOccupancyTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(void);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// Reset
void reset(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EBDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag EcalTrigPrimDigiCollection_;

enum runClassification { notdata, physics, testpulse, laser, pedestal }; 

MonitorElement* meEvent_[36];
MonitorElement* meOccupancy_[36];
MonitorElement* meOccupancyMem_[36];

 MonitorElement *meDCCOccupancy_;
MonitorElement* meEBDigiOccupancy_;
MonitorElement* meEBDigiOccupancyProjEta_;
MonitorElement* meEBDigiOccupancyProjPhi_;
 MonitorElement* meEBDigiNumber_;
 MonitorElement* meEBDigiNumberPerFED_;
MonitorElement* meEBRecHitOccupancy_;
MonitorElement* meEBRecHitOccupancyProjEta_;
MonitorElement* meEBRecHitOccupancyProjPhi_;
MonitorElement* meEBRecHitNumber_;
MonitorElement* meEBRecHitNumberPerFED_;
MonitorElement* meEBRecHitOccupancyThr_;
MonitorElement* meEBRecHitOccupancyProjEtaThr_;
MonitorElement* meEBRecHitOccupancyProjPhiThr_;
MonitorElement* meEBTrigPrimDigiOccupancy_;
MonitorElement* meEBTrigPrimDigiOccupancyProjEta_;
MonitorElement* meEBTrigPrimDigiOccupancyProjPhi_;
MonitorElement* meEBTrigPrimDigiOccupancyThr_;
MonitorElement* meEBTrigPrimDigiOccupancyProjEtaThr_;
MonitorElement* meEBTrigPrimDigiOccupancyProjPhiThr_;

float recHitEnergyMin_;
float trigPrimEtMin_;

bool init_;

};

#endif
