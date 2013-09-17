#ifndef EcalBarrelMonitorModule_H
#define EcalBarrelMonitorModule_H

/*
 * \file EcalBarrelMonitorModule.h
 *
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class MonitorElement;
class DQMStore;

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class EcalBarrelMonitorModule: public edm::EDAnalyzer{

public:

/// Constructor
EcalBarrelMonitorModule(const edm::ParameterSet& ps);

/// Destructor
virtual ~EcalBarrelMonitorModule();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(void);

// EndJob
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

int runNumber_;
int evtNumber_;

int runType_;
int evtType_;

bool fixedRunNumber_;

bool fixedRunType_;

bool isPhysics_;

int ievt_;

edm::EDGetTokenT<EcalRawDataCollection> EcalRawDataCollection_;
edm::EDGetTokenT<EBDigiCollection> EBDigiCollection_;
edm::EDGetTokenT<EcalRecHitCollection> EcalRecHitCollection_;
edm::EDGetTokenT<EcalTrigPrimDigiCollection> EcalTrigPrimDigiCollection_;

bool verbose_;
bool debug_;

bool enableEventDisplay_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

MonitorElement* meStatus_;

MonitorElement* meRun_;
MonitorElement* meEvt_;

MonitorElement* meRunType_;
MonitorElement* meEvtType_;

MonitorElement* meEBDCC_;

MonitorElement* meEBdigis_[2];
MonitorElement* meEBhits_[2];
MonitorElement* meEBtpdigis_[2];

MonitorElement* meEvent_[36];

bool init_;

};

#endif
