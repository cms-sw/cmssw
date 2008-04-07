#ifndef EcalBarrelMonitorModule_H
#define EcalBarrelMonitorModule_H

/*
 * \file EcalBarrelMonitorModule.h
 *
 * $Date: 2008/04/07 07:24:33 $
 * $Revision: 1.53 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class MonitorElement;
class DQMStore;

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
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

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

int ievt_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EBDigiCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag EcalTrigPrimDigiCollection_;

bool verbose_;
bool debug_;

bool enableEventDisplay_;

DQMStore* dbe_;

bool enableCleanup_;

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
