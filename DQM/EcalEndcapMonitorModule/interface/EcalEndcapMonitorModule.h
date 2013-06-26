#ifndef EcalEndcapMonitorModule_H
#define EcalEndcapMonitorModule_H

/*
 * \file EcalEndcapMonitorModule.h
 *
 * $Date: 2012/04/26 07:45:14 $
 * $Revision: 1.23 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class MonitorElement;
class DQMStore;

class EcalEndcapMonitorModule: public edm::EDAnalyzer{

public:

/// Constructor
EcalEndcapMonitorModule(const edm::ParameterSet& ps);

/// Destructor
virtual ~EcalEndcapMonitorModule();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(void);

// EndJob
void endJob(void);

/// Setup
void setup(void);

/// BeginRun
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// Reset
void reset(void);

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

edm::InputTag EcalRawDataCollection_;
edm::InputTag EEDigiCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag EcalTrigPrimDigiCollection_;

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

MonitorElement* meEEDCC_;

MonitorElement* meEEdigis_[2];
MonitorElement* meEEhits_[2];
MonitorElement* meEEtpdigis_[2];

MonitorElement* meEvent_[18];

bool init_;

};

#endif
