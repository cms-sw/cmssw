#ifndef EcalEndcapMonitorModule_H
#define EcalEndcapMonitorModule_H

/*
 * \file EcalEndcapMonitorModule.h
 *
 * $Date: 2007/05/12 09:32:24 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 *
*/

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

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
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int runType_;
int evtType_;

int runNumber_;
int evtNumber_;

bool fixedRunNumber_;

int ievt_;

edm::InputTag EcalTBEventHeader_;
edm::InputTag EcalRawDataCollection_;
edm::InputTag EBDigiCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;

bool verbose_;

bool enableMonitorDaemon_;

bool enableEventDisplay_;

DaqMonitorBEInterface* dbe_;

bool enableCleanup_;

MonitorElement* meStatus_;

MonitorElement* meRun_;
MonitorElement* meEvt_;

MonitorElement* meRunType_;
MonitorElement* meEvtType_;

MonitorElement* meEEDCC_;

MonitorElement* meEEdigi_;
MonitorElement* meEEhits_;

MonitorElement* meEvent_[18];

bool init_;

std::string outputFile_;

};

#endif
