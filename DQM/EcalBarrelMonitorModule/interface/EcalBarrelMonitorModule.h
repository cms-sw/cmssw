#ifndef EcalBarrelMonitorModule_H
#define EcalBarrelMonitorModule_H

/*
 * \file EcalBarrelMonitorModule.h
 *
 * $Date: 2006/06/27 20:56:20 $
 * $Revision: 1.37 $
 * \author G. Della Ricca
 *
*/

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

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

int evtType_;
int runType_;

int ievt_;
int irun_;

bool verbose_;

bool enableMonitorDaemon_;

bool enableEventDisplay_;

DaqMonitorBEInterface* dbe_;

MonitorElement* meStatus_;

MonitorElement* meRun_;
MonitorElement* meEvt_;

MonitorElement* meEvtType_;
MonitorElement* meRunType_;

MonitorElement* meEBDCC_;

MonitorElement* meEBdigi_;
MonitorElement* meEBhits_;

MonitorElement* meEvent_[36];

bool init_;

std::string outputFile_;

};

#endif
