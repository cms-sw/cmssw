#ifndef EELedTask_H
#define EELedTask_H

/*
 * \file EELedTask.h
 *
 * $Date: 2007/07/02 19:51:04 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EELedTask: public edm::EDAnalyzer{

public:

/// Constructor
EELedTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EELedTask();

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

DaqMonitorBEInterface* dbe_;

bool enableCleanup_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EEDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;

MonitorElement* meShapeMapA_[18];
MonitorElement* meAmplMapA_[18];
MonitorElement* meTimeMapA_[18];
MonitorElement* meAmplPNMapA_[18];
MonitorElement* meShapeMapB_[18];
MonitorElement* meAmplMapB_[18];
MonitorElement* meTimeMapB_[18];
MonitorElement* meAmplPNMapB_[18];
MonitorElement* mePnAmplMapG01_[18];
MonitorElement* mePnPedMapG01_[18];
MonitorElement* mePnAmplMapG16_[18];
MonitorElement* mePnPedMapG16_[18];

bool init_;

};

#endif
