#ifndef EELedTask_H
#define EELedTask_H

/*
 * \file EELedTask.h
 *
 * $Date: 2008/03/01 13:33:52 $
 * $Revision: 1.6 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

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

DQMStore* dqmStore_;

bool enableCleanup_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EEDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;

MonitorElement* meShapeMapL1A_[18];
MonitorElement* meAmplMapL1A_[18];
MonitorElement* meTimeMapL1A_[18];
MonitorElement* meAmplPNMapL1A_[18];
MonitorElement* meShapeMapL1B_[18];
MonitorElement* meAmplMapL1B_[18];
MonitorElement* meTimeMapL1B_[18];
MonitorElement* meAmplPNMapL1B_[18];
MonitorElement* mePnAmplMapG01L1_[18];
MonitorElement* mePnPedMapG01L1_[18];
MonitorElement* mePnAmplMapG16L1_[18];
MonitorElement* mePnPedMapG16L1_[18];

MonitorElement* meShapeMapL2A_[18];
MonitorElement* meAmplMapL2A_[18];
MonitorElement* meTimeMapL2A_[18];
MonitorElement* meAmplPNMapL2A_[18];
MonitorElement* meShapeMapL2B_[18];
MonitorElement* meAmplMapL2B_[18];
MonitorElement* meTimeMapL2B_[18];
MonitorElement* meAmplPNMapL2B_[18];
MonitorElement* mePnAmplMapG01L2_[18];
MonitorElement* mePnPedMapG01L2_[18];
MonitorElement* mePnAmplMapG16L2_[18];
MonitorElement* mePnPedMapG16L2_[18];

bool init_;

};

#endif
