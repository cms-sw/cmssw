#ifndef EELaserTask_H
#define EELaserTask_H

/*
 * \file EELaserTask.h
 *
 * $Date: 2007/04/02 16:23:12 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EELaserTask: public edm::EDAnalyzer{

public:

/// Constructor
EELaserTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EELaserTask();

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

edm::InputTag EcalRawDataCollection_;
edm::InputTag EBDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;

MonitorElement* meShapeMapL1A_[36];
MonitorElement* meAmplMapL1A_[36];
MonitorElement* meTimeMapL1A_[36];
MonitorElement* meAmplPNMapL1A_[36];
MonitorElement* meShapeMapL1B_[36];
MonitorElement* meAmplMapL1B_[36];
MonitorElement* meTimeMapL1B_[36];
MonitorElement* meAmplPNMapL1B_[36];
MonitorElement* mePnAmplMapG01L1_[36];
MonitorElement* mePnPedMapG01L1_[36];
MonitorElement* mePnAmplMapG16L1_[36];
MonitorElement* mePnPedMapG16L1_[36];

MonitorElement* meShapeMapL2A_[36];
MonitorElement* meAmplMapL2A_[36];
MonitorElement* meTimeMapL2A_[36];
MonitorElement* meAmplPNMapL2A_[36];
MonitorElement* meShapeMapL2B_[36];
MonitorElement* meAmplMapL2B_[36];
MonitorElement* meTimeMapL2B_[36];
MonitorElement* meAmplPNMapL2B_[36];
MonitorElement* mePnAmplMapG01L2_[36];
MonitorElement* mePnPedMapG01L2_[36];
MonitorElement* mePnAmplMapG16L2_[36];
MonitorElement* mePnPedMapG16L2_[36];

MonitorElement* meShapeMapL3A_[36];
MonitorElement* meAmplMapL3A_[36];
MonitorElement* meTimeMapL3A_[36];
MonitorElement* meAmplPNMapL3A_[36];
MonitorElement* meShapeMapL3B_[36];
MonitorElement* meAmplMapL3B_[36];
MonitorElement* meTimeMapL3B_[36];
MonitorElement* meAmplPNMapL3B_[36];
MonitorElement* mePnAmplMapG01L3_[36];
MonitorElement* mePnPedMapG01L3_[36];
MonitorElement* mePnAmplMapG16L3_[36];
MonitorElement* mePnPedMapG16L3_[36];

MonitorElement* meShapeMapL4A_[36];
MonitorElement* meAmplMapL4A_[36];
MonitorElement* meTimeMapL4A_[36];
MonitorElement* meAmplPNMapL4A_[36];
MonitorElement* meShapeMapL4B_[36];
MonitorElement* meAmplMapL4B_[36];
MonitorElement* meTimeMapL4B_[36];
MonitorElement* meAmplPNMapL4B_[36];
MonitorElement* mePnAmplMapG01L4_[36];
MonitorElement* mePnPedMapG01L4_[36];
MonitorElement* mePnAmplMapG16L4_[36];
MonitorElement* mePnPedMapG16L4_[36];

bool init_;

};

#endif
