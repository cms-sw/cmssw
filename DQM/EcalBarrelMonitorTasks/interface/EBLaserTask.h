#ifndef EBLaserTask_H
#define EBLaserTask_H

/*
 * \file EBLaserTask.h
 *
 * $Date: 2008/02/29 15:03:52 $
 * $Revision: 1.29 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBLaserTask: public edm::EDAnalyzer{

public:

/// Constructor
EBLaserTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBLaserTask();

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
