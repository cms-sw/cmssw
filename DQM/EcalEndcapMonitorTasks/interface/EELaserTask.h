#ifndef EELaserTask_H
#define EELaserTask_H

/*
 * \file EELaserTask.h
 *
 * $Date: 2012/04/27 13:46:13 $
 * $Revision: 1.18 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

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
edm::InputTag EEDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;
std::vector<int> laserWavelengths_;

MonitorElement* meShapeMapL1_[18];
MonitorElement* meAmplMapL1_[18];
MonitorElement* meTimeMapL1_[18];
MonitorElement* meAmplPNMapL1_[18];
MonitorElement* mePnAmplMapG01L1_[18];
MonitorElement* mePnPedMapG01L1_[18];
MonitorElement* mePnAmplMapG16L1_[18];
MonitorElement* mePnPedMapG16L1_[18];

MonitorElement* meShapeMapL2_[18];
MonitorElement* meAmplMapL2_[18];
MonitorElement* meTimeMapL2_[18];
MonitorElement* meAmplPNMapL2_[18];
MonitorElement* mePnAmplMapG01L2_[18];
MonitorElement* mePnPedMapG01L2_[18];
MonitorElement* mePnAmplMapG16L2_[18];
MonitorElement* mePnPedMapG16L2_[18];

MonitorElement* meShapeMapL3_[18];
MonitorElement* meAmplMapL3_[18];
MonitorElement* meTimeMapL3_[18];
MonitorElement* meAmplPNMapL3_[18];
MonitorElement* mePnAmplMapG01L3_[18];
MonitorElement* mePnPedMapG01L3_[18];
MonitorElement* mePnAmplMapG16L3_[18];
MonitorElement* mePnPedMapG16L3_[18];

MonitorElement* meShapeMapL4_[18];
MonitorElement* meAmplMapL4_[18];
MonitorElement* meTimeMapL4_[18];
MonitorElement* meAmplPNMapL4_[18];
MonitorElement* mePnAmplMapG01L4_[18];
MonitorElement* mePnPedMapG01L4_[18];
MonitorElement* mePnAmplMapG16L4_[18];
MonitorElement* mePnPedMapG16L4_[18];

 MonitorElement* meAmplSummaryMapL1_[2];
 MonitorElement* meAmplSummaryMapL2_[2];
 MonitorElement* meAmplSummaryMapL3_[2];
 MonitorElement* meAmplSummaryMapL4_[2];

bool init_;

};

#endif
