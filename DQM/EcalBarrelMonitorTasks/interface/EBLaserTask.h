#ifndef EBLaserTask_H
#define EBLaserTask_H

/*
 * \file EBLaserTask.h
 *
 * $Date: 2012/04/27 13:46:00 $
 * $Revision: 1.39 $
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
edm::InputTag EBDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;
std::vector<int> laserWavelengths_;

MonitorElement* meShapeMapL1_[36];
MonitorElement* meAmplMapL1_[36];
MonitorElement* meTimeMapL1_[36];
MonitorElement* meAmplPNMapL1_[36];
MonitorElement* mePnAmplMapG01L1_[36];
MonitorElement* mePnPedMapG01L1_[36];
MonitorElement* mePnAmplMapG16L1_[36];
MonitorElement* mePnPedMapG16L1_[36];

MonitorElement* meShapeMapL2_[36];
MonitorElement* meAmplMapL2_[36];
MonitorElement* meTimeMapL2_[36];
MonitorElement* meAmplPNMapL2_[36];
MonitorElement* mePnAmplMapG01L2_[36];
MonitorElement* mePnPedMapG01L2_[36];
MonitorElement* mePnAmplMapG16L2_[36];
MonitorElement* mePnPedMapG16L2_[36];

MonitorElement* meShapeMapL3_[36];
MonitorElement* meAmplMapL3_[36];
MonitorElement* meTimeMapL3_[36];
MonitorElement* meAmplPNMapL3_[36];
MonitorElement* mePnAmplMapG01L3_[36];
MonitorElement* mePnPedMapG01L3_[36];
MonitorElement* mePnAmplMapG16L3_[36];
MonitorElement* mePnPedMapG16L3_[36];

MonitorElement* meShapeMapL4_[36];
MonitorElement* meAmplMapL4_[36];
MonitorElement* meTimeMapL4_[36];
MonitorElement* meAmplPNMapL4_[36];
MonitorElement* mePnAmplMapG01L4_[36];
MonitorElement* mePnPedMapG01L4_[36];
MonitorElement* mePnAmplMapG16L4_[36];
MonitorElement* mePnPedMapG16L4_[36];

 MonitorElement* meAmplSummaryMapL1_;
 MonitorElement* meAmplSummaryMapL2_;
 MonitorElement* meAmplSummaryMapL3_;
 MonitorElement* meAmplSummaryMapL4_;

bool init_;

};

#endif
