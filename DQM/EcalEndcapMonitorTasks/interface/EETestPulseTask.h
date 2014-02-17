#ifndef EETestPulseTask_H
#define EETestPulseTask_H

/*
 * \file EETestPulseTask.h
 *
 * $Date: 2012/04/27 13:46:13 $
 * $Revision: 1.17 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EETestPulseTask: public edm::EDAnalyzer{

public:

/// Constructor
EETestPulseTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETestPulseTask();

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
std::vector<int> MGPAGains_;
std::vector<int> MGPAGainsPN_;

MonitorElement* meShapeMapG01_[18];
MonitorElement* meShapeMapG06_[18];
MonitorElement* meShapeMapG12_[18];

MonitorElement* meAmplMapG01_[18];
MonitorElement* meAmplMapG06_[18];
MonitorElement* meAmplMapG12_[18];

MonitorElement* mePnAmplMapG01_[18];
MonitorElement* mePnAmplMapG16_[18];

MonitorElement* mePnPedMapG01_[18];
MonitorElement* mePnPedMapG16_[18];

// Quality check on crystals, one per each gain

float amplitudeThreshold_;

bool init_;

};

#endif
