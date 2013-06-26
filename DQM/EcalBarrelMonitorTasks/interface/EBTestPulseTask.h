#ifndef EBTestPulseTask_H
#define EBTestPulseTask_H

/*
 * \file EBTestPulseTask.h
 *
 * $Date: 2012/04/27 13:46:00 $
 * $Revision: 1.36 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBTestPulseTask: public edm::EDAnalyzer{

public:

/// Constructor
EBTestPulseTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTestPulseTask();

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
std::vector<int> MGPAGains_;
std::vector<int> MGPAGainsPN_;

MonitorElement* meShapeMapG01_[36];
MonitorElement* meShapeMapG06_[36];
MonitorElement* meShapeMapG12_[36];

MonitorElement* meAmplMapG01_[36];
MonitorElement* meAmplMapG06_[36];
MonitorElement* meAmplMapG12_[36];

MonitorElement* mePnAmplMapG01_[36];
MonitorElement* mePnAmplMapG16_[36];

MonitorElement* mePnPedMapG01_[36];
MonitorElement* mePnPedMapG16_[36];

// Quality check on crystals, one per each gain

float amplitudeThreshold_;

bool init_;

};

#endif
