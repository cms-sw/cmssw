#ifndef EERawDataTask_H
#define EERawDataTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EERawDataTask : public edm::EDAnalyzer {

 public:

/// Constructor
EERawDataTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EERawDataTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

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

edm::InputTag FEDRawDataCollection_;
edm::InputTag EcalRawDataCollection_;

MonitorElement* meEECRCErrors_;

MonitorElement* meEERunNumberErrors_;
MonitorElement* meEEL1AErrors_;
MonitorElement* meEEOrbitNumberErrors_;
MonitorElement* meEEBunchCrossingErrors_;
MonitorElement* meEETriggerTypeErrors_;

bool init_;

};

#endif
