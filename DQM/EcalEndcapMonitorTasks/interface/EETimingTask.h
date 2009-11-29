#ifndef EETimingTask_H
#define EETimingTask_H

/*
 * \file EETimingTask.h
 *
 * $Date: 2009/10/26 17:33:51 $
 * $Revision: 1.11 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EETimingTask: public edm::EDAnalyzer{

public:

/// Constructor
EETimingTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETimingTask();

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
edm::InputTag EcalUncalibratedRecHitCollection_;

MonitorElement* meTime_[18];
MonitorElement* meTimeMap_[18];
MonitorElement* meTimeAmpli_[18];

MonitorElement* meTimeSummary1D_[2];
MonitorElement* meTimeSummaryMap_[2], *meTimeSummaryMapProjR_[2], *meTimeSummaryMapProjPhi_[2];
MonitorElement* meTimeDelta_;

bool init_;

};

#endif
