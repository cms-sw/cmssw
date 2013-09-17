#ifndef EBTimingTask_H
#define EBTimingTask_H

/*
 * \file EBTimingTask.h
 *
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

class MonitorElement;
class DQMStore;

class EBTimingTask: public edm::EDAnalyzer{

public:

/// Constructor
EBTimingTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTimingTask();

static const float shiftProf2D_;

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

float energyThreshold_;

edm::EDGetTokenT<EcalRawDataCollection> EcalRawDataCollection_;
edm::EDGetTokenT<EcalRecHitCollection> EcalRecHitCollection_;
edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> L1GtEvmReadoutRecord_;

MonitorElement* meTime_[36];
MonitorElement* meTimeMap_[36];
MonitorElement* meTimeAmpli_[36];

MonitorElement* meTimeAmpliSummary_;
MonitorElement* meTimeSummary1D_;
MonitorElement* meTimeSummaryMap_, *meTimeSummaryMapProjEta_, *meTimeSummaryMapProjPhi_;

bool init_;

bool useBeamStatus_;
bool stableBeamsDeclared_;

};

const float EBTimingTask::shiftProf2D_ = 50.;

#endif
