#ifndef EETimingTask_H
#define EETimingTask_H

/*
 * \file EETimingTask.h
 *
 * $Date: 2011/09/15 21:54:51 $
 * $Revision: 1.22 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

class MonitorElement;
class DQMStore;

class EETimingTask: public edm::EDAnalyzer{

public:

/// Constructor
EETimingTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETimingTask();

static const float shiftProf2D;

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

edm::InputTag EcalRawDataCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag L1GtEvmReadoutRecord_;

MonitorElement* meTime_[18];
MonitorElement* meTimeMap_[18];
MonitorElement* meTimeAmpli_[18];

MonitorElement* meTimeAmpliSummary_[2];
MonitorElement* meTimeSummary1D_[2];
 MonitorElement* meTimeSummaryMap_[2];
MonitorElement* meTimeDelta_, *meTimeDelta2D_;

edm::ESHandle<CaloGeometry> pGeometry_;

bool init_;
bool initCaloGeometry_;

bool useBeamStatus_;
bool stableBeamsDeclared_;

};

const float EETimingTask::shiftProf2D = 50.;

#endif
