#ifndef EETimingTask_H
#define EETimingTask_H

/*
 * \file EETimingTask.h
 *
 * $Date: 2008/02/29 15:07:54 $
 * $Revision: 1.6 $
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
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dbe_;

bool enableCleanup_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;

MonitorElement* meTimeMap_[18];
MonitorElement* meTimeAmpli_[18];

bool init_;

};

#endif
