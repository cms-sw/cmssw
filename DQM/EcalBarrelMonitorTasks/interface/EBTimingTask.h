#ifndef EBTimingTask_H
#define EBTimingTask_H

/*
 * \file EBTimingTask.h
 *
 * $Date: 2008/02/29 15:03:59 $
 * $Revision: 1.8 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBTimingTask: public edm::EDAnalyzer{

public:

/// Constructor
EBTimingTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTimingTask();

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

MonitorElement* meTimeMap_[36];
MonitorElement* meTimeAmpli_[36];

bool init_;

};

#endif
