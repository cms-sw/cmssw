#ifndef EECosmicTask_H
#define EECosmicTask_H

/*
 * \file EECosmicTask.h
 *
 * $Date: 2008/01/09 11:03:29 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DaqMonitorBEInterface;

class EECosmicTask: public edm::EDAnalyzer{

public:

/// Constructor
EECosmicTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EECosmicTask();

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

DaqMonitorBEInterface* dbe_;

bool enableCleanup_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EcalUncalibRecHitCollection_; 
edm::InputTag EcalRecHitCollection_;

MonitorElement* meCutMap_[18];

MonitorElement* meSelMap_[18];

MonitorElement* meSpectrumMap_[18];

double lowThreshold_;
double highThreshold_;
double minJitter_;
double maxJitter_;

bool init_;

};

#endif
