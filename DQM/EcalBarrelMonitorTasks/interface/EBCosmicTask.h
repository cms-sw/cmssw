#ifndef EBCosmicTask_H
#define EBCosmicTask_H

/*
 * \file EBCosmicTask.h
 *
 * $Date: 2008/02/04 19:41:10 $
 * $Revision: 1.29 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DaqMonitorBEInterface;

class EBCosmicTask: public edm::EDAnalyzer{

public:

/// Constructor
EBCosmicTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBCosmicTask();

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
edm::InputTag EcalUncalibratedRecHitCollection_;
edm::InputTag EcalRecHitCollection_;

MonitorElement* meCutMap_[36];

MonitorElement* meSelMap_[36];

MonitorElement* meSpectrum_[2][36];

double lowThreshold_;
double highThreshold_;
double minJitter_;
double maxJitter_;

bool init_;

};

#endif
